import numpy as np
import shap
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import mean_squared_error as MSE
from data_load import DataLoad # custom data loading module
from metrics import Metrics # custom metrics module

class ShapDA:
    """
    Shapley Additive explanations-based Domain Adaptation (ShapDA)
    Implements the full procedure from the paper:
    Citation (bibtex):
    @article{BABOR2026108477,
    title = {Interpretable domain adaptation enables robust lactic acid fermentation monitoring from waste},
    journal = {Results in Engineering},
    volume = {29},
    pages = {108477},
    year = {2026},
    issn = {2590-1230},
    doi = {https://doi.org/10.1016/j.rineng.2025.108477},
    url = {https://www.sciencedirect.com/science/article/pii/S2590123025045219},
    author = {Majharulislam Babor and Shanghua Liu and Arman Arefi and Agata Olszewska-Widdrat and Joachim Venus and Barbara Sturm and Marina M.-C. Höhne},
    keywords = {Out-of-distribution, Domain adaptation, Invariant features, FTIR, Explainable AI, Model interpretation},
    }
    full reference:
    Majharulislam Babor, Shanghua Liu, Arman Arefi, Agata Olszewska-Widdrat, Joachim Venus, Barbara Sturm, Marina M.-C. Höhne,
    Interpretable domain adaptation enables robust lactic acid fermentation monitoring from waste,
    Results in Engineering,
    Volume 29, 2026, 108477, ISSN 2590-1230,
    https://doi.org/10.1016/j.rineng.2025.108477.
    (https://www.sciencedirect.com/science/article/pii/S2590123025045219)
    
    - Train baseline PLSR on labeled source data.
    - Compute permutation SHAP on a balanced source+target explanation set
      using a small source reference (background).
    - Rank features by mean absolute SHAP (MAS).
    - Build Mahalanobis domain-alignment curve as features are added.
    - Use mean MD threshold + mean jump rule to select domain-invariant features (DIF).
    - Retrain PLSR using only DIF and use it for prediction in both domains.
    """

    def __init__(
        self,
        k_max=100,
        n_components_grid=range(2, 30),   # for PLS CV
        max_evals=7000,
        random_state=42,
        verbose =False
    ):
        self.k_max = k_max
        self.n_components_grid = list(n_components_grid)
        self.max_evals = max_evals
        self.random_state = random_state

        # Will be filled after fit()
        self.scaler_ = None
        self.baseline_pls_ = None
        self.dif_indices_ = None
        self.shap_values_ = None
        self.mas_ = None
        self.md_curve_ = None
        self.md_threshold_ = None
        self.md_jump_diff_ = None
        self.md_jump_threshold_ = None
        self.cutoff_k_ = None
        self.shap_background_idx_ = None
        self.shap_background_X_ = None
        self.final_pls_ = None   # PLS model trained on DIF
        self.feature_names_ = None
        self.verbose = verbose
    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def fit(
        self,
        Xs_train,
        ys_train,
        Xt_da,
        X_background,
        Xs_da = None,
        feature_names=None,
       
    ):
        """
        Fit ShapDA.

        Parameters
        ----------
        Xs_train : (n_s_train, d)
            Labeled source training spectra (used to train baseline and final PLS).
        ys_train : (n_s_train,)
            Source labels (e.g., glucose, lactic acid).
        Xs_da : (n_s_all, d)
            All source spectra used for SHAP explanation (can be all batches or train+val).
        Xt_da : (n_t_da, d)
            Unlabeled target spectra used only for SHAP explanation (X_t,da in paper).
        X_background : (n_bg, d)
            Small background set from source. Used as SHAP reference.
        feature_names : list[str] or None
            Optional feature names (e.g., wavenumbers as strings).
        """
        if Xs_da is None:
            Xs_da = Xs_train
        self.k_max = min(self.k_max, Xs_train.shape[1])
        # -------- Step 0: store metadata ----------
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(Xs_train.shape[1])]
        elif len(feature_names) != Xs_train.shape[1]:
            raise ValueError("feature_names length does not match number of features in Xs_train.")
        self.feature_names_ = feature_names
        self.shap_background_X_ = X_background
        self.shap_background_idx_ = None  # if needed

        # -------- Step 1: scale all data using source TRAIN scaler ----------
        if self.verbose:
            print("\n[ShapDA Step 1]: Scaling data...")
        self.scaler_ = StandardScaler()
        Xs_train_sc = self.scaler_.fit_transform(Xs_train)
        Xs_all_sc   = self.scaler_.transform(Xs_da)
        Xt_da_sc    = self.scaler_.transform(Xt_da)
        X_bg_sc     = self.scaler_.transform(X_background)

        # -------- Step 2: train baseline PLSR on full source TRAIN ----------
        if self.verbose:
            print("\n[ShapDA Step 2]: Training baseline PLSR...")
        self.baseline_pls_, self.n_comp_baseline_pls = self._fit_pls_with_1se(
            Xs_train_sc, ys_train, [i for i in self.n_components_grid if i <= Xs_train_sc.shape[1]]
        )

        # -------- Step 3: build balanced explanation set X_expl = [Xs_all, Xt_da] ----------
        if self.verbose:
            print("\n[ShapDA Step 3]: Building explanation set for SHAP...")
        X_expl_sc = np.vstack([Xs_all_sc, Xt_da_sc])
        if self.verbose:
            print(f"Explanation set shape: {X_expl_sc.shape}, with {Xs_all_sc.shape[0]} source and {Xt_da_sc.shape[0]} target samples.")
        # -------- Step 4: compute permutation SHAP values ----------
        if self.verbose:
            print("\n[ShapDA Step 4]: Computing SHAP values...")
        shap_vals = self._compute_shap_values(
            model=self.baseline_pls_,
            X_ref=X_bg_sc,
            X_expl=X_expl_sc,
        )
        # shap_vals.values shape: (n_expl, d)
        self.shap_values_ = shap_vals

        # -------- Step 5: compute MAS and rank features ----------
        if self.verbose:
            print("\n[ShapDA Step 5]: Computing MAS and ranking features...")
        self.mas_ = np.mean(np.abs(shap_vals.values), axis=0)  # (d,)
        # rank_idx[0] = most important feature
        rank_idx = np.argsort(self.mas_)[::-1]

        # -------- Step 6: build Mahalanobis domain-alignment curve ----------
        if self.verbose:
            print("\n[ShapDA Step 6]: Building Mahalanobis distance curve...")
        # Domain-specific matrices, aligned with X_expl
        n_s_all = Xs_all_sc.shape[0]
        Xs_expl_sc = X_expl_sc[:n_s_all]
        Xt_expl_sc = X_expl_sc[n_s_all:]

        md_curve = self._compute_md_curve(Xs_expl_sc, Xt_expl_sc, rank_idx)
        self.md_curve_ = md_curve

        # -------- Step 7: compute MD threshold and mean-jump cutoff ----------
        if self.verbose:
            print("\n[ShapDA Step 7]: Determining Domain-Invariant Feature (DIF) cutoff...")
        # 1) Mean jump threshold
        jump_threshold = np.mean(self.diff_md) 
        
        # Find indices where the distance rises significantly (above jump threshold)
        jumps_idx = np.where(self.diff_md > jump_threshold)[0] 
        # we need idx where jumps started (if kth feature caused a jump in MD/misalignment, k-1 features had better domain alignment)
        jumps_idx = jumps_idx - 1
    
            # 2) Mean reference (MD) line for horizontal threshold
        mean_md = self.md_.mean()
        
            # 3) Highlight significant jumps
        k_jumps = self.k[jumps_idx] # at which k we find jumps
        md_jumps = self.md_[jumps_idx]
            # 4) Domain-invariant cutoff line (k = last jump below threshold)
        self.dif_cutoff_k = max(i for i, j in zip(k_jumps, md_jumps) if j < mean_md).astype(int)
    
        if self.verbose: 
            print(f"ShapDA selected {self.dif_cutoff_k} domain-invariant features (DIFs).")
        self.dif_indices_ = rank_idx[:self.dif_cutoff_k]

        # -------- Step 8: retrain final PLSR on DIFs only ----------
        if self.verbose:
            print("\n[ShapDA Step 8]: Training final PLSR on DIF features...")
        Xs_train_dif = Xs_train_sc[:, self.dif_indices_]
        self.final_pls_, self.n_comp_final_pls = self._fit_pls_with_1se(
            Xs_train_dif, ys_train, [i for i in self.n_components_grid if i <= self.dif_cutoff_k]
        )
        if self.verbose:
            print("\n::::ShapDA fitting complete::::")
        return self

    def transform(self, X):
        """
        Apply ShapDA feature selection (DIF) + scaling to new data.

        Parameters
        ----------
        X : (n_samples, d)
            New spectra (source or target).

        Returns
        -------
        X_dif : (n_samples, k)
            Scaled spectra restricted to DIF subset.
        """
        if self.scaler_ is None or self.dif_indices_ is None:
            raise RuntimeError("ShapDA must be fitted before calling transform().")
        X_sc = self.scaler_.transform(X)
        return X_sc[:, self.dif_indices_]

    def predict(self, X):
        """
        Predict using the ShapDA final PLSR model on DIF features.

        Parameters
        ----------
        X : (n_samples, d)
            New spectra.

        Returns
        -------
        y_pred : (n_samples,)
        """
        if self.final_pls_ is None:
            raise RuntimeError("ShapDA must be fitted before calling predict().")
        X_dif = self.transform(X)
        return self.final_pls_.predict(X_dif).ravel()

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------
    def _fit_pls_with_1se(self, X, y, ncomp_grid):
        """
        Fit PLSRegression with 1-SE rule on source training data.

        X: (n_samples, d)
        y: (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        mean_rmse = []
        std_rmse = []

        for n_comp in ncomp_grid:
            rmses = []
            for tr_idx, val_idx in kf.split(X, y):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_tr, y_tr)
                y_pred = pls.predict(X_val).ravel()
                rmses.append(np.sqrt(MSE(y_val, y_pred)))
            mean_rmse.append(np.mean(rmses))
            std_rmse.append(np.std(rmses))

        mean_rmse = np.array(mean_rmse)
        std_rmse = np.array(std_rmse)

        # 1-SE rule: choose smallest n_comp whose RMSE <= (min + SE_of_best)
        best_idx = np.argmin(mean_rmse)
        best_mean = mean_rmse[best_idx]
        best_se = std_rmse[best_idx] / np.sqrt(kf.get_n_splits())
        threshold = best_mean + best_se

        candidates = np.where(mean_rmse <= threshold)[0]
        chosen_idx = candidates[0]  # smallest number of components under threshold
        chosen_n_comp = ncomp_grid[chosen_idx]
        # if self.verbose:
            # print(f"Selected n_components = {chosen_n_comp} via 1-SE rule.")
        pls_final = PLSRegression(n_components=chosen_n_comp)
        pls_final.fit(X, y)
        return pls_final, chosen_n_comp

    def _compute_shap_values(self, model, X_ref, X_expl):
        """
        Compute permutation SHAP values.

        model: fitted regressor with .predict(X) -> (n_samples,)
        X_ref: background / reference, shape (n_bg, d)
        X_expl: explanation set, shape (n_expl, d)
        """
        # SHAP expects a function f(X) -> predictions
        def f_predict(x):
            return model.predict(x).ravel()

        explainer = shap.Explainer(
            f_predict,
            X_ref,
            algorithm="permutation"
        )
        shap_values = explainer(X_expl, max_evals=self.max_evals)
        return shap_values

    def _compute_md_curve(self, Xs, Xt, rank_idx):
        """
        Compute Mahalanobis distance MD(k) as we add SHAP-ranked features.

        Xs: source spectra used for explanation (n_s, d)
        Xt: target spectra used for explanation (n_t, d)
        rank_idx: array of feature indices sorted by importance (desc)
        """
        Xs = np.asarray(Xs)
        Xt = np.asarray(Xt)
        d = Xs.shape[1]
        assert Xs.shape[1] == Xt.shape[1]
        
        md_values = np.zeros((self.k_max, 3))  # k, MD(k), diff_MD(k)
        for k in range(1, self.k_max + 1):
            idx_k = rank_idx[:k]
            Xs_k = Xs[:, idx_k]
            Xt_k = Xt[:, idx_k]

            m_dist = self.mahalanobis_dist(Xs_k, Xt_k)
            md_values[k-1, :2] = np.array([len(idx_k), m_dist]) # if we take :k top features, produces m_dist
            # mahal_results.append((k, m_dist))
        # Compute first-order differences (gradient of Mahalanobis distance)
        md_values[1:, 2] = np.diff(md_values[:, 1])
        
        self.k, self.md_, self.diff_md = md_values[:,0], md_values[:,1], md_values[:,2]
        

    def mahalanobis_dist(self, Xs_k, Xt_k):
        # Mahalanobis distance calculation function for k features
        # Stack both domains
        X_combined = np.vstack([Xs_k, Xt_k])
        
        # Estimate full covariance matrix (shared)
        cov = EmpiricalCovariance().fit(X_combined)
        cov_inv = cov.precision_

        # Mean vectors
        mu_s = np.mean(Xs_k, axis=0)
        mu_t = np.mean(Xt_k, axis=0)

        # Mahalanobis distance between means
        delta = mu_s - mu_t
        dist = np.sqrt(delta.T @ cov_inv @ delta)
        return dist
    

if __name__ == "__main__":
    
    # Example usage:
    # load spectra data Xs_train, ys_train, Xt_da, Xs_val, ys_val, Xt_val
    D = DataLoad(dataset='glucose')
    D.load_data()
    # source, target data and wavelengths (features)
    Xs, ys, Xt, yt, wl = D.X_source, D.y_source, D.X_target, D.y_target, D.wl
    ys_train, Xs_train = ys[:214], Xs[:214]
    ys_test,   Xs_test   = ys[214:], Xs[214:]
    Xt_da =  Xt[:len(Xs)]        # unlabeled (for adaptation only) similar size as source data
    # Xs contains the full source data used for adaptation (without labels),
    # while Xs_train is used exclusively to fit the base and final models (with labels).
    shapda_model = ShapDA(verbose=1)
    shapda_model.fit(Xs_train, ys_train, 
                     Xt_da=Xt_da, Xs_da=Xs, 
                     X_background=Xs_train[:4], 
                     feature_names=[f"{i}" " $cm^{-1}$" for i in wl])
    print("\n::::ShapDA Performance on Target Domain::::")
    yt_pred = shapda_model.predict(Xt)
    mt = Metrics(yt, yt_pred)
    print(f"ShapDA Target Test RMSE: {mt.RMSE():.1f}, R2: {mt.R2():.2f}, RRMSE: {mt.RMSEP():.1f}, RPD: {mt.RPD():.1f}")
    
    

