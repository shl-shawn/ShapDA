
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
from sklearn.covariance import EmpiricalCovariance

np.set_printoptions(precision=2)
plt.rcParams["font.family"] = 'menlo'#"Times New Roman"
plt.rcParams["font.size"] = 23
plt.style.use('ggplot')#ggplot
plt.rcParams['axes.facecolor']='white'
# plt.rcParams['xtick.color']='orangered'


# plt.rcParams['xtick.color']='orangered'
import warnings
warnings.filterwarnings('ignore')



def mahalanobis_dist(Xs_k, Xt_k):
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



def mahalanobis_distance(x, mean, inv_cov):
    # Function to calculate Mahalanobis distance from mean and inverse covariance matrix
    return mahalanobis(x, mean, inv_cov)


def calculate_mahalanobis_distance(X_source, X_target):
    # Calculate the mean and covariance matrix of the source data
    mean_source = np.mean(X_source, axis=0)
    cov_source = np.cov(X_source, rowvar=False)
    # if np.linalg.det(cov_source) == 0:
    #     print("Covariance matrix is singular. Considering regularizing or reducing dimensionality.")
    cov_source += np.eye(cov_source.shape[0]) * 1e-10
    inv_cov_source = inv(cov_source)

    # Calculate Mahalanobis distance for each instance in the source data
    distances_source = [mahalanobis_distance(x, mean_source, inv_cov_source) for x in X_source]

    # Calculate Mahalanobis distance for each instance in the target data
    distances_target = [mahalanobis_distance(x, mean_source, inv_cov_source) for x in X_target]
    return distances_source, distances_target


def visualize_tsne_with_mahal_dist(Xs, Xt, mahal_dist_source, mahal_dist_target, title='tSNE_raw_data'):
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(np.vstack((Xs, Xt)))
    N = len(Xs)

    fig, ax = plt.subplots(1,2,figsize=(8,3.5))
    ax[0].scatter(X_reduced[N:,0], X_reduced[N:,1], 15, c='lightcoral', alpha=0.9, marker='x', edgecolors='none')#c=y, cmap='cool',
    ax[0].scatter(X_reduced[:N,0], X_reduced[:N,1], 15, c='cadetblue', alpha=1, marker='o', edgecolors='none')
    ax[0].set_xlabel("t-SNE 1")
    ax[0].set_ylabel("t-SNE 2")
    ax[0].spines['left'].set_color('gray')
    ax[0].spines['bottom'].set_color('gray')

    
    # color data
    values = np.append(mahal_dist_source,mahal_dist_target)  # Random values between 0 and 100
    # Create a colormap
    cmap = plt.get_cmap('terrain')
    # Normalize the values to the range [0, 1]
    norm = Normalize(vmin=np.min(values), vmax=np.max(values))
    # Map the normalized values to colors
    colors = cmap(norm(values))
    
    # scatter2 = ax[3].scatter(X_reduced[:,0], X_reduced[:,2], 18, alpha=1, marker='o', c=np.append(mahal_dist_source,mahal_dist_target ), cmap='terrain')
    scatter1 = ax[1].scatter(X_reduced[N:,0], X_reduced[N:,1], 15, alpha=1, marker='x', c=colors[N:], cmap='terrain', edgecolors='none')
    scatter2 = ax[1].scatter(X_reduced[:N,0], X_reduced[:N,1], 15, alpha=1, marker='o', c=colors[:N], cmap='terrain', edgecolors='none')
    # Add a colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[1])
    cbar.set_label('Mahalanobis Distance [1]', fontsize=12)
    ax[1].set_xlabel("t-SNE 1")
    ax[1].set_ylabel("t-SNE 2")
    ax[1].spines['left'].set_color('gray')
    ax[1].spines['bottom'].set_color('gray')
    # ax[1].set_xticks([-40, -20, 0, 20, 40])
    # ax[1].set_yticks([-20, 0, 20])
    # cbar = fig.colorbar(scatter2, ax=ax[3], label='Mahalanobis Distance [1]')
    for k in range(2):
        ax[k].tick_params(top=False, right=False)
    # fig.colorbar(scatter, label='Mahalanobis Distance')
    plt.rcParams.update({'font.size': 15})  # Set the desired font size
    plt.rc('axes', titlesize=16)  # Font size for the title
    plt.rc('axes', labelsize=16)  # Font size for x and y labels
    fig.tight_layout()
    # plt.savefig(f"./Figures/{title}.jpeg", dpi=300)
    plt.show()
    return None



def Gaussian_kernel_density_estimation(xs, xt, label_s='SG', label_t='CW', single_figure=True, title='Gaussian KDE'):
    df_ss = pd.DataFrame(xs, columns=[label_s]).sort_values(label_s)
    df_cs = pd.DataFrame(xt, columns=[label_t]).sort_values(label_t)

    # Compute the kernel density estimate
    kde_ss = gaussian_kde(xs)
    kde_cs = gaussian_kde(xt)

    # Adjust the bandwidth of the KDE plot
    kde_ss.set_bandwidth(bw_method=0.4)
    kde_cs.set_bandwidth(bw_method=0.4)
    print(f'Mahalanobis distances')
    print('Gaussian kernel density estimation')
    # Create a KDE plot of a feature in your dataset with a smoother curve
    if single_figure:
        fig,ax = plt.subplots(figsize=(5, 3.5))
        # # ax.set_title('Gaussian kernel density estimation')

        ax.set_xlabel(f'Mahalanobis Distance [1]', fontsize=15)
        ax.set_ylabel('Density [M.D.$^{-1}$]', fontsize=15)
        ax.plot(df_ss[label_s], kde_ss(df_ss[label_s]), color='cadetblue', label=label_s)
        ax.fill_between(df_ss[label_s], kde_ss(df_ss[label_s]), color='cadetblue', alpha=0.24)
        ax.tick_params(axis='both', labelsize=14)
        ax.plot(df_cs[label_t], kde_cs(df_cs[label_t]), color = 'orangered', label=label_t)
        ax.fill_between(df_cs[label_t], kde_cs(df_cs[label_t]), color='orangered', alpha=0.24)
        
        for axis in ['top','right']:
            ax.spines[axis].set_color('white')
            # ax.x_ticks[axis].set_color('white')
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color('darkgrey')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True)) 
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        # ax.set_ylim(0, 72)
        ax.set_xlim(1, 30)
        ax.legend(frameon=False, fontsize=14)

    else:
        fig, ax = plt.subplots(1,2,figsize=(6, 3.5))

        ax[0].plot(df_ss[label_s], kde_ss(df_ss[label_s]), color='cadetblue', label=label_s)
        ax[0].fill_between(df_ss[label_s], kde_ss(df_ss[label_s]),color='cadetblue', alpha=0.24)
        ax[0].set_xticks([16.05, 16.15, 16.25])
        ax[0].set_ylim(0, 12)
        ax[0].tick_params(axis='both', labelsize=14)
        ax[1].plot(df_cs[label_t], kde_cs(df_cs[label_t]), color = 'orangered', label=label_t)
        ax[1].fill_between(df_cs[label_t], kde_cs(df_cs[label_t]), color='orangered', alpha=0.24)
        ax[1].set_ylim(0, 0.01)
        ax[1].set_yticks([0, 0.005, 0.01])
        ax[1].tick_params(axis='both', labelsize=14)
        ax[0].set_ylabel('Density [M.D.$^{-1}$]', fontsize=15)
        
        for k in range(2):
            ax[k].set_xlabel(f'Mahalanobis Distance [1]', fontsize=15)
            ax[k].legend(loc='upper right', frameon=False, fontsize=14)

        for axis in ['top','right']:
            ax[0].spines[axis].set_color('white')
            ax[1].spines[axis].set_color('white')
            
        for axis in ['bottom','left']:
            ax[0].spines[axis].set_linewidth(1.5)
            ax[0].spines[axis].set_color('darkgrey')
            ax[1].spines[axis].set_linewidth(1.5)
            ax[1].spines[axis].set_color('darkgrey')
  
    plt.grid(False)
    fig.tight_layout()
    fig.set_facecolor('white')
    plt.savefig(f"./Figures/feature_mahalanobis_KDE_{title}.jpeg", dpi=700)
    plt.show()
    return None

def find_domain_invariant_cutoff(Xs, Xt, sorted_indices_features):
    """
    Find the last k where the Mahalanobis distance is below the threshold.
    """

    # Store (k, distance)
    n = 100
    mahal_results = np.zeros((n, 3))

    for k in range(1, n+1):  # test from top 1 to 100 features
        topk_idx = sorted_indices_features[:k]
        
        Xs_k = Xs[:, topk_idx]
        Xt_k = Xt[:, topk_idx]
        
        m_dist = mahalanobis_dist(Xs_k, Xt_k)
        mahal_results[k-1, :2] = np.array([len(topk_idx), m_dist]) # if we take :k top features, produces m_dist
        # mahal_results.append((k, m_dist))
    # Compute first-order differences (gradient of Mahalanobis distance)
    mahal_results[1:, 2] = np.diff(mahal_results[:, 1])
    
    k, MD, diff_MD = mahal_results[:,0], mahal_results[:,1], mahal_results[:,2]
    
    jump_threshold = np.mean(diff_MD) 
    
    # Find indices where the distance rises significantly (above jump threshold)
    jumps_idx = np.where(diff_MD > jump_threshold)[0] 
    # we need idx where jumps started (if kth feature caused a jump in MD/misalignment, k-1 features had better domain alignment)
    jumps_idx = jumps_idx - 1
 
        # 2) Mean reference (MD) line for horizontal threshold
    mean_md = MD.mean()
    
        # 3) Highlight significant jumps
    k_jumps = k[jumps_idx] # at which k we find jumps
    MD_jumps = MD[jumps_idx]
        # 4) Domain-invariant cutoff line (k = last jump below threshold)
    domain_invariant_cutoff_k = max(i for i, j in zip(k_jumps, MD_jumps) if j < mean_md)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 4))

    # 1) Mahalanobis-distance curve
    ax.plot(
        mahal_results[:, 0], mahal_results[:, 1],
        marker='o', markersize=3,
        color='cadetblue',
        markerfacecolor='none',   # leaves the circle empty
        markeredgecolor='cadetblue', label='domain alignment curve'
    )

    ax.axhline(mean_md, color='gray', ls='--', alpha=0.4, label='threshold (mean MD)')


    ax.scatter(
        k_jumps, MD_jumps,
        marker='x', s=80,
        color='lightcoral',
        label='jump at k-th features',
        zorder=3
    )

    ax.axvline(domain_invariant_cutoff_k, color='black', ls=':', lw=1.5, alpha=0.6, label=f"cutoff k = {int(domain_invariant_cutoff_k)}")

    # Styling
    ax.set_xlabel('Important Features Count', fontsize=16)
    ax.set_ylabel('Mahalanobis distance [1]', fontsize=16   )
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.tick_params(axis='both', labelsize=15)  # You can adjust the value (e.g., 14, 16, etc.)
    ax.legend(frameon=False, fontsize=11, loc='right')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.savefig(f"./figures/domain_invariant_features_cut_off.jpeg", dpi=500)
    plt.show()
    return int(domain_invariant_cutoff_k), mahal_results, jumps_idx, mean_md
