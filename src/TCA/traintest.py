
from metrics import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression as PLSR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
pd.set_option('display.float_format', '{:.2f}'.format)


def train(X_train, y_train, modelnames=['PLSR'], plsr_comp=12):
    X_train   = X_train.astype(np.float32)
    y_train   = np.array(y_train).astype(np.float32).reshape(-1, 1) #
    
    models = []
    for k in modelnames:
        if k == 'PLSR':
            # PLSR
            plsr_comp = min(plsr_comp, X_train.shape[1])
            m = PLSR(n_components=plsr_comp)
            m.fit(X_train, y_train)
        # add other regression models
        
        
        models.append(m)
    return models

    

def test(models, Xs_test, ys_test, Xt_test, yt_test,  modelnames=['PLSR'], figure=True):
    
    Xs_test = Xs_test.astype(np.float32)
    ys_test = np.array(ys_test).astype(np.float32).reshape(-1, 1) #
    
    Xt  = Xt_test.astype(np.float32)
    yt = np.array(yt_test).astype(np.float32).reshape(-1, 1)
    
    
    results = []
    metrics = []
    yhats  = []
    # test on source domain
    for j in ['Source', 'Target']:
        if j == 'Source':
            Xtest = Xs_test.copy()
            ytest = ys_test.copy()
        else:
            Xtest = Xt.copy()
            ytest = yt.copy()
            
        for k in range(len(models)):
            
            yhat = np.maximum(models[k].predict(Xtest), 0)
            if modelnames[k]=='PLSR':
                yhats.append(yhat)
            M = Metrics(ytest, yhat)
            metrics.append(M)
            results.append([ j, modelnames[k], np.round(M.R2(), 2), np.round(M.RMSE(), 1), np.round(M.RMSEP(), 1), np.round(M.RPD(), 1)])
    results = np.array(results)
    if figure:
        colors = ['cadetblue',  'lightcoral']
        fig, ax = plt.subplots(1,2, figsize=(8,3.5))
        for k in range(len(results)):
            #plot
            ax[k].scatter(metrics[k].y, metrics[k].yhat, color= colors[k],label=f"{results[k, 0]}-{results [k, 1]}")
            ax[k].plot(metrics[k].x_plain(), metrics[k].y_reg(), "steelblue")
            ax[k].set_xlabel("Actual [g/L]")
            if k==0:
                ax[k].set_ylabel("Predicted [g/L]")
            # ax[k].legend(frameon=False)
            ax[k].spines['left'].set_color('gray')
            ax[k].spines['bottom'].set_color('gray')
    
        plt.tight_layout()
        plt.savefig("./Figures/Prediction_results.jpeg", dpi=600)
        plt.show()
  
    results = pd.DataFrame(results, columns=['Test on', 'Model', 'R-sq', 'RMSE', 'RMSEP [%]', 'RPD'])
    
    return results




if __name__=="__main__":
    pass