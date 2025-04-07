
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
            results.append([ j, modelnames[k], np.round(M.R2(), 2), np.round(M.RMSE(), 5), np.round(M.RMSEP(), 5), np.round(M.RPD(), 1)])
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


def classification(X_train, X_test, y_train, y_test, wl):
   
    clf = LDA(n_components=1, solver='svd', store_covariance=False, tol=0.01)
    clf.fit(X_train, y_train)


    # Predict the probabilities of the test set
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # print(f'The AUC score is {auc_score:.2f}')

    # Predict the classes of the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # print(f'The accuracy of the SVM model is {accuracy:.2f}')

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    X_train_transformed = clf.decision_function(X_train)
    X_test_transformed  = clf.decision_function(X_test)

    return auc_score, accuracy, fpr, tpr, X_train_transformed, X_test_transformed

def classification_results_visualisation(dataset1):
    # print(f"===Classification using {model}: only raw spectra===")
    X_train1, X_test1, y_train1, y_test1, wl1 = dataset1
    
    auc_score1, accuracy1, fpr1, tpr1, X_train_transformed1, X_test_transformed1 = \
            classification(X_train1, X_test1, y_train1, y_test1, wl1)
    
    fig, ax = plt.subplots(1,2,figsize=(7,3))
    ax[1].plot(fpr1, tpr1, label=f'AUC = {auc_score1:.2f}', alpha=0.5)
    ax[1].plot([0, 1], [0, 1], linestyle='--', alpha=0.5)
    ax[1].set_xlabel('False positive rate')
    ax[1].set_ylabel('True positive rate')
    ax[1].text(0.64, 0.1, f'ACC = {accuracy1*100:.0f} %')
    ax[1].legend(frameon=False)
    # ax[0].set_title('ROC Curve'
    ax[0].scatter(y_test1, X_test_transformed1, color='crimson', facecolor='None',label='Test set', alpha=0.5)
    ax[0].scatter(y_train1, X_train_transformed1, 5,color='indigo', label='Training set', alpha=0.5)
    ax[0].set_xlabel('Domains')
    ax[0].set_ylabel("Linear Discriminant [1]")
    ax[0].set_xlim(-0.5,1.5)
    ax[0].set_xticks([0, 1], ['SG', 'CW'])
    ax[0].legend(frameon=False, ) #loc='upper left'

    # ax[3].label_outer()
    for axis in ['top','right']:
        ax[0].spines[axis].set_color('white')
        ax[1].spines[axis].set_color('white')


    for axis in ['bottom','left']:
        ax[0].spines[axis].set_linewidth(1.5)
        ax[0].spines[axis].set_color('darkgrey')
        # ax[0].set_facecolor("whitesmoke")
        ax[1].spines[axis].set_linewidth(1.5)
        ax[1].spines[axis].set_color('darkgrey')
        # ax[1].set_facecolor("whitesmoke")
   
    # plt.subplots_adjust(wspace=.05, hspace=0)
    plt.tight_layout()
    plt.show()

    return None



if __name__=="__main__":
    pass