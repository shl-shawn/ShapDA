
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

np.set_printoptions(precision=2)
plt.rcParams["font.family"] = 'menlo'#"Times New Roman"
plt.rcParams["font.size"] = 23
plt.style.use('ggplot')#ggplot
plt.rcParams['axes.facecolor']='white'
# plt.rcParams['xtick.color']='orangered'


# plt.rcParams['xtick.color']='orangered'
import warnings
warnings.filterwarnings('ignore')




def mahalanobis_distance(x, mean, inv_cov):
    # Function to calculate Mahalanobis distance
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
    tsne = TSNE(n_components=3, random_state=42)
    X_reduced = tsne.fit_transform(np.vstack((Xs, Xt)))
    N = len(Xs)


    # plt.rcParams["font.family"] = 'menlo'#"Times New Roman"
    # plt.rcParams["font.size"] = 18
    # plt.style.use('ggplot')#ggplot
    # plt.rcParams['axes.facecolor']='white'
    fig, ax = plt.subplots(1,4,figsize=(18,4))

    ax[0].scatter(X_reduced[N:,0], X_reduced[N:,1], 15, c='lightcoral', alpha=0.9, marker='x', edgecolors='none')#c=y, cmap='cool',
    ax[0].scatter(X_reduced[:N,0], X_reduced[:N,1], 15, c='cadetblue', alpha=1, marker='o', edgecolors='none')

    ax[0].set_xlabel("t-SNE 1")
    ax[0].set_ylabel("t-SNE 2")
    ax[0].spines['left'].set_color('gray')
    ax[0].spines['bottom'].set_color('gray')
    ax[0].set_xticks([-15, 0, 15 ])
    ax[0].set_yticks([-10, 0, 10 ])
    # ax[0].legend(["GS", "Cw"])

    ax[2].scatter(X_reduced[N:,0], X_reduced[N:,2], 15,  c='lightcoral', alpha=0.9, marker='x', edgecolors='none')
    ax[2].scatter(X_reduced[:N,0], X_reduced[:N,2], 15, c='cadetblue', alpha=1, marker='o', edgecolors='none')

    ax[2].set_xlabel("t-SNE 1")
    ax[2].set_ylabel("t-SNE 3")
    ax[2].spines['left'].set_color('gray')
    ax[2].spines['bottom'].set_color('gray')
    ax[2].set_xticks([-15, 0, 15 ])
    ax[2].set_yticks([-10, 0, 10 ])

    ax[1].scatter(X_reduced[N:,1], X_reduced[N:,2], 15, c='lightcoral', alpha=0.9,marker='x', edgecolors='orangered')
    ax[1].scatter(X_reduced[:N,1], X_reduced[:N,2], 15, c='cadetblue', alpha=1,marker='o', edgecolors='none')

    ax[1].set_xlabel("t-SNE 2")
    ax[1].set_ylabel("t-SNE 3")
    ax[1].spines['left'].set_color('gray')
    ax[1].spines['bottom'].set_color('gray')
    ax[1].set_xticks([-10, 0, 10 ])
    ax[1].set_yticks([-10, 0, 10])
    
    # color data
    values = np.append(mahal_dist_source,mahal_dist_target)  # Random values between 0 and 100
    # Create a colormap
    cmap = plt.get_cmap('terrain')
    # Normalize the values to the range [0, 1]
    norm = Normalize(vmin=np.min(values), vmax=np.max(values))
    # Map the normalized values to colors
    colors = cmap(norm(values))
    
    # scatter2 = ax[3].scatter(X_reduced[:,0], X_reduced[:,2], 18, alpha=1, marker='o', c=np.append(mahal_dist_source,mahal_dist_target ), cmap='terrain')
    scatter1 = ax[3].scatter(X_reduced[N:,0], X_reduced[N:,2], 15, alpha=1, marker='x', c=colors[N:], cmap='terrain', edgecolors='none')
    scatter2 = ax[3].scatter(X_reduced[:N,0], X_reduced[:N,2], 15, alpha=1, marker='o', c=colors[:N], cmap='terrain', edgecolors='none')
    # Add a colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[3])
    cbar.set_label('Mahalanobis Distance [1]')
    
    
    ax[3].set_xlabel("t-SNE 1")
    ax[3].set_ylabel("t-SNE 3")
    ax[3].spines['left'].set_color('gray')
    ax[3].spines['bottom'].set_color('gray')
    ax[3].set_xticks([-15, 0, 15 ])
    ax[3].set_yticks([-10, 0, 10 ])
    # cbar = fig.colorbar(scatter2, ax=ax[3], label='Mahalanobis Distance [1]')
    for k in range(4):
        ax[k].tick_params(top=False, right=False)
    # fig.colorbar(scatter, label='Mahalanobis Distance')
    plt.rcParams.update({'font.size': 15})  # Set the desired font size
    plt.rc('axes', titlesize=16)  # Font size for the title
    plt.rc('axes', labelsize=16)  # Font size for x and y labels



    
    
    fig.tight_layout()
    
    plt.savefig(f"./Figures/{title}.jpeg", dpi=300)
    plt.show()
    return None



def Gaussian_kernel_density_estimation(xs, xt, label_s='SG', label_t='CW', single_figure=True):
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
        fig,ax = plt.subplots(figsize=(4, 3.5))
        # # ax.set_title('Gaussian kernel density estimation')

        ax.set_xlabel(f'Mahalanobis Distance [1]')
        ax.set_ylabel('Density [M.D.$^{-1}$]')
        ax.plot(df_ss[label_s], kde_ss(df_ss[label_s]), color='cadetblue', label=label_s)
        ax.fill_between(df_ss[label_s], kde_ss(df_ss[label_s]), color='cadetblue', alpha=0.24)
        
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
        ax.set_xlim(1, 33)
    
    else:
        fig, ax = plt.subplots(1,2,figsize=(7, 3))

        ax[0].plot(df_ss[label_s], kde_ss(df_ss[label_s]), color='cadetblue', label=label_s)
        ax[0].fill_between(df_ss[label_s], kde_ss(df_ss[label_s]),color='cadetblue', alpha=0.24)
        
        ax[1].plot(df_cs[label_t], kde_cs(df_cs[label_t]), color = 'orangered', label=label_t)
        ax[1].fill_between(df_cs[label_t], kde_cs(df_cs[label_t]), color='orangered', alpha=0.24)
        

        ax[0].set_ylabel('Density [M.D.$^{-1}$]')
        for k in range(2):
            ax[k].set_xlabel(f'Mahalanobis Distance [1]')
            ax[k].xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax[k].legend(frameon=False)
            # Set maximum number of ticks on both x and y axes
            ax[k].xaxis.set_major_locator(MaxNLocator(nbins=4, )) 
            ax[k].yaxis.set_major_locator(MaxNLocator(nbins=4, ))
            ax[k].tick_params(top=False, right=False)

        for axis in ['top','right']:
            ax[0].spines[axis].set_color('white')
            ax[1].spines[axis].set_color('white')
        for axis in ['bottom','left']:
            ax[0].spines[axis].set_linewidth(1.5)
            ax[0].spines[axis].set_color('darkgrey')
            ax[1].spines[axis].set_linewidth(1.5)
            ax[1].spines[axis].set_color('darkgrey')
        
        
    plt.grid(False)
    plt.legend(frameon=False)
    fig.tight_layout()
    fig.set_facecolor('white')
    plt.savefig(f"./Figures/adapted_feature_mahalanobis_KDE.jpeg", dpi=700)
    plt.show()
    return None


