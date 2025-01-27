import numpy as np
import matplotlib.pyplot as plt


def make_PCA_plot_n_components_validation(pca, cumulative_variance, n_components_90,
                                          n_components_95, name, model, data_name, tolerance=None):
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, n_components_95 + 1), pca.explained_variance_ratio_[:n_components_95], color='blue', alpha=0.8)
    plt.step(range(1, n_components_95 + 1), cumulative_variance[:n_components_95], where='mid',
             label='Cumulative explained variance')
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% explained variance')
    plt.axvline(x=n_components_90, color='g', linestyle='--', label=f'n_components = {n_components_90}')

    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('Number of principal components')
    plt.ylabel('Fraction of variance explained')
    plt.legend(loc='best')
    if tolerance is None:
        plt.savefig(f"PCA Images/{data_name}/PCA_{name}_{model}.png")
    else:
        plt.savefig(f"PCA Images/{data_name}/PCA_{name}_{model}_{tolerance}.png")
