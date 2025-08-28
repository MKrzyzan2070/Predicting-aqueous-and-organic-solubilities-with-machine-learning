import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pickle
import json
import Code
import lightgbm as lgb
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import os
from matplotlib.patches import FancyArrow
from PIL import Image
import seaborn as sns
import matplotlib.colors as mcolors
from scipy import stats
from matplotlib.ticker import MaxNLocator

from Code.AqSolDB_molecule_solubility_prediction import feature_type_list

np.random.seed(1)

#########################
import logging
import warnings
logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
#########################


def KFold_validation_test_validation(path, make_pickle, tolerance, model, do_PCA, dataset):

    # Defining the Stratified KFold for the primary train/test splitting and the 5-fold cross validation later on
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    rmse_scores = []
    r2_scores = []
    mae_scores = []

    path = path.replace(".csv", f"_{tolerance}.csv")
    pattern = r'/([^/_]+(?:_[^/_]+)*)[^/]*?_\d+\.\d+\.csv'
    match = re.search(pattern, path)
    name = match.group(1)

    if make_pickle is True:
        # Loading the dataset
        solute_solvent_df = pd.read_csv(path)
        #na_indices = np.where(solute_solvent_df.isna())

        if dataset == "Combisolu":
            solute_solvent_df.rename(columns={"experimental_logS [mol/L]": "Solubility"}, inplace=True)
        elif dataset == "BigSolDB":
            solute_solvent_df.rename(columns={'experimental_logX': 'Solubility'}, inplace=True)

        # Creating the x-data and the y-data:
        if "index" in solute_solvent_df.columns:
            solute_solvent_df.drop(columns='index', inplace=True)
        if "Unnamed: 0" in solute_solvent_df.columns:
            solute_solvent_df.drop(columns='Unnamed: 0', inplace=True)
        solute_solvent_df.reset_index(drop=True, inplace=True)
        # Shuffling, just in case:
        solute_solvent_df = solute_solvent_df.sample(frac=1, random_state=1).reset_index(drop=True)

        solute_solvent_df_x = np.array(solute_solvent_df.drop(columns=["Unnamed: 0", "solute_smiles", "solvent_smiles",
                                                                       "Solubility", "solute_InChIKey", "solvent_InChIKey",
                                                                       "temperature", "index"], errors='ignore'))
        solute_solvent_df_y = np.array(solute_solvent_df["Solubility"])
        # print(solute_solvent_df_y)

        ################################################
        # The stratified train/test split:
        solute_solvent_df_y_binned = pd.qcut(solute_solvent_df_y, q=5, labels=False)
        ################################################
        # MinMax scaling the x data of the dataframe:
        scaler = MinMaxScaler()
        solute_solvent_df_x_scaled = scaler.fit_transform(solute_solvent_df_x)

        ################################################################################################
        ################################################################################################
        #### Performing the PCA:
        if do_PCA is True:
            # The initial PCA fit (finding the minimal number of n_components):
            pca = PCA()
            pca.fit(solute_solvent_df_x_scaled)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            # The minimum number of components:
            n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
            n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

            # Applying the PCA:
            pca_reduced = PCA(n_components=n_components_90)
            solute_solvent_df_x_scaled = pca_reduced.fit_transform(solute_solvent_df_x_scaled)

            # Making the PCA plot for validation:
            if dataset == "CombiSolu":
                Code.PCA_plot.make_PCA_plot_n_components_validation(pca, cumulative_variance, n_components_90,
                                                                    n_components_95, name, model, "CombiSolu",
                                                                    tolerance)
            elif dataset == "BigSolDB":
                Code.PCA_plot.make_PCA_plot_n_components_validation(pca, cumulative_variance, n_components_90,
                                                                    n_components_95, name, model, "BigSolDB",
                                                                    tolerance)
        ################################################################################################
        ################################################################################################

        # THE PRIMARY TRAIN/TEST SPLITTING:
        x_data_train_split, x_data_test_split, y_data_train_split, y_data_test_split = train_test_split(
            solute_solvent_df_x_scaled, solute_solvent_df_y, train_size=0.8,
            stratify=solute_solvent_df_y_binned, random_state=1
        )
        ###############################

        ################################################
        # The stratified 5-fold cross validation:
        y_data_train_split_binned = pd.qcut(y_data_train_split, q=5, labels=False)
        ################################################

        if dataset == "CombiSolu":
            dataset_name = "Combi_Solu"
        elif dataset == "BigSolDB":
            dataset_name = "BigSolDB"
        ###################### RandomForest ######################
        if model == "RF":
            # Training datasets the model (getting the best parameters):
            random_forest_model_fit, grid_search_r2_score = Code.Train_RF_model(kf, x_data_train_split, y_data_train_split_binned,
                                                                                name, y_data_train_split, dataset_name, tolerance)
            # Saving the model:
            with open(f"Pickled Models/{dataset_name}_Plot/Plot_{name}_{model}_{tolerance}.pkl", 'wb') as file:
                pickle.dump({
                    'model': random_forest_model_fit,
                }, file)
        ##########################################################

        ###################### LightGBM ######################
        elif model == "LightGBM":
            # Training datasets the model (getting the best parameters):
            lgbm_model_fit, grid_search_r2_score = Code.Train_LGBM_model(kf, x_data_train_split, y_data_train_split_binned,
                                                                         name, y_data_train_split, dataset_name, tolerance)
            with open(f"Pickled Models/{dataset_name}_Plot/Plot_{name}_{model}_{tolerance}.pkl", 'wb') as file:
                pickle.dump({
                    'model': lgbm_model_fit,
                }, file)

        #########################################
        # Saving the kfold data split:
        df_train_split_x = pd.DataFrame(x_data_train_split)
        df_train_split_y = pd.DataFrame(y_data_train_split)
        df_test_split_x = pd.DataFrame(x_data_test_split)
        df_test_split_y = pd.DataFrame(y_data_test_split)
        df_train_split_x.to_csv(f"Pickled Models/{dataset_name}_Plot/Train_{name}_{tolerance}_x.csv", index=False)
        df_train_split_y.to_csv(f"Pickled Models/{dataset_name}_Plot/Train_{name}_{tolerance}_y.csv", index=False)
        df_test_split_x.to_csv(f"Pickled Models/{dataset_name}_Plot/Test_{name}_{tolerance}_x.csv", index=False)
        df_test_split_y.to_csv(f"Pickled Models/{dataset_name}_Plot/Test_{name}_{tolerance}_y.csv", index=False)
        #########################################

        i = 0
        for train_index, test_index in kf.split(x_data_train_split, y_data_train_split_binned):
            i += 1
            # Splitting the data for each fold
            x_train, x_test = x_data_train_split[train_index], x_data_train_split[test_index]
            y_train, y_test = y_data_train_split[train_index], y_data_train_split[test_index]

            file_path = f"Hyperparameters/{dataset_name}/{name}_{model}_{tolerance}.json"
            with open(file_path, 'r') as f:
                best_params = json.load(f)

            if model == "RF":
                ML_model = RandomForestRegressor(**best_params)
            elif model == "LightGBM":
                ML_model = lgb.LGBMRegressor(**best_params)
            # Training datasets the model:
            ML_model_fit = ML_model.fit(x_train, y_train)
            # Saving the model:
            with open(f"Pickled Models/{dataset_name}_KFold/KFold_{i}_{name}_{model}_{tolerance}.pkl",
                      'wb') as file:
                pickle.dump({
                    'model': ML_model_fit,
                }, file)
            #########################################
            # Saving the kfold data split:
            df_train_fold_x = pd.DataFrame(x_train)
            df_train_fold_y = pd.DataFrame(y_train)
            df_test_fold_x = pd.DataFrame(x_test)
            df_test_fold_y = pd.DataFrame(y_test)
            df_train_fold_x.to_csv(f"Pickled Models/{dataset_name}_KFold/KFold_{i}_{name}_{tolerance}_train_x.csv", index=False)
            df_train_fold_y.to_csv(f"Pickled Models/{dataset_name}_KFold/KFold_{i}_{name}_{tolerance}_train_y.csv", index=False)
            df_test_fold_x.to_csv(f"Pickled Models/{dataset_name}_KFold/KFold_{i}_{name}_{tolerance}_test_x.csv", index=False)
            df_test_fold_y.to_csv(f"Pickled Models/{dataset_name}_KFold/KFold_{i}_{name}_{tolerance}_test_y.csv", index=False)
            #########################################

            y_predict = ML_model_fit.predict(x_test)

            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_predict)))
            r2_scores.append(r2_score(y_test, y_predict))
            mae_scores.append(mean_absolute_error(y_test, y_predict))

    else:

        if dataset == "CombiSolu":
            dataset_name = "Combi_Solu"
        elif dataset == "BigSolDB":
            dataset_name = "BigSolDB"
        for i in range(1, 5+1):
            #########################################
            x_test = np.array(pd.read_csv(f"Pickled Models/{dataset_name}_KFold/KFold_{i}_{name}_{tolerance}_test_x.csv",
                                 index_col=None))
            y_test = np.array(pd.read_csv(f"Pickled Models/{dataset_name}_KFold/KFold_{i}_{name}_{tolerance}_test_y.csv",
                                 index_col=None))
            #########################################
            with open(f"Pickled Models/{dataset_name}_KFold/KFold_{i}_{name}_{model}_{tolerance}.pkl", 'rb') as file:
                pickle_model = pickle.load(file)
                ML_model_fit = pickle_model['model']
            y_predict = ML_model_fit.predict(x_test)

            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_predict)))
            r2_scores.append(r2_score(y_test, y_predict))
            mae_scores.append(mean_absolute_error(y_test, y_predict))

    #print(f"Average RMSE: {np.round(np.mean(rmse_scores),3)}")
    #print(f"Average R2: {np.round(np.mean(r2_scores),3)}")
    if make_pickle is True:
        with open(f"Pickled Models/{dataset_name}_KFold/Scores/KFold_validation_score_{name}_{model}_{tolerance}.txt", 'w') as file:
            file.write(f"The R2 score from the GridSearch for reference is {grid_search_r2_score}\n")
            file.write(f"The 5-fold validation R2 score is: {np.mean(r2_scores)}\n")
            file.write(f"The 5-fold validation RMSE score is: {np.mean(rmse_scores)}\n")
            file.write(f"The 5-fold validation MAE score is: {np.mean(mae_scores)}\n")
    else:
        with open(f"Pickled Models/{dataset_name}_KFold/Scores/Pickle_Read_KFold_validation_score_{name}_{model}_{tolerance}.txt", 'w') as file:
            file.write(f"The 5-fold validation R2 score is: {np.mean(r2_scores)}\n")
            file.write(f"The 5-fold validation RMSE score is: {np.mean(rmse_scores)}\n")
            file.write(f"The 5-fold validation MAE score is: {np.mean(mae_scores)}\n")

##################################################################################################################
##################################################################################################################


def Plot(path, tolerance, model, make_pickle, dataset):
    pattern = r'/([^/_]+(?:_[^/_]+)*)[^/]*?_\d+\.\d+\.csv'
    path = path.replace(".csv", f"_{tolerance}.csv")
    match = re.search(pattern, path)
    name = match.group(1)

    if dataset == "CombiSolu":
        dataset_name = "Combi_Solu"
    elif dataset == "BigSolDB":
        dataset_name = "BigSolDB"

    # Load the model from pickle
    with open(f"Pickled Models/{dataset_name}_Plot/Plot_{name}_{model}_{tolerance}.pkl", 'rb') as file:
        data = pickle.load(file)
        ML_model_fit = data['model']

    # Load test and train data
    x_data_test_split = np.array(pd.read_csv(f"Pickled Models/{dataset_name}_Plot/Test_"
                                             f"{name}_{tolerance}_x.csv", index_col=None))
    y_data_test_split = np.array(pd.read_csv(f"Pickled Models/{dataset_name}_Plot/Test_"
                                             f"{name}_{tolerance}_y.csv", index_col=None))
    x_data_train_split = np.array(pd.read_csv(f"Pickled Models/{dataset_name}_Plot/Train_"
                                              f"{name}_{tolerance}_x.csv", index_col=None))
    y_data_train_split = np.array(pd.read_csv(f"Pickled Models/{dataset_name}_Plot/Train_"
                                              f"{name}_{tolerance}_y.csv", index_col=None))

    # Making the predictions
    y_predict_test = ML_model_fit.predict(x_data_test_split)
    y_predict_train = ML_model_fit.predict(x_data_train_split)

    rmse_score_value_test = np.sqrt(mean_squared_error(y_data_test_split, y_predict_test))
    r2_score_value_test = r2_score(y_data_test_split, y_predict_test)
    mae_score_value_test = mean_absolute_error(y_data_test_split, y_predict_test)
    rmse_score_value_train = np.sqrt(mean_squared_error(y_data_train_split, y_predict_train))
    r2_score_value_train = r2_score(y_data_train_split, y_predict_train)
    mae_score_value_train = mean_absolute_error(y_data_train_split, y_predict_train)

    if make_pickle is True:
        with open(f"Pickled Models/{dataset_name}_Plot/Scores/train_test_score_{name}_{model}_{tolerance}.txt", 'w') as file:
            file.write(f"The test RMSE score is: {rmse_score_value_test}\n")
            file.write(f"The test R2 score is: {r2_score_value_test}\n")
            file.write(f"The test MAE score is: {mae_score_value_test}\n")
            file.write("-----------------------------------------------\n")
            file.write(f"The train RMSE score is: {rmse_score_value_train}\n")
            file.write(f"The train R2 score is: {r2_score_value_train}\n")
            file.write(f"The train MAE score is: {mae_score_value_train}\n")
    else:
        with open(f"Pickled Models/{dataset_name}_Plot/Scores/Pickle_Read_train_test_score_{name}_{model}_{tolerance}.txt", 'w') as file:
            file.write(f"The test RMSE score is: {rmse_score_value_test}\n")
            file.write(f"The test R2 score is: {r2_score_value_test}\n")
            file.write(f"The test MAE score is: {mae_score_value_test}\n")
            file.write("-----------------------------------------------\n")
            file.write(f"The train RMSE score is: {rmse_score_value_train}\n")
            file.write(f"The train R2 score is: {r2_score_value_train}\n")
            file.write(f"The train MAE score is: {mae_score_value_train}\n")

    ##################################################################################################################
    ##################################################################################################################

    ########################################### PLOTTING  ############################################################

    # The hexbin plot:
    fig, ax_main = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.1, top=1.0, right=0.72, bottom=-0.1)
    confidence_interval = np.percentile(np.array(y_predict_test) - np.array(y_data_test_split), [2.5, 97.5])
    if dataset == "CombiSolu":
        ticks = [-8, -6, -4, -2, 0, 2]
    elif dataset == "BigSolDB":
        ticks = [-5, -4, -3, -2, -1, 0]
    ax_main.set_aspect('equal')
    y_predict_test = np.array(y_predict_test)
    y_data_test_split = np.array(y_data_test_split.reshape(1, -1))[0]

    axis_min = min(ticks)
    axis_max = max(ticks)

    cmap = mcolors.LinearSegmentedColormap.from_list("custom", [
        '#FFF7CC',  # Light yellow
        '#F9A825',  # Medium yellow-orange
        '#EF6C00',  # Dark orange
        '#C62828',  # Dark red
        '#7F0000'  # Deep red
    ])

    hexbin = ax_main.hexbin(y_predict_test, y_data_test_split, gridsize=30, cmap=cmap, mincnt=2,
                            edgecolors='black', linewidths=1.5, extent=(axis_min, axis_max, axis_min, axis_max))
    ##############################################################################################

    # The dark blue line:
    ax_main.plot([axis_min, axis_max], [axis_min, axis_max], color='#283f7d', linestyle='-', linewidth=4)
    ##############################################################################################

    # Making the outside lines thicker and background white:
    for spine in ax_main.spines.values():
        spine.set_linewidth(2.0)
    ax_main.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ##############################################################################################

    # Some axes stuff:
    if dataset == "CombiSolu":
        ax_main.set_xlabel('Predicted log(S / mol/dm³)', fontsize=20)
        ax_main.set_ylabel('Experimental datasets log(S / mol/dm³)', fontsize=20)
    elif dataset == "BigSolDB":
        ax_main.set_xlabel('Predicted log(x)', fontsize=20)
        ax_main.set_ylabel('Experimental datasets log(x)', fontsize=20)

    ax_main.set_xticks(ticks)
    ax_main.set_yticks(ticks)
    plt.tick_params(axis='both', which='major', labelsize=18)

    ax_main.set_xlim(axis_min, axis_max)
    ax_main.set_ylim(axis_min, axis_max)
    #############################################################################################

    # Making the histogram and KDE plots:
    left, bottom, width, height = ax_main.get_position().bounds
    ax_histx = fig.add_axes([left, bottom + height, width, 0.15], sharex=ax_main)
    ax_histy = fig.add_axes([left + width, bottom, 0.15, height], sharey=ax_main)

    ax_histx.hist(y_predict_test, bins=20, density=True, alpha=0.75, color='#FFF7CC', edgecolor='black')
    kde_x = stats.gaussian_kde(y_predict_test)
    x_range = np.linspace(axis_min, axis_max, 200)
    ax_histx.plot(x_range, kde_x(x_range), color='#0f172a', linewidth=3)

    ax_histy.hist(y_data_test_split, bins=20, density=True, alpha=0.75, color='#FFF7CC', orientation='horizontal',
                  edgecolor='black')
    kde_y = stats.gaussian_kde(y_data_test_split)
    y_range = np.linspace(axis_min, axis_max, 200)
    ax_histy.plot(kde_y(y_range), y_range, color='#0f172a', linewidth=3)

    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                         labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    ax_histy.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                         labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    ax_histx.set_xlim(ax_main.get_xlim())
    ax_histy.set_ylim(ax_main.get_ylim())
    #############################################################################################

    # The colorbar:
    pos = ax_main.get_position()
    cax = fig.add_axes([pos.x1 + 0.16, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(hexbin, cax=cax)
    cbar.set_label('Count', fontsize=20, labelpad=22, rotation=270)
    ax_main.set_position([pos.x0, pos.y0, pos.width, pos.height])
    cbar.ax.tick_params(labelsize=18, length=6, width=2)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #############################################################################################

    # Removing spines for KDE plots
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)

    ax_main.text(x=0.05, y=0.95,
                 s=f"RMSE: {rmse_score_value_test:.3f}\nMAE: {mae_score_value_test:.3f}\n$R^2$: {r2_score_value_test:.3f}",
                 fontsize=20, verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5', linewidth=1.5),
                 transform=ax_main.transAxes)

    if name == "GC_BigSolDB":
        feature_name = "GC"
    elif name == "MACCS_BigSolDB":
        feature_name = "MACCS"
    elif name == "GC_MACCS_BigSolDB":
        feature_name = "GC-MACCS"
    elif name == "UNIFAC_BigSolDB":
        feature_name = "UNIFAC"
    # Feature name:
    ax_main.text(0.90, 0.10, feature_name, transform=ax_main.transAxes, ha='right', va='bottom', fontsize=24)
    ###############################################################################

    ###############################################################################
    if dataset == "CombiSolu":
        # Creating the inset axis for the mini-plot
        ax_inset = inset_axes(ax_main, width="100%", height="100%", loc='lower right',
                              bbox_to_anchor=(0.6, 0.0, 0.4, 0.4), bbox_transform=ax_main.transAxes, borderpad=1)

        # Plotting the mini-plot
        ax_inset.scatter(y_predict_test, y_data_test_split, color='black', s=6)
        ax_inset.plot([-9, 3], [-9, 3], color='blue', linestyle='-', linewidth=1)
        ax_inset.plot([-9, 3], [-9 + confidence_interval[0], 3 + confidence_interval[0]], color='red', linestyle='--',
                      linewidth=1)
        ax_inset.plot([-9, 3], [-9 + confidence_interval[1], 3 + confidence_interval[1]], color='red', linestyle='--',
                      linewidth=1)
        # Setting the limits of the inset plot to avoid cutting off the data
        ax_inset.set_xlim(-3, 1)
        ax_inset.set_ylim(-3, 1)

        mark_inset(ax_main, ax_inset, loc1=3, loc2=1, fc="none", ec="0.5")

        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
    ###############################################################################

    # Saving the plot
    if dataset == "CombiSolu":
        dataset_name = "CombiSolu"
    elif dataset == "BigSolDB":
        dataset_name = "BigSolDB"

    plt.savefig(f"Plots Images/{dataset_name}/Plot_{name}_{model}_{tolerance}.png")
    plt.close(fig)
    ##################################################################################################################
    ##################################################################################################################


def Predict_Organic_Solubility(path_dataset_for_training, path_dataset_for_prediction,
                               make_pickle, tolerance, model, feature_type, dataset):

    ########################################## DATA ####################################################
    solute_solvent_df = pd.read_csv(path_dataset_for_training)
    if dataset == "CombiSolu":
        solute_solvent_df.rename(columns={"experimental_logS [mol/L]": "Solubility"}, inplace=True)
    elif dataset == "BigSolDB":
        solute_solvent_df.rename(columns={"experimental_logX": "Solubility"}, inplace=True)
    # Creating the x-data and the y-data:
    y_data = solute_solvent_df["Solubility"]
    if "index" in solute_solvent_df.columns:
        solute_solvent_df.drop(columns='index', inplace=True)
    if "Unnamed: 0" in solute_solvent_df.columns:
        solute_solvent_df.drop(columns='Unnamed: 0', inplace=True)
    if "temperature" in solute_solvent_df.columns:
        solute_solvent_df.drop(columns="temperature", inplace=True)

    x_data = solute_solvent_df.drop(columns=["solute_smiles", "solvent_smiles", "Solubility", "solute_InChIKey",
                                             "solvent_InChIKey"])
    scaler = MinMaxScaler()
    scaler = scaler.fit(x_data)
    x_data_scaled = scaler.transform(x_data)
    ####################################################################################################

    if dataset == "CombiSolu":
        dataset_name = "Combi_Solu"
        dataset_name_2 = "CombiSolu"
    elif dataset == "BigSolDB":
        dataset_name = "BigSolDB"
        dataset_name_2 = "BigSolDB"

    if make_pickle is True:
        # Loading the training dataset which is called the solute_solvent_df
        # It is basically the CombiSolu-Exp dataset. The trained model will then predict
        # the solubility of organic compounds for each of the solvents in the CombiSolu-Exp dataset
        ########################################## TRAIN ####################################################
        file_path = f"Hyperparameters/{dataset_name}/{feature_type}_{dataset_name_2}_{model}_{tolerance}.json"
        with open(file_path, 'r') as f:
            best_params = json.load(f)

        # Defining the models:
        if model == "RF":
            ML_model = RandomForestRegressor(**best_params)
        elif model == "LightGBM":
            ML_model = lgb.LGBMRegressor(**best_params)
        ML_model_fit = ML_model.fit(x_data_scaled, y_data)

        # Saving the model:
        with open(f'Pickled Models/{dataset_name}_Pipeline_Pred/Pipeline_{feature_type}_'
                       f'{dataset_name_2}_{model}_{tolerance}.pkl', 'wb') as file:
                pickle.dump({
                    'model': ML_model_fit,
                }, file)
        ####################################################################################################

    else:
        ########################################## READ ####################################################
        with open(f'Pickled Models/{dataset_name}_Pipeline_Pred/Pipeline_{feature_type}_'
                       f'{dataset_name_2}_{model}_{tolerance}.pkl',
                       'rb') as file:
            data = pickle.load(file)
            ML_model_fit = data['model']
        ####################################################################################################

    ########################################## PREDICTION ###################################################
    test_df = pd.read_csv(path_dataset_for_prediction)
    test_solute_smiles_list = list(test_df["solute_smiles"])

    #### Just getting the unique values too
    #############################
    test_solvent_smiles_list = list(test_df["solvent_smiles"])
    test_solute_inchikey_list = list(test_df["solute_InChIKey"])
    test_solvent_inchikey_list = list(test_df["solvent_InChIKey"])

    if dataset == "CombiSolu":
        test_df.rename(columns={"experimental_logS [mol/L]": "Solubility"}, inplace=True)
    elif dataset == "BigSolDB":
        test_df.rename(columns={"experimental_logX": "Solubility"}, inplace=True)
    if "Unnamed: 0" in test_df.columns:
        test_df.drop(columns=["Unnamed: 0"], inplace=True)
    if "temperature" in test_df.columns:
        test_df.drop(columns=["temperature"], inplace=True)

    x_data_for_prediction = test_df.drop(
        columns=["solute_smiles", "solvent_smiles", "Solubility", "solute_InChIKey",
                 "solvent_InChIKey"])

    #print(test_df)
    x_data_for_prediction_scaled = scaler.transform(x_data_for_prediction)

    the_prediction = ML_model_fit.predict(x_data_for_prediction_scaled)
    test_prediction_df = pd.DataFrame({"Solubility Prediction": the_prediction,
                                       "Solvent_smiles": test_solvent_smiles_list,
                                       "Molecule_smiles": test_solute_smiles_list,
                                       "Solvent_InChIKey": test_solvent_inchikey_list,
                                       "Molecule_InChIKey": test_solute_inchikey_list})

    # I don't know why the column for the Solvent_InChIKey looked weird. A quick fix is necessary:
    def fun_to_apply(row):
        weird_string = row["Solvent_InChIKey"]
        match = re.search(r"\('([A-Z0-9-]+)',\)", weird_string)
        if match is not None:
            return match.group(1)
        else:
            return weird_string
    test_prediction_df["Solvent_InChIKey"] = test_prediction_df.apply(fun_to_apply, axis=1)
    test_prediction_df["Solvent_InChIKey"] = test_prediction_df.apply(fun_to_apply, axis=1)

    # Saving as CSV file:
    if make_pickle is True:
        test_prediction_df.to_csv(f"Pipeline Predictions/{dataset_name}/Pipeline_{feature_type}_"
                                  f"{dataset_name_2}_{model}_{tolerance}.csv", index=False)
    else:
        test_prediction_df.to_csv(f"Pipeline Predictions/{dataset_name}/Pipeline_PickleRead_{feature_type}_"
                                  f"{dataset_name_2}_{model}_{tolerance}.csv", index=False)

    return None


def Mol_Inter_Analysis(prediction_df, molecule_smiles_list, feature_type, model, dataset):
    for molecule_smiles in molecule_smiles_list:

        best_solvents_dict = {}
        worst_solvents_dict = {"Solvent_smiles": [], "Solubility Prediction": []}
        molecule_df = prediction_df[prediction_df["Molecule_smiles"] == molecule_smiles].copy()
        molecule_df.sort_values(by="Solubility Prediction", ascending=False, inplace=True)

        if dataset == "CombiSolu":
            dataset_name = "Combi_Solu"
        elif dataset == "BigSolDB":
            dataset_name = "BigSolDB"

        if "/" in molecule_smiles:
            molecule_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(molecule_smiles), isomericSmiles=False)

        # Saving the solvents:
        main_path = f"Pipeline Predictions/{dataset_name}/Best_Worst_Showcase/{feature_type}_{model}/{molecule_smiles}"
        if not os.path.exists(main_path):
            os.makedirs(main_path)
            best_path = main_path + "/Best"
            worst_path = main_path + "/Worst"
            os.makedirs(best_path)
            os.makedirs(worst_path)

        # Saving the dataframe of the organic solvents and the solubility:
        molecule_df.to_csv(f"Pipeline Predictions/{dataset_name}/Best_Worst_Showcase/{feature_type}_{model}/"
                           f"{molecule_smiles}/Solubility Prediction Dataframe.csv")

        # Obtaining the best 5 solvents:
        best_solvents_dict["Solvent_smiles"] = list(molecule_df["Solvent_smiles"])[0:5]
        best_solvents_dict["Solubility Prediction"] = list(molecule_df["Solubility Prediction"])[0:5]

        # Obtaining the worst 5 solvents:
        worst_solvents_dict["Solvent_smiles"] = list(molecule_df["Solvent_smiles"])[-5:]
        worst_solvents_dict["Solubility Prediction"] = list(molecule_df["Solubility Prediction"])[-5:]

        i = -1
        for solubility_value in best_solvents_dict["Solubility Prediction"]:
            i += 1
            solvent_smiles = best_solvents_dict["Solvent_smiles"][i]
            img = Draw.MolToImage(Chem.MolFromSmiles(solvent_smiles), size=(400, 400))
            file_path = main_path + "/Best/" + f"{solvent_smiles}.png"
            img.save(file_path)

        i = -1
        for solubility_value in worst_solvents_dict["Solubility Prediction"]:
            i += 1
            solvent_smiles = worst_solvents_dict["Solvent_smiles"][i]
            img = Draw.MolToImage(Chem.MolFromSmiles(solvent_smiles), size=(400, 400))
            file_path = main_path + "/Worst/" + f"{solvent_smiles}.png"
            img.save(file_path)

        fig = plt.figure(figsize=(20, 8))  # Adjust size as needed

        grid = plt.GridSpec(2, 7, width_ratios=[1, 1, 1, 1, 1, 0.1, 2.5])

        # Making images for 5 best solvents
        for i in range(5):
            ax = fig.add_subplot(grid[0, i])
            img_path = f"{main_path}/Best/{best_solvents_dict['Solvent_smiles'][i]}.png"
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"{best_solvents_dict['Solubility Prediction'][i]:.3f}", fontsize=14)  # Increased font size
            ax.axis('off')

        # Making images for 5 worst solvents
        for i in range(5):
            ax = fig.add_subplot(grid[1, i])
            img_path = f"{main_path}/Worst/{worst_solvents_dict['Solvent_smiles'][i]}.png"
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"{worst_solvents_dict['Solubility Prediction'][i]:.3f}", fontsize=14)  # Increased font size
            ax.axis('off')

        # Display the molecule image in a larger plot
        mol_ax = fig.add_subplot(grid[:, 6])  # Span across both rows at the last column
        mol_img = Draw.MolToImage(Chem.MolFromSmiles(molecule_smiles), size=(500, 500))  # Adjusted size
        mol_ax.imshow(mol_img)
        mol_ax.axis('off')

        fig_path = main_path + f"/Best_Worst_{molecule_smiles}.png"
        plt.savefig(fig_path)
        plt.close()


def plot_solvents_on_axis(prediction_df, molecule_smiles_list, feature_type, model, dataset):
    for molecule_smiles in molecule_smiles_list:
        molecule_df = prediction_df[prediction_df["Molecule_smiles"] == molecule_smiles].copy()
        molecule_df.sort_values(by="Solubility Prediction", ascending=True, inplace=True)

        if "/" in molecule_smiles:
            molecule_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(molecule_smiles), isomericSmiles=False)

        num_solvents = len(molecule_df)
        max_vertical = 4
        columns = (num_solvents // max_vertical) + (1 if num_solvents % max_vertical else 0)
        column_spacing = 3
        fig_width = columns * column_spacing
        fig_height = max_vertical * 1.5

        # Plotting
        fig, ax = plt.subplots(figsize=(fig_width, fig_height+2.5))
        image_width = 1.5
        image_height = 1.2
        stair_step_x = 0.3
        ##########################################################
        if molecule_smiles == "C1=CC=C(C=C1)C=O":
            title = "Benzaldehyde"
        elif molecule_smiles == "C1=CC=C(C=C1)O":
            title = "Phenol"
        else:
            # let's just leave other cases for now
            title = ""

        ax.set_title(f"{title}", fontsize=38, horizontalalignment='center',
             position=(0.45, 1.1), pad=30)

        mol = Chem.MolFromSmiles(molecule_smiles)
        img = Draw.MolToImage(mol, size=(400, 400))
        img = img.convert("RGBA")
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)

        carbon_count = 0
        oxygen_count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                carbon_count += 1
            elif atom.GetAtomicNum() == 8:
                oxygen_count += 1

        if carbon_count + oxygen_count < 12:
            image_size_inches = [img.size[0] / (3.5 * 300), img.size[1] / (3.5 * 300)]
            ax_img = fig.add_axes([0.45, 0.72, image_size_inches[0], image_size_inches[1]])
        else:
            image_size_inches = [img.size[0] / (4.5 * 300), img.size[1] / (4.5 * 300)]
            img = img.rotate(180)
            ax_img = fig.add_axes([0.47, 0.75, image_size_inches[0], image_size_inches[1]])

        ax_img.imshow(img, aspect='equal')
        ax_img.axis('off')
        ##################################################################

        arrow = FancyArrow(-8, fig_height / 2, fig_width+1, 0,
                           width=fig_height / 2, head_width=fig_height / 1.2, head_length=fig_width / 6,
                           color='#ADD8E6', alpha=0.5, zorder=0)
        ax.add_patch(arrow)

        for i, (index, row) in enumerate(molecule_df.iterrows()):
            solvent_smiles = row['Solvent_smiles']
            solvent_mol = Chem.MolFromSmiles(solvent_smiles)
            opts = Draw.MolDrawOptions()
            opts.bondLineWidth = 8

            # Determining the column
            column_index = i // max_vertical

            # Determining the row
            row_index = i % max_vertical

            x_value = column_index * column_spacing + row_index * stair_step_x
            y_value = fig_height - (row_index + 1) * image_height - 0.7

            img = Draw.MolToImage(solvent_mol, size=(300, 300), options=opts, backgroundcolor=None)

            y_start = y_value
            y_end = y_start + image_height
            x_start = x_value
            x_end = x_start + image_width

            # Converting PIL image to have a transparent background. SUPER IMPORTANT!
            img = img.convert("RGBA")
            datas = img.getdata()

            newData = []
            for item in datas:
                if item[0] == 255 and item[1] == 255 and item[2] == 255:  # Finding white background
                    newData.append((255, 255, 255, 0))  # Making white -> transparent
                else:
                    newData.append(item)
            img.putdata(newData)
            ax.imshow(img, aspect='equal', extent=(x_start, x_end, y_start, y_end), zorder=2)

        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.axis('off')

        if dataset == "CombiSolu":
            dataset_name = "Combi_Solu"
        elif dataset == "BigSolDB":
            dataset_name = "BigSolDB"

        main_path = f"Pipeline Predictions/{dataset_name}/Best_Worst_Showcase/{feature_type}_{model}/{molecule_smiles}"
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        fig_path = main_path + f"/Solvent_Spectrum_{molecule_smiles}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=200)
        plt.close()


def plot_solubility_violin(prediction_df, molecule_smiles_list, feature_type, model, dataset):

    for molecule_smiles in molecule_smiles_list:
        molecule_df = prediction_df[prediction_df["Molecule_smiles"] == molecule_smiles].copy()
        if "/" in molecule_smiles:
            molecule_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(molecule_smiles), isomericSmiles=False)

        violin = sns.violinplot(y=molecule_df['Solubility Prediction'],
                                inner='quartile',
                                linewidth=2,
                                color='skyblue',
                                saturation=0.7)

        violin.set_title(f"{feature_type} {model}", fontsize=24)
        violin.set_ylabel("Solubility Prediction", fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)

        plt.gca().set_facecolor('white')
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

        if dataset == "CombiSolu":
            dataset_name = "Combi_Solu"
        elif dataset == "BigSolDB":
            dataset_name = "BigSolDB"

        main_path = f"Pipeline Predictions/{dataset_name}/Best_Worst_Showcase/{feature_type}_{model}/{molecule_smiles}"
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        fig_path = main_path + f"/Solubility_Violin_{molecule_smiles}.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()