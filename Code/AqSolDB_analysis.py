import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
import json
import Code
import lightgbm as lgb
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from scipy import stats

np.random.seed(1)

#########################
import logging
import warnings
logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
#########################


def KFold_validation_test_validation(path, make_pickle, model, do_PCA):

    # Defining the Stratified KFold for the primary train/test splitting and the 5-fold cross validation later on
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    rmse_scores = []
    r2_scores = []
    mae_scores = []

    pattern = r'/([^/_]+(?:_[^/_]+)*)[^/]*?.csv'
    match = re.search(pattern, path)
    name = match.group(1)

    if make_pickle is True:
        # Loading the dataset
        solute_solvent_df = pd.read_csv(path)
        solute_solvent_df = solute_solvent_df.dropna()

        # Creating the x-data and the y-data:
        if "index" in solute_solvent_df.columns:
            solute_solvent_df.drop(columns='index', inplace=True)
        if "Unnamed: 0" in solute_solvent_df.columns:
            solute_solvent_df.drop(columns='Unnamed: 0', inplace=True)
        solute_solvent_df.reset_index(drop=True, inplace=True)

        try:
            solute_solvent_df.rename(columns={"experimental_logS [mol/L]": "Solubility"}, inplace=True)
        except:
            pass

        solute_solvent_df_x = np.array(solute_solvent_df.drop(columns=["solute_smiles","solvent_smiles", "Solubility",
                                                                       "solute_InChIKey", "solvent_InChIKey"]))
        solute_solvent_df_y = np.array(solute_solvent_df["Solubility"])

        ################################################
        # The stratified train/test split:
        solute_solvent_df_y_binned  = pd.qcut(solute_solvent_df_y, q=5, labels=False)
        ################################################
        # MinMax scaling the x data of the dataframe:
        scaler = MinMaxScaler()
        solute_solvent_df_x_scaled = scaler.fit_transform(solute_solvent_df_x)

        ################################################################################################
        #### Performing the PCA (ultimately not used but kept just in case):
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
            Code.PCA_plot.make_PCA_plot_n_components_validation(pca, cumulative_variance, n_components_90,
                                                                n_components_95, name, model, "AqSolu")
        ################################################################################################

        x_data_train_split, x_data_test_split, y_data_train_split, y_data_test_split = train_test_split(
            solute_solvent_df_x_scaled, solute_solvent_df_y, train_size=0.8,
            stratify=solute_solvent_df_y_binned, random_state=1
        )
        ################################################
        # The stratified 5-fold cross validation:
        y_data_train_split_binned = pd.qcut(y_data_train_split, q=5, labels=False)
        ################################################

        dataset_name = "Aq_Solu"
        ###################### RandomForest ######################
        if model == "RF":
            # Training the model (getting the best parameters):
            random_forest_model_fit, grid_search_r2_score = Code.Train_RF_model(kf, x_data_train_split, y_data_train_split_binned,
                                                                                name, y_data_train_split, dataset_name)
            # Saving the model:
            with open(f"Pickled Models/Aq_Solu_Plot/Plot_{name}_{model}.pkl", 'wb') as file:
                pickle.dump({
                    'model': random_forest_model_fit,
                }, file)
        ##########################################################

        ###################### LightGBM ######################
        elif model == "LightGBM":
            # Training the model (getting the best parameters):
            lgbm_model_fit, grid_search_r2_score = Code.Train_LGBM_model(kf, x_data_train_split, y_data_train_split_binned,
                                                                         name, y_data_train_split, dataset_name)
            with open(f"Pickled Models/Aq_Solu_Plot/Plot_{name}_{model}.pkl", 'wb') as file:
                pickle.dump({
                    'model': lgbm_model_fit,
                }, file)

        #########################################
        # Saving the kfold data split:
        df_train_split_x = pd.DataFrame(x_data_train_split)
        df_train_split_y = pd.DataFrame(y_data_train_split)
        df_test_split_x = pd.DataFrame(x_data_test_split)
        df_test_split_y = pd.DataFrame(y_data_test_split)
        df_train_split_x.to_csv(f"Pickled Models/Aq_Solu_Plot/Train_{name}_x.csv", index=False)
        df_train_split_y.to_csv(f"Pickled Models/Aq_Solu_Plot/Train_{name}_y.csv", index=False)
        df_test_split_x.to_csv(f"Pickled Models/Aq_Solu_Plot/Test_{name}_x.csv", index=False)
        df_test_split_y.to_csv(f"Pickled Models/Aq_Solu_Plot/Test_{name}_y.csv", index=False)
        #########################################

        i = 0
        for train_index, test_index in kf.split(x_data_train_split, y_data_train_split_binned):
            i += 1
            # Splitting the data for each fold
            x_train, x_test = x_data_train_split[train_index], x_data_train_split[test_index]
            y_train, y_test = y_data_train_split[train_index], y_data_train_split[test_index]

            file_path = f"Hyperparameters/{dataset_name}/{name}_{model}.json"
            with open(file_path, 'r') as f:
                best_params = json.load(f)
            # Defining the models:
            if model == "RF":
                ML_model = RandomForestRegressor(**best_params)
            elif model == "LightGBM":
                ML_model = lgb.LGBMRegressor(**best_params)

            # Training the model:
            ML_model_fit = ML_model.fit(x_train, y_train)
            # Saving the model:
            with open(f"Pickled Models/Aq_Solu_KFold/KFold_{i}_{name}_{model}.pkl", 'wb') as file:
               pickle.dump({
                   'model': ML_model_fit,
               }, file)

            #########################################
            # Saving the kfold data split:
            df_train_fold_x = pd.DataFrame(x_train)
            df_train_fold_y = pd.DataFrame(y_train)
            df_test_fold_x = pd.DataFrame(x_test)
            df_test_fold_y = pd.DataFrame(y_test)
            df_train_fold_x.to_csv(f"Pickled Models/Aq_Solu_KFold/KFold_{i}_{name}_train_x.csv", index=False)
            df_train_fold_y.to_csv(f"Pickled Models/Aq_Solu_KFold/KFold_{i}_{name}_train_y.csv", index=False)
            df_test_fold_x.to_csv(f"Pickled Models/Aq_Solu_KFold/KFold_{i}_{name}_test_x.csv", index=False)
            df_test_fold_y.to_csv(f"Pickled Models/Aq_Solu_KFold/KFold_{i}_{name}_test_y.csv", index=False)
            #########################################

            y_predict = ML_model_fit.predict(x_test)
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_predict)))
            r2_scores.append(r2_score(y_test, y_predict))
            mae_scores.append(mean_absolute_error(y_test, y_predict))

    else:
        for i in range(1, 5+1):
            #########################################
            x_test = np.array(pd.read_csv(f"Pickled Models/Aq_Solu_KFold/KFold_{i}_{name}_test_x.csv",
                                 index_col=None))
            y_test = np.array(pd.read_csv(f"Pickled Models/Aq_Solu_KFold/KFold_{i}_{name}_test_y.csv",
                                 index_col=None))
            #########################################

            with open(f"Pickled Models/Aq_Solu_KFold/KFold_{i}_{name}_{model}.pkl", 'rb') as file:
                pickle_model = pickle.load(file)
                ML_model = pickle_model['model']
            y_predict = ML_model.predict(x_test)

            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_predict)))
            r2_scores.append(r2_score(y_test, y_predict))
            mae_scores.append(mean_absolute_error(y_test, y_predict))

    if make_pickle is True:
        with open(f"Pickled Models/Aq_Solu_KFold/Scores/KFold_validation_score_{name}_{model}.txt", 'w') as file:
            file.write(f"The R2 score from the GridSearch for reference is {grid_search_r2_score}\n")
            file.write(f"The 5-fold validation R2 score is: {np.mean(r2_scores)}\n")
            file.write(f"The 5-fold validation RMSE score is: {np.mean(rmse_scores)}\n")
            file.write(f"The 5-fold validation MAE score is: {np.mean(mae_scores)}\n")
    else:
        with open(f"Pickled Models/Aq_Solu_KFold/Scores/Pickle_Read_KFold_validation_score_{name}_{model}.txt", 'w') as file:
            file.write(f"The 5-fold validation R2 score is: {np.mean(r2_scores)}\n")
            file.write(f"The 5-fold validation RMSE score is: {np.mean(rmse_scores)}\n")
            file.write(f"The 5-fold validation MAE score is: {np.mean(mae_scores)}\n")



def Plot(path, model, make_pickle):
    pattern = r'/([^/_]+(?:_[^/_]+)*)[^/]*?.csv'
    match = re.search(pattern, path)
    name = match.group(1)

    ##################################### LOADING AND TRAINING THE MODEL ############################################
    with open(f"Pickled Models/Aq_Solu_Plot/Plot_{name}_{model}.pkl", 'rb') as file:
        data = pickle.load(file)
        ML_model_fit = data['model']

    x_data_test_split = np.array(pd.read_csv(f"Pickled Models/Aq_Solu_Plot/Test_{name}_x.csv",
                                    index_col=None))
    y_data_test_split = np.array(pd.read_csv(f"Pickled Models/Aq_Solu_Plot/Test_{name}_y.csv",
                                    index_col=None))
    x_data_train_split = np.array(pd.read_csv(f"Pickled Models/Aq_Solu_Plot/Train_{name}_x.csv",
                                    index_col=None))
    y_data_train_split = np.array(pd.read_csv(f"Pickled Models/Aq_Solu_Plot/Train_{name}_y.csv",
                                    index_col=None))
    #########################################

    #########################################
    # Making the predictions
    y_predict_test = ML_model_fit.predict(x_data_test_split)
    rmse_score_value_test = np.sqrt(mean_squared_error(y_data_test_split, y_predict_test))
    r2_score_value_test = r2_score(y_data_test_split, y_predict_test)
    mae_score_value_test = mean_absolute_error(y_data_test_split, y_predict_test)
    y_predict_train = ML_model_fit.predict(x_data_train_split)
    rmse_score_value_train = np.sqrt(mean_squared_error(y_data_train_split, y_predict_train))
    r2_score_value_train = r2_score(y_data_train_split, y_predict_train)
    mae_score_value_train = mean_absolute_error(y_data_train_split, y_predict_train)
    #########################################

    if make_pickle is True:
        with open(f"Pickled Models/Aq_Solu_Plot/Scores/train_test_score_{name}_{model}.txt", 'w') as file:
            file.write(f"The test RMSE score is: {rmse_score_value_test}\n")
            file.write(f"The test R2 score is: {r2_score_value_test}\n")
            file.write(f"The test MAE score is: {mae_score_value_test}\n")
            file.write("-----------------------------------------------\n")
            file.write(f"The train RMSE score is: {rmse_score_value_train}\n")
            file.write(f"The train R2 score is: {r2_score_value_train}\n")
            file.write(f"The train MAE score is: {mae_score_value_train}\n")
    else:
        with open(f"Pickled Models/Aq_Solu_Plot/Scores/Pickle_Read_train_test_score_{name}_{model}.txt", 'w') as file:
            file.write(f"The test RMSE score is: {rmse_score_value_test}\n")
            file.write(f"The test R2 score is: {r2_score_value_test}\n")
            file.write(f"The test MAE score is: {mae_score_value_test}\n")
            file.write("-----------------------------------------------\n")
            file.write(f"The train RMSE score is: {rmse_score_value_train}\n")
            file.write(f"The train R2 score is: {r2_score_value_train}\n")
            file.write(f"The train MAE score is: {mae_score_value_train}\n")

    ########################################### PLOTTING  ############################################################
    fig, ax_main = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.1, top=1.0, right=0.72, bottom=-0.1)
    ticks = [-10, -8, -6, -4, -2, 0, 2]
    ax_main.set_aspect('equal')
    y_predict_test = np.array(y_predict_test)
    y_data_test_split = np.array(y_data_test_split.reshape(1, -1))[0]

    axis_min = min(ticks)
    axis_max = max(ticks)

    cmap = mcolors.LinearSegmentedColormap.from_list("custom", [
        '#C9E3FF',  # Light blue
        '#1F5F99',  # Darker blue
        '#0A3B5E',  # Deep blue
        '#071A33',  # Very dark blue
        '#051121' # Super dark blue
    ])

    hexbin = ax_main.hexbin(y_predict_test, y_data_test_split, gridsize=30, cmap=cmap, mincnt=2,
                            edgecolors='black', linewidths=1.5, extent=(axis_min, axis_max, axis_min, axis_max))

    # The orange line:
    ax_main.plot([axis_min, axis_max], [axis_min, axis_max], color='#FF5F1F', linestyle='-', linewidth=4)

    ##############################################################################################
    # Making the outside lines thicker and background white:
    for spine in ax_main.spines.values():
        spine.set_linewidth(2.0)
    ax_main.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ##############################################################################################

    # Some axes stuff:
    ax_main.set_xlabel('Predicted log(S / mol/dm³)', fontsize=20)
    ax_main.set_ylabel('Experimental log(S / mol/dm³)', fontsize=20)

    ax_main.set_xticks(ticks)
    ax_main.set_yticks(ticks)
    plt.tick_params(axis='both', which='major', labelsize=18)

    ax_main.set_xlim(axis_min, axis_max)
    ax_main.set_ylim(axis_min, axis_max)
    ##############################################################################################

    # Making the histogram and KDE plots:
    left, bottom, width, height = ax_main.get_position().bounds
    ax_histx = fig.add_axes([left, bottom + height, width, 0.15], sharex=ax_main)
    ax_histy = fig.add_axes([left + width, bottom, 0.15, height], sharey=ax_main)

    ax_histx.hist(y_predict_test, bins=20, density=True, alpha=0.75, color='#C9E3FF', edgecolor='black')
    kde_x = stats.gaussian_kde(y_predict_test)
    x_range = np.linspace(axis_min, axis_max, 200)
    ax_histx.plot(x_range, kde_x(x_range), color='#0f172a', linewidth=3)

    ax_histy.hist(y_data_test_split, bins=20, density=True, alpha=0.75, color='#C9E3FF', orientation='horizontal',
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

    if name == "GC_AqSolDB":
        feature_name = "GC"
    elif name == "MACCS_AqSolDB":
        feature_name = "MACCS"
    elif name == "GC_MACCS_AqSolDB":
        feature_name = "GC-MACCS"

    # Feature name:
    ax_main.text(0.90, 0.10, feature_name, transform=ax_main.transAxes, ha='right', va='bottom', fontsize=24)
    ###############################################################################

    plt.savefig(f"Plots Images/AqSolu/Plot_{name}_{model}.png")
    plt.close(fig)
    ##################################################################################################################
    ##################################################################################################################

def Predict_Aquatic_Solubility(path_dataset_for_training, path_dataset_for_prediction,
                               make_pickle, model, feature_type):

    ########################################## DATA ####################################################
    # pattern = r'.*\/([^\/]+?)_\d*\.*\d*\.csv'
    solute_solvent_df = pd.read_csv(path_dataset_for_training)

    # Shuffling, just in case (with random state, of course):
    solute_solvent_df = solute_solvent_df.sample(frac=1, random_state=1).reset_index(drop=True)
    solute_solvent_df.rename(columns={"experimental_logS [mol/L]": "Solubility"}, inplace=True)

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

    dataset_name = "Aq_Solu"
    dataset_name_2 = "AqSolDB"
    if make_pickle is True:
        ########################################## TRAIN ####################################################
        file_path = f"Hyperparameters/{dataset_name}/{feature_type}_{dataset_name_2}_{model}.json"
        with open(file_path, 'r') as f:
            best_params = json.load(f)

        file_path = f"Hyperparameters/{dataset_name}/{feature_type}_{dataset_name_2}_{model}.json"
        with open(file_path, 'r') as f:
            best_params = json.load(f)

        # Defining the models:
        if model == "RF":
            ML_model = RandomForestRegressor(**best_params)
        elif model == "LightGBM":
            ML_model = lgb.LGBMRegressor(**best_params)
        ML_model_fit = ML_model.fit(x_data_scaled, y_data)

        # Saving the model:
        with open(f'Pickled Models/Aq_Solu_Pipeline_Pred/Pipeline_{feature_type}_'
                  f'{dataset_name_2}_{model}.pkl', 'wb') as file:
            pickle.dump({
                'model': ML_model_fit,
            }, file)
        ####################################################################################################

    else:
        ########################################## READ ####################################################
        with open(f'Pickled Models/Aq_Solu_Pipeline_Pred/Pipeline_{feature_type}_'
                  f'{dataset_name_2}_{model}.pkl',
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

    test_df.rename(columns={"experimental_logS [mol/L]": "Solubility"}, inplace=True)
    if "Unnamed: 0" in test_df.columns:
        test_df.drop(columns=["Unnamed: 0"], inplace=True)
    if "temperature" in test_df.columns:
        test_df.drop(columns=["temperature"], inplace=True)

    x_data_for_prediction = test_df.drop(
        columns=["solute_smiles", "solvent_smiles", "Solubility", "solute_InChIKey",
                 "solvent_InChIKey"])
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
        test_prediction_df.to_csv(f"Pipeline Predictions/Aq_Solu/Pipeline_{feature_type}_"
                                  f"{dataset_name_2}_{model}.csv", index=False)
    else:
        test_prediction_df.to_csv(f"Pipeline Predictions/Aq_Solu/Pipeline_PickleRead_{feature_type}_"
                                  f"{dataset_name_2}_{model}.csv", index=False)

    return None
