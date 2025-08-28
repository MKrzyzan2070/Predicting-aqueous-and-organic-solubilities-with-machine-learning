import pandas as pd
from Code import BigSolDB_analysis
pd.set_option('display.max_columns', None)


def molecule_solubility_prediction(make_pickle, tolerance, model, feature_type_list,
    dataset_name_1, exp_inchikey=None):
    ######################### Predicting the solubility of organic compounds in organic solvents ##################################

    for feature_type in feature_type_list:
        # Handling the usual case:
        if exp_inchikey is None:
            path_dataset_for_predictions = (f"Datasets/{dataset_name_1}_Datasets_Processed/Dataset_for_Predictions/"
                                            f"Prediction_{feature_type}_{dataset_name_1}_{tolerance}.csv")
            path_dataset_for_training = f"Datasets/{dataset_name_1}_Datasets_Processed/{feature_type}_{dataset_name_1}_{tolerance}.csv"

            # @@@@@@@@ Making the solubility predictions:
            BigSolDB_analysis.Predict_Organic_Solubility(path_dataset_for_training, path_dataset_for_predictions,
                                           make_pickle, tolerance, model, feature_type, dataset_name_1)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # Handling the comparison with experiments:
        else:
            path_dataset_for_predictions = (f"Experimental comparison/Datasets for prediction/"
                                            f"For_Prediction_{exp_inchikey}.csv")
            path_dataset_for_training = (f"Experimental comparison/Training datasets/Training_"
                                         f"{feature_type}_BigSolDB_5.0_{exp_inchikey}.csv")
            # @@@@@@@@ Making the solubility predictions:
            BigSolDB_analysis.Predict_Organic_Solubility(path_dataset_for_training, path_dataset_for_predictions,
                                           make_pickle, tolerance, model, feature_type, dataset_name_1)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def analyse(dataset_name_1, dataset_name_2, feature_type, model, tolerance):
    # Reading the CSV:
    prediction_df = pd.read_csv(f"Pipeline Predictions/{dataset_name_2}/Pipeline_{feature_type}_"
                                f"{dataset_name_1}_{model}_{tolerance}.csv")

    # @@@@@@@@ Saving the identified solvents as the PNG picture:
    molecule_smiles_list = list(set(prediction_df["Molecule_smiles"]))
    BigSolDB_analysis.Mol_Inter_Analysis(prediction_df, molecule_smiles_list, feature_type, model, dataset_name_1)
    BigSolDB_analysis.plot_solvents_on_axis(prediction_df, molecule_smiles_list, feature_type, model, dataset_name_1)
    BigSolDB_analysis.plot_solubility_violin(prediction_df, molecule_smiles_list, feature_type, model, dataset_name_1)
