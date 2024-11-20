import pandas as pd
from Code import AqSolDB_analysis
pd.set_option('display.max_columns', None)


######################### Predicting the solubility of organic compounds in organic solvents ##################################
make_pickle = True
model = "LightGBM"
feature_type_list = ["GC_MACCS"]
dataset_name = "AqSolDB"

def molecule_solubility_prediction(make_pickle, model, feature_type_list, dataset_name):
    for feature_type in feature_type_list:
        path_dataset_for_predictions = (f"Datasets/AqSolDB_Datasets_Processed/Dataset_for_Predictions/"
                                        f"Prediction_{feature_type}_{dataset_name}.csv")
        path_dataset_for_training = f"Datasets/AqSolDB_Datasets_Processed/{feature_type}_{dataset_name}.csv"

        # @@@@@@@@ Making the solubility predictions:
        AqSolDB_analysis.Predict_Aquatic_Solubility(path_dataset_for_training, path_dataset_for_predictions,
                                       make_pickle, model, feature_type)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
