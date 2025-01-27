import pandas as pd
from Code import Pipeline_Code

# I M P O R T A N T !!!!!
# The following code is mainly just for making the images and plots to visualise the identification
# of co-solvents. As such, BigSolDB.py and AqSolDB has to be run beforehand to generate relevant datasets.
# One must make sure that correct settings are chosen before running those scripts. The different, possible settings
# are given at the beginning of the file and the setting corresponding to the pipeline must be declared.

# T H E    S T E P S
# First, run the BigSolDB.py with the outlined settings (pipeline)
# Second, run the AqSolDB.py with the outlined settings (pipeline)
# Third, run this script
# The predictions can be found in the Prediction datasets Predictions/Pipeline_combined folder


model = "LightGBM"
feature_type_list = ["GC_MACCS"]
dataset_name_org = "BigSolDB"
dataset_name_aq = "AqSolDB"
tolerance = 5.0

# This is the cutoff obtained from the immiscibility/miscibility analysis
cutoff_1 = -0.34
cutoff_2 = 0.08
cutoff_3 = 0.44

for feature_type in feature_type_list:
    organic_solubility_prediction_df = pd.read_csv(f"Pipeline Predictions/BigSolDB/Pipeline_{feature_type}_"
                                  f"{dataset_name_org}_{model}_{tolerance}.csv")
    aquatic_solubility_prediction_df = pd.read_csv(f"Pipeline Predictions/Aq_Solu/Pipeline_{feature_type}_"
                                  f"{dataset_name_aq}_{model}.csv")

    molecule_smiles_list = list(set(organic_solubility_prediction_df["Molecule_smiles"]))

    #Filtering immiscible co-solvents:
    Pipeline_Code.Filter_immiscible_cosolvents(molecule_smiles_list, organic_solubility_prediction_df, aquatic_solubility_prediction_df,
                                feature_type, model, cutoff_1, cutoff_2, cutoff_3)

    # Drawing the water miscible co-solvents:
    Pipeline_Code.Draw_co_solvent_molecules(molecule_smiles_list, feature_type, model)

    # Making the co-solvent plot:
    Pipeline_Code.Make_co_solvent_ranking_plot(molecule_smiles_list, feature_type, model)

    #Highlighting the solvents for experimental study:
    Pipeline_Code.Solvent_selection_visalisation(molecule_smiles_list, feature_type, model, cutoff_1, cutoff_2, cutoff_3)