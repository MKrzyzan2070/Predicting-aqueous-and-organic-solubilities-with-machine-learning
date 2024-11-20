import pandas as pd
from Code import Pipeline_Code

# IMPORTANT
# First BigSolDB.py has to be run with the outlined settings (pipeline)
# Second AqSolDB.py has to be run with the outlined settings (pipeline)
# Those scripts will generate the CSV files of the predictions.
# The CSV files are being read in the following code:
# The predictions can be found in the Pipeline Predictions/Pipeline_combined folder

model = "LightGBM"
feature_type_list = ["GC_MACCS"]
dataset_name_org = "BigSolDB"
dataset_name_aq = "AqSolDB"
tolerance = 5.0
cutoff_1 = 0.03
cutoff_2 = 0.16
cutoff_3 = 0.28

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