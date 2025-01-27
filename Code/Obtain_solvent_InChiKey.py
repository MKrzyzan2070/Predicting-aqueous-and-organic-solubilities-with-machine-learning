import pandas as pd
import json

make_pickle = True
tolerance = 2.5
model = "LightGBM"
feature_type = "MACCS"
dataset_name = "CombiSolu"


prediction_df = pd.read_csv(f"../Pipeline Predictions/Combi_Solu/Pipeline_{feature_type}_"
                              f"{dataset_name}_{model}_{tolerance}.csv")

solvent_InChIKey_list = list(set(prediction_df["Solvent_InChIKey"]))

with open("JSON_files/Solvent_InChIKey_list.json", 'w') as file:
    json.dump(solvent_InChIKey_list, file)
