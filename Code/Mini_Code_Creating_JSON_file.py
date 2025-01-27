import pandas as pd
import json

def Make_solvent_JSON():
    df = pd.read_csv("Datasets/BigSolDB_Datasets_Processed/GC_BigSolDB_5.0.csv")

    solvent_list = list(set(list(df["solvent_InChIKey"])))
    with open("Code/JSON_files/Solvent_InChIKey_list_BigSolDB.json", 'w') as file:
        json.dump(solvent_list, file, indent=4)