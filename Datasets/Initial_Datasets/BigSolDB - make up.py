import numpy as np
import pandas as pd

# The point for all of this is to make the BigSolDB file into a file that has the exact same formatting as CombiSolu
# to just use the same code. These are the names of the columns that are being read by the CombiSolu_dataset maker:
# "solute_smiles", "solvent_smiles", "temperature", "experimental_logS [mol/L]"

# These are the columns present in the BigSolDB:
# SMILES,"T,K",Solubility,Solvent,SMILES_Solvent,Source
bigsol_df = pd.read_csv("BigSolDB - preprocessed.csv")[['SMILES', 'T,K', 'Solubility', 'SMILES_Solvent']]

new_column_names = {
    'SMILES': 'solute_smiles',
    'T,K': 'temperature',
    'Solubility': 'experimental_X',
    'SMILES_Solvent': 'solvent_smiles'
}
bigsol_df = bigsol_df.rename(columns=new_column_names)

# Chanegin the X to logX. Otherwise the model will have problems with mini values like 0.0004
def fun_to_apply(row):
    x = row["experimental_X"]
    log_x = np.log10(x)
    return log_x

bigsol_df["experimental_logX"] = bigsol_df.apply(fun_to_apply, axis=1)
bigsol_df.drop(columns=["experimental_X"], inplace=True)

bigsol_df.to_csv("BigSolDB.csv")