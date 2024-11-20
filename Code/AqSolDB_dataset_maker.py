import pandas as pd
from Code import UNIFAC_parameters_creation
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
import numpy as np
from rdkit.Chem import MACCSkeys
from thermo import Joback
import pubchempy as pcp
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors

# A comment regarding the SettingWithCopyWarning:
# I use similar code to the CombiSolu dataset maker. No warning messages there
# but messages here. It does not cause major issues, so it's going to be left like that.
pd.options.mode.chained_assignment = None

def is_organic(smiles):
    allowed_atoms = set(['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'As'])
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_atoms:
            return False
    return True

def make_AqSolDB_datasets(test_set_InChIKey_list, mole_fraction_list):

    #########################################################################################################
    # Loading the initial AqSolDB dataset
    path = "Datasets/Initial_Datasets/AqSolDB-curated-solubility-dataset.csv"
    Solubility_df = pd.read_csv(path)
    Solubility_df = Solubility_df[["Solubility", "SMILES", "InChIKey"]]
    #########################################################################################################

    #########################################################################################################
    # ALREADY HAVE THE INCHIKEY FOR THE MOLECULES!!!
    #########################################################################################################

    #########################################################################################################
    # VERY IMPORTANT!!! The same molecule sometimes has different SMILES strings which the causes problems
    # Function to find and save different SMILES strings for identical InChIKeys
    def find_diff_smiles_by_inchikey(df, inchikey_column, smiles_column):
        diff_smiles_dict = {}
        grouped = df.groupby(inchikey_column)
        for inchikey, group in grouped:
            unique_smiles = group[smiles_column].unique()
            if len(unique_smiles) > 1:
                diff_smiles_dict[inchikey] = list(unique_smiles)
        return diff_smiles_dict
    #########################################################################################################

    #########################################################################################################
    different_solute_smiles = find_diff_smiles_by_inchikey(Solubility_df,
                                                           'InChIKey', 'SMILES')

    for InChIKey in different_solute_smiles:
        # Let's just take the first SMILES string from the list:
        smiles = different_solute_smiles[InChIKey][0]
        def fun_to_apply(row):
            if row["InChIKey"] == InChIKey:
                return smiles
            else:
                return row["SMILES"]
        Solubility_df["SMILES"] = Solubility_df.apply(fun_to_apply, axis=1)
    #########################################################################################################

    #########################################################################################################
    # Ensuring that the training set does not contain the test set - molecules of interest
    Solubility_df = Solubility_df[~Solubility_df["InChIKey"].isin(test_set_InChIKey_list)]
    #########################################################################################################

    #########################################################################################################
    # Filtering off the inorganic compounds:
    Solubility_df['IsOrganic'] = Solubility_df['SMILES'].apply(is_organic)
    Solubility_df = Solubility_df[Solubility_df['IsOrganic']]
    Solubility_df.drop(columns=['IsOrganic'], inplace=True)
    Solubility_df.reset_index(drop=True, inplace=True)
    #########################################################################################################

    #########################################################################################################
    # Just changing the name of the variable dataframe
    solute_solvent_df = Solubility_df.copy()
    #########################################################################################################

    #########################################################################################################
    # Now, the molecules of interest will be added. Later they are going to be excluded, of course
    smiles_list = []
    Solubility_list = []

    for InChIKey in  test_set_InChIKey_list:
        Solubility_list.append(0)
        compound = pcp.get_compounds(InChIKey, 'inchikey')
        smiles = compound[0].isomeric_smiles
        smiles_list.append(smiles)

    test_df = pd.DataFrame({"InChIKey": test_set_InChIKey_list, "Solubility": Solubility_list, "SMILES": smiles_list
                       })
    solute_solvent_df = pd.concat([solute_solvent_df, test_df])
    solute_solvent_df.reset_index(inplace=True, drop=True)
    #########################################################################################################

    #########################################################################################################
    # AqSolDB must be changed to accommodate for the UNIFAC calculations
    solute_solvent_df.rename(columns={"SMILES": "solute_smiles"}, inplace=True)
    #Solvent smiles is just water in this case, which has SMILES string of "O"
    solute_solvent_df["solvent_smiles"] = ["O" for i in range(len(solute_solvent_df))]
    #Temperature is assumed to be constant and equal to the room temperature. This step is necessary for UNIFAC
    solute_solvent_df["temperature"] = [298.15 for i in range(len(solute_solvent_df))]
    solute_solvent_df.rename(columns={"InChIKey": "solute_InChIKey"}, inplace=True)
    solute_solvent_df["solvent_InChIKey"] = list("XLYOFNOQVPJJNP-UHFFFAOYSA-N" for i in range(len(solute_solvent_df)))
    #########################################################################################################

    ####### @@@@@@@
    # This will be useful for the fingerprints later on:
    solute_solvent_df_bare = solute_solvent_df.copy()
    ####### @@@@@@@

    #########################################################################################################
    #The UNIFAC features are obtained:
    solute_solvent_df, UNIFAC_column_list = UNIFAC_parameters_creation.Make_UNIFAC_parameters_from_SMILES_Version1(solute_solvent_df,
                                                                                               mole_fraction_list)

    # length_before = len(solute_solvent_df)
    solute_solvent_df = solute_solvent_df[solute_solvent_df[UNIFAC_column_list[0]] != "ERROR"]
    # length_after = len(solute_solvent_df)
    # print(f"Information: \n {round((length_before - length_after)*100/length_before, 1)}% of data has been lost when "
    #       f"obtaining UNIFAC features for the dataset")
    #########################################################################################################

    #########################################################################################################
    molwt_list = []
    mollogp_list = []
    molmr_list = []
    tpsa_list = []
    labuteasa_list = []
    maxpartialcharge_list = []
    minpartialcharge_list = []
    numhba_list = []
    numhbd_list = []

    # Initialize lists for each PEOE_VSA descriptor
    peoe_vsa_lists = {i: [] for i in range(1, 15)}

    for index in solute_solvent_df.index:
        solute_smiles = solute_solvent_df.loc[index, "solute_smiles"]
        mol_solute = Chem.MolFromSmiles(solute_smiles)

        if mol_solute is None:
            continue

        molwt_list.append(Descriptors.MolWt(mol_solute))
        mollogp_list.append(Crippen.MolLogP(mol_solute))
        molmr_list.append(Descriptors.MolMR(mol_solute))
        tpsa_list.append(Descriptors.TPSA(mol_solute))
        labuteasa_list.append(Descriptors.LabuteASA(mol_solute))
        maxpartialcharge_list.append(Descriptors.MaxPartialCharge(mol_solute, force=True))
        minpartialcharge_list.append(Descriptors.MinPartialCharge(mol_solute, force=True))
        numhba_list.append(rdMolDescriptors.CalcNumHBA(mol_solute))
        numhbd_list.append(rdMolDescriptors.CalcNumHBD(mol_solute))

        for i in range(1, 15):
            peoe_vsa_lists[i].append(getattr(Descriptors, f'PEOE_VSA{i}')(mol_solute))

    solute_solvent_df['MolWt'] = molwt_list
    solute_solvent_df['MolLogP'] = mollogp_list
    solute_solvent_df['MolMR'] = molmr_list
    solute_solvent_df['TPSA'] = tpsa_list
    solute_solvent_df['LabuteASA'] = labuteasa_list
    solute_solvent_df['MaxPartialCharge'] = maxpartialcharge_list
    solute_solvent_df['MinPartialCharge'] = minpartialcharge_list
    solute_solvent_df['NumHBA'] = numhba_list
    solute_solvent_df['NumHBD'] = numhbd_list

    for i in range(1, 15):
        solute_solvent_df[f'PEOE_VSA{i}'] = peoe_vsa_lists[i]
    #########################################################################################################

    #########################################################################################################
    # Ensuring that the training set does not contain the test set - molecules of interest
    test_solute_solvent_df = solute_solvent_df[solute_solvent_df["solute_InChIKey"].isin(test_set_InChIKey_list)]
    solute_solvent_df = solute_solvent_df[~solute_solvent_df["solute_InChIKey"].isin(test_set_InChIKey_list)]
    #########################################################################################################

    #########################################################################################################
    # Obtaining UNIFAC:
    # This step is not done for pipeline.
    UNIFAC_list = ['solute_smiles', 'solvent_smiles', 'solute_InChIKey', 'solvent_InChIKey', 'Solubility']
    for UNIFAC_column in UNIFAC_column_list:
        if "gamma" in UNIFAC_column and "solvent" in UNIFAC_column:
            continue
        else:
            UNIFAC_list.append(UNIFAC_column)

    #########################################################################################################
    
    #########################################################################################################
    # Getting other GC features:
    GC_list = [
        'solute_smiles', 'solvent_smiles', 'solute_InChIKey', 'solvent_InChIKey',
        'Solubility', 'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
        'MaxPartialCharge', 'MinPartialCharge', 'NumHBA', 'NumHBD'
    ]

    for i in range(1, 15):
        GC_list.append(f'PEOE_VSA{i}')

    for UNIFAC_column in UNIFAC_column_list:
        GC_list.append(UNIFAC_column)
    
    GC_df = solute_solvent_df[GC_list]
    test_GC_df = test_solute_solvent_df[GC_list]

    Joback_Hfus_list = []
    Joback_Hvap_list = []
    Joback_Tm_list = []
    Joback_Tb_list = []

    for solute_smiles in list(GC_df["solute_smiles"]):
        Joback_Hfus_list.append(Joback.Hfus(Joback(solute_smiles).counts))
        Joback_Hvap_list.append(Joback.Hvap(Joback(solute_smiles).counts))
        Joback_Tm_list.append(Joback.Tm(Joback(solute_smiles).counts))
        Joback_Tb_list.append(Joback.Tb(Joback(solute_smiles).counts))

    GC_df.loc[0:, "Joback_Hfus"] = Joback_Hfus_list
    GC_df.loc[0:, "Joback_Hvap"] = Joback_Hvap_list
    GC_df.loc[0:, "Joback_Tm"] = Joback_Tm_list
    GC_df.loc[0:, "Joback_Tb"] = Joback_Tb_list

    Joback_Hfus_list = []
    Joback_Hvap_list = []
    Joback_Tm_list = []
    Joback_Tb_list = []

    for solute_smiles in list(test_GC_df["solute_smiles"]):
        Joback_Hfus_list.append(Joback.Hfus(Joback(solute_smiles).counts))
        Joback_Hvap_list.append(Joback.Hvap(Joback(solute_smiles).counts))
        Joback_Tm_list.append(Joback.Tm(Joback(solute_smiles).counts))
        Joback_Tb_list.append(Joback.Tb(Joback(solute_smiles).counts))

    test_GC_df.loc[0:, "Joback_Hfus"] = Joback_Hfus_list
    test_GC_df.loc[0:, "Joback_Hvap"] = Joback_Hvap_list
    test_GC_df.loc[0:, "Joback_Tm"] = Joback_Tm_list
    test_GC_df.loc[0:, "Joback_Tb"] = Joback_Tb_list

    #print(GC_df.columns)
    # Removing the molecules fow which the feature was unable to be obtained:
    GC_df.dropna(inplace=True)

    GC_df.to_csv("Datasets/AqSolDB_Datasets_Processed/GC_AqSolDB.csv")
    test_GC_df.to_csv("Datasets/AqSolDB_Datasets_Processed/Dataset_for_Predictions/"
                                "Prediction_GC_AqSolDB.csv")
    #########################################################################################################

    #########################################################################################################
    #########################################################################################################
    # Getting MACCS:
    # Making sure that solute_solvent_bare has the same molecules as present in gc_df after UNIFAC
    # features had been obtained for them:
    solute_solvent_df_bare = solute_solvent_df_bare.reset_index(drop=True)
    GC_df.reset_index(inplace=True)
    GC_df['combined_key'] = GC_df['solute_InChIKey'] + '_' + GC_df[
        'solvent_InChIKey']
    solute_solvent_df_bare['combined_key'] = solute_solvent_df_bare['solute_InChIKey'] + '_' + solute_solvent_df_bare[
        'solvent_InChIKey']
    valid_keys = set(GC_df['combined_key'])
    solute_solvent_df_bare = solute_solvent_df_bare[solute_solvent_df_bare['combined_key'].isin(valid_keys)]
    solute_solvent_df_bare = solute_solvent_df_bare.set_index('combined_key').loc[
        GC_df['combined_key']].reset_index()
    solute_solvent_df_bare = solute_solvent_df_bare.drop(columns=['combined_key'])
    solute_solvent_df_bare = solute_solvent_df_bare.reset_index(drop=True)

    GC_df.drop(columns={"combined_key"}, inplace=True)
    ###### @@@@

    MACCS_df = solute_solvent_df_bare[
        ['solute_smiles', 'solvent_smiles', 'solute_InChIKey', 'solvent_InChIKey', 'Solubility']].copy()
    fingerprint_length = len(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles("O")))

    def calculate_fingerprints(row):
        fingerprints = np.zeros(fingerprint_length, dtype=int)  # Double length to accommodate both fingerprints
        # Generate fingerprint for the solute
        mol_solute = Chem.MolFromSmiles(row["solute_smiles"])
        if mol_solute:
            fingerprint_solute = np.array(MACCSkeys.GenMACCSKeys(mol_solute), dtype=int)
            fingerprints[:fingerprint_length] = fingerprint_solute
        return fingerprints

    # Applying the function to each row
    MACCS_df['Fingerprints'] = MACCS_df.apply(calculate_fingerprints, axis=1)
    fingerprint_data = np.stack(MACCS_df['Fingerprints'].values)
    column_names = [f"SoluteFingerprint{i}" for i in range(fingerprint_length)]
    fingerprint_df = pd.DataFrame(fingerprint_data, columns=column_names)
    MACCS_df.drop(columns=['Fingerprints'], inplace=True)
    MACCS_df = pd.concat([MACCS_df, fingerprint_df], axis=1)

    # Ensuring that the training set does not contain the test set - molecules of interest
    test_MACCS_df = MACCS_df[MACCS_df["solute_InChIKey"].isin(test_set_InChIKey_list)]
    MACCS_df = MACCS_df[~MACCS_df["solute_InChIKey"].isin(test_set_InChIKey_list)]

    MACCS_df.to_csv(f"Datasets/AqSolDB_Datasets_Processed/MACCS_AqSolDB.csv", index=False)
    test_MACCS_df.to_csv(f"Datasets/AqSolDB_Datasets_Processed/Dataset_for_Predictions/"
                         f"Prediction_MACCS_AqSolDB.csv")
    #########################################################################################################
    #########################################################################################################

    #########################################################################################################
    # CF and GC - ultimate fingerprints:
    GC_MACCS_df = pd.merge(GC_df, MACCS_df, how='left')
    test_GC_MACCS_df = pd.merge(test_GC_df, test_MACCS_df, how='left')

    GC_MACCS_df.to_csv(f"Datasets/AqSolDB_Datasets_Processed/GC_MACCS_AqSolDB.csv")
    test_GC_MACCS_df.to_csv(f"Datasets/AqSolDB_Datasets_Processed/Dataset_for_Predictions/"
                      f"Prediction_GC_MACCS_AqSolDB.csv")
    #########################################################################################################



