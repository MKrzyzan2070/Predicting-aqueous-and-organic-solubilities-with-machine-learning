import pandas as pd
from Code import UNIFAC_parameters_creation
from rdkit import Chem
from rdkit.Chem import Descriptors
from thermo import Joback
import pubchempy as pcp
from rdkit.Chem import MACCSkeys
import numpy as np
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)  # or use the number of columns df.shape[1]


# Function to ensure that the molecule has only the atoms in the list
def is_organic(smiles):
    allowed_atoms = {'C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'As'}
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_atoms:
            return False
    return True


# In case the SMILES string is not present. This sometimes happens
def remove_when_no_smiles(smiles):
    if smiles == "-":
        return False
    else:
        return True

# That's for handling the multiple entries:
def average_solubility_and_temperature(df):
    aggregation_functions = {
        'solute_InChIKey': 'first',
        'solvent_InChIKey': 'first',
        'experimental_logX': 'mean',
        'temperature': 'mean'
    }

    grouped = df.groupby(['solute_smiles', 'solvent_smiles'])
    averaged_df = grouped.agg(aggregation_functions).reset_index()

    return averaged_df


def make_BigSolDB_datasets(test_set_InChIKey_list, mole_fraction_list, temperature, tolerance):
    # These are the processing steps:
    # 1. Only the datapoints that fall within the temperature range (tolerance) are considered
    # 2. Water is removed from the dataset as the dataset should only contain organic solvents
    # 3. Removing cases when there is no SMILES string which sometimes happens
    # 4. Sometimes there are two different SMILES strinsg for the same molecule and that needs to be made uniform
    # 5. If there are more than 1 solubility value, the final value is the average
    # 6. Inorganic compounds are excluded from the dataset
    # 7. Datapoints for solvents that appear less than 10 times are excluded

    ############ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ############ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #########################################################################################################
    # Loading the initial BigSolDB dataset
    path = "Datasets/Initial_Datasets/BigSolDB.csv"
    GC_df = pd.read_csv(path, usecols=["solute_smiles", "solvent_smiles", "temperature",
                                                   "experimental_logX"])
    #########################################################################################################

    #########################################################################################################
    GC_df = GC_df[
        (GC_df["temperature"] > temperature - tolerance) & (
                GC_df["temperature"] < temperature + tolerance)]
    #########################################################################################################

    #########################################################################################################
    GC_df = GC_df[GC_df["solvent_smiles"] != "O"]
    #########################################################################################################

    #########################################################################################################
    # Deleting the case when there is no SMILES string:
    GC_df['Correct Smiles'] = GC_df['solvent_smiles'].apply(remove_when_no_smiles)
    GC_df = GC_df[GC_df['Correct Smiles']]
    GC_df.drop(columns=['Correct Smiles'], inplace=True)
    GC_df.reset_index(drop=True, inplace=True)

    GC_df['Correct Smiles'] = GC_df['solute_smiles'].apply(remove_when_no_smiles)
    GC_df = GC_df[GC_df['Correct Smiles']]
    GC_df.drop(columns=['Correct Smiles'], inplace=True)
    GC_df.reset_index(drop=True, inplace=True)
    ########################################################################################################

    #########################################################################################################
    # Generating InChIKeys from the SMILES:
    def fun_to_apply(row):
        smiles = row["solute_smiles"]
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol)

    GC_df["solute_InChIKey"] = GC_df.apply(fun_to_apply, axis=1)

    def fun_to_apply(row):
        smiles = row["solvent_smiles"]
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol)

    GC_df["solvent_InChIKey"] = GC_df.apply(fun_to_apply, axis=1)

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

    different_solute_smiles = find_diff_smiles_by_inchikey(GC_df, 'solute_InChIKey', 'solute_smiles')
    different_solvent_smiles = find_diff_smiles_by_inchikey(GC_df, 'solvent_InChIKey', 'solvent_smiles')

    for InChIKey in different_solute_smiles:
        # Let's just take the first SMILES string from the list:
        smiles = different_solute_smiles[InChIKey][0]

        def fun_to_apply(row):
            if row["solute_InChIKey"] == InChIKey:
                return smiles
            else:
                return row["solute_smiles"]

        GC_df["solute_smiles"] = GC_df.apply(fun_to_apply, axis=1)

    for InChIKey in different_solvent_smiles:
        # Let's just take the first SMILES string from the list:
        smiles = different_solvent_smiles[InChIKey][0]

        def fun_to_apply(row):
            if row["solvent_InChIKey"] == InChIKey:
                return smiles
            else:
                return row["solvent_smiles"]

        GC_df["solvent_smiles"] = GC_df.apply(fun_to_apply, axis=1)
    #########################################################################################################

    #########################################################################################################
    # If more than two values are present for a different temperature:
    GC_df = average_solubility_and_temperature(GC_df)
    #########################################################################################################

    #########################################################################################################
    # Filtering off the inorganic compounds for both the solute and the solvent:
    GC_df['IsOrganic'] = GC_df['solvent_smiles'].apply(is_organic)
    GC_df = GC_df[GC_df['IsOrganic']]
    GC_df.drop(columns=['IsOrganic'], inplace=True)
    GC_df.reset_index(drop=True, inplace=True)

    GC_df['IsOrganic'] = GC_df['solute_smiles'].apply(is_organic)
    GC_df = GC_df[GC_df['IsOrganic']]
    GC_df.drop(columns=['IsOrganic'], inplace=True)
    GC_df.reset_index(drop=True, inplace=True)
    #########################################################################################################

    # @@@@@@@@@@@@@@
    ########################################################################################################
    solvent_list = []
    for solvent in list(GC_df["solvent_InChIKey"]):
        solvent_list.append(solvent)
    accepted_solvent_list = []
    for solvent in solvent_list:
        # This can be whatever column  really:
        solute_solvent_pairs_num = len(list(GC_df[GC_df['solvent_InChIKey'] ==
                                                              solvent]['experimental_logX']))


        #################################################################
        # ONLY THE SOLVENTS WHICH APPEAR FREQUENTLY WILL BE CONSIDERED! This is always being done!!!
        exclude_rare_solvents = True
        if exclude_rare_solvents is True:
            if solute_solvent_pairs_num > 9:
                accepted_solvent_list.append(solvent)
        else:
            accepted_solvent_list.append(solvent)
        #################################################################

    GC_df = GC_df[GC_df['solvent_InChIKey'].isin(accepted_solvent_list)]
    ########################################################################################################
    ############ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ############ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    ####### @@@@@@@
    #########################################################################################################
    # The point of this is for the comparison between the prediction and reality:
    # Of course, if the molecule is not present in the dataset then empty exp dataset will be created
    exp_solute_solvent_df = GC_df[GC_df["solute_InChIKey"].isin(test_set_InChIKey_list)]
    exp_solute_solvent_df.to_csv(f"Datasets/BigSolDB_Datasets_Processed/Experimental_DF.csv")
    #########################################################################################################
    ####### @@@@@@@

    ####### @@@@@@@
    #########################################################################################################
    # Ensuring that the training set does not contain the test set - molecules of interest
    # This is a bit redundant at this point, because the features for the molecule of interest
    # will be added later, but it is done nonetheless for the workflow
    GC_df = GC_df[~GC_df["solute_InChIKey"].isin(test_set_InChIKey_list)]
    GC_df = GC_df[~GC_df["solvent_InChIKey"].isin(test_set_InChIKey_list)]
    #########################################################################################################
    ####### @@@@@@@

    #########################################################################################################
    #########################################################################################################
    # Now, the molecules of interest will be added. Later they are going to be excluded, of course
    test_dict = {"solute_smiles": [], "solvent_smiles": [],
                 "temperature": [], "experimental_logX": [],
                 "solute_InChIKey": [], "solvent_InChIKey": []}

    solvent_InChIKey_list = []
    solvent_smiles_list = []
    grouped_solute_solvent_df = GC_df.groupby(["solvent_InChIKey"])
    for inchikey, group in grouped_solute_solvent_df:
        smiles = group["solvent_smiles"].unique()
        solvent_InChIKey_list.append(inchikey)
        solvent_smiles_list.append(str(smiles[0]))

    test_set_smiles_list = []
    for inchikey in test_set_InChIKey_list:
        compound = pcp.get_compounds(inchikey, 'inchikey')
        smiles = compound[0].isomeric_smiles
        test_set_smiles_list.append(smiles)

    i_1 = -1
    for solute_smiles in test_set_smiles_list:
        i_1 += 1
        i_2 = -1
        for solvent_smiles in solvent_smiles_list:
            i_2 += 1

            solute_InChIKey = test_set_InChIKey_list[i_1]
            solvent_InChIKey = solvent_InChIKey_list[i_2]

            test_dict["solute_smiles"].append(solute_smiles)
            test_dict["solute_InChIKey"].append(solute_InChIKey)

            test_dict["solvent_smiles"].append(solvent_smiles)
            test_dict["solvent_InChIKey"].append(solvent_InChIKey)

            test_dict["temperature"].append(temperature)

            # 0 is just a placeholder, of course,
            test_dict["experimental_logX"].append(0)

    test_df = pd.DataFrame(test_dict)

    GC_df = pd.concat([GC_df, test_df])
    GC_df.reset_index(inplace=True, drop=True)

    def fun_to_apply(row):
        if isinstance(row["solvent_InChIKey"], tuple):
            return str(row["solvent_InChIKey"][0])
        else:
            return row["solvent_InChIKey"]
    GC_df["solvent_InChIKey"] = GC_df.apply(fun_to_apply, axis=1)

    def fun_to_apply(row):
        if isinstance(row["solute_InChIKey"], tuple):
            return str(row["solute_InChIKey"][0])
        else:
            return row["solute_InChIKey"]
    GC_df["solute_InChIKey"] = GC_df.apply(fun_to_apply, axis=1)
    #########################################################################################################
    #########################################################################################################

    ####### @@@@@@@
    #########################################################################################################
    # This will be useful for the MACCS later on:
    solute_solvent_df_bare = GC_df.copy()
    # This is good, I checked and it does contain the molecules of interest at the end of the dataframe:
    # print(solute_solvent_df_bare)
    #########################################################################################################
    ####### @@@@@@@

    #########################################################################################################
    # The UNIFAC features are obtained:
    GC_df, UNIFAC_column_list = UNIFAC_parameters_creation.Make_UNIFAC_parameters_from_SMILES_Version1(
        GC_df,
        mole_fraction_list)
    GC_df = GC_df[GC_df[UNIFAC_column_list[0]] != "ERROR"]
    #########################################################################################################

    #########################################################################################################
    molwt_list_solute = []
    mollogp_list_solute = []
    molmr_list_solute = []
    tpsa_list_solute = []
    labuteasa_list_solute = []
    maxpartialcharge_list_solute = []
    minpartialcharge_list_solute = []
    aromaticity_list_solute = []
    numhba_list_solute = []
    numhbd_list_solute = []

    peoe_vsa_lists_solute = {i: [] for i in range(1, 15)}

    molwt_list_solvent = []
    mollogp_list_solvent = []
    molmr_list_solvent = []
    tpsa_list_solvent = []
    labuteasa_list_solvent = []
    maxpartialcharge_list_solvent = []
    minpartialcharge_list_solvent = []
    aromaticity_list_solvent = []
    numhba_list_solvent = []
    numhbd_list_solvent = []

    peoe_vsa_lists_solvent = {i: [] for i in range(1, 15)}

    for index in GC_df.index:
        # Molecule object from RDKit for solute and solvent
        solute_smiles = GC_df.loc[index, "solute_smiles"]
        solvent_smiles = GC_df.loc[index, "solvent_smiles"]
        mol_solute = Chem.MolFromSmiles(solute_smiles)
        mol_solvent = Chem.MolFromSmiles(solvent_smiles)

        # GC for solute:
        molwt_list_solute.append(Descriptors.MolWt(mol_solute))
        mollogp_list_solute.append(Descriptors.MolLogP(mol_solute))
        molmr_list_solute.append(Descriptors.MolMR(mol_solute))
        tpsa_list_solute.append(Descriptors.TPSA(mol_solute))
        labuteasa_list_solute.append(Descriptors.LabuteASA(mol_solute))
        maxpartialcharge_list_solute.append(Descriptors.MaxPartialCharge(mol_solute, force=True))
        minpartialcharge_list_solute.append(Descriptors.MinPartialCharge(mol_solute, force=True))
        aromaticity_list_solute.append(Descriptors.NumAromaticRings(mol_solute))
        numhba_list_solute.append(rdMolDescriptors.CalcNumHBA(mol_solute))
        numhbd_list_solute.append(rdMolDescriptors.CalcNumHBD(mol_solute))

        for i in range(1, 15):
            peoe_vsa_lists_solute[i].append(getattr(Descriptors, f'PEOE_VSA{i}')(mol_solute))

        # GC for solvent:
        molwt_list_solvent.append(Descriptors.MolWt(mol_solvent))
        mollogp_list_solvent.append(Descriptors.MolLogP(mol_solvent))
        molmr_list_solvent.append(Descriptors.MolMR(mol_solvent))
        tpsa_list_solvent.append(Descriptors.TPSA(mol_solvent))
        labuteasa_list_solvent.append(Descriptors.LabuteASA(mol_solvent))
        maxpartialcharge_list_solvent.append(Descriptors.MaxPartialCharge(mol_solvent, force=True))
        minpartialcharge_list_solvent.append(Descriptors.MinPartialCharge(mol_solvent, force=True))
        aromaticity_list_solvent.append(Descriptors.NumAromaticRings(mol_solvent))
        numhba_list_solvent.append(rdMolDescriptors.CalcNumHBA(mol_solvent))
        numhbd_list_solvent.append(rdMolDescriptors.CalcNumHBD(mol_solvent))

        for i in range(1, 15):
            peoe_vsa_lists_solvent[i].append(getattr(Descriptors, f'PEOE_VSA{i}')(mol_solvent))

    GC_df['MolWt_solute'] = molwt_list_solute
    GC_df['MolLogP_solute'] = mollogp_list_solute
    GC_df['MolMR_solute'] = molmr_list_solute
    GC_df['TPSA_solute'] = tpsa_list_solute
    GC_df['LabuteASA_solute'] = labuteasa_list_solute
    GC_df['MaxPartialCharge_solute'] = maxpartialcharge_list_solute
    GC_df['MinPartialCharge_solute'] = minpartialcharge_list_solute
    GC_df['Aromaticity_solute'] = aromaticity_list_solute
    GC_df['NumHBA_solute'] = numhba_list_solute
    GC_df['NumHBD_solute'] = numhbd_list_solute

    for i in range(1, 15):
        GC_df[f'PEOE_VSA{i}_solute'] = peoe_vsa_lists_solute[i]

    GC_df['MolWt_solvent'] = molwt_list_solvent
    GC_df['MolLogP_solvent'] = mollogp_list_solvent
    GC_df['MolMR_solvent'] = molmr_list_solvent
    GC_df['TPSA_solvent'] = tpsa_list_solvent
    GC_df['LabuteASA_solvent'] = labuteasa_list_solvent
    GC_df['MaxPartialCharge_solvent'] = maxpartialcharge_list_solvent
    GC_df['MinPartialCharge_solvent'] = minpartialcharge_list_solvent
    GC_df['Aromaticity_solvent'] = aromaticity_list_solvent
    GC_df['NumHBA_solvent'] = numhba_list_solvent
    GC_df['NumHBD_solvent'] = numhbd_list_solvent

    for i in range(1, 15):
        GC_df[f'PEOE_VSA{i}_solvent'] = peoe_vsa_lists_solvent[i]
    #########################################################################################################

    #########################################################################################################
    # The test set will be the one containing molecules of interest:
    test_solute_solvent_df = GC_df[GC_df["solute_InChIKey"].isin(test_set_InChIKey_list)]
    GC_df = GC_df[~GC_df["solute_InChIKey"].isin(test_set_InChIKey_list)]
    #########################################################################################################

    #########################################################################################################
    # Obtaining the UNIFAC column names:
    basic_column_list = ['solute_smiles', 'solvent_smiles', 'solute_InChIKey', 'solvent_InChIKey',
                   'experimental_logX', 'temperature']
    for UNIFAC_column in UNIFAC_column_list:
        basic_column_list.append(UNIFAC_column)
    #########################################################################################################

    #########################################################################################################
    # Getting other GC:
    GC_df.reset_index(inplace=True)
    GC_list = [
        'solute_smiles', 'solvent_smiles', 'solute_InChIKey', 'solvent_InChIKey', 'experimental_logX',
        'temperature',
        'MolLogP_solute', 'MolMR_solute', 'TPSA_solute', 'MolWt_solute', 'LabuteASA_solute',
        'MaxPartialCharge_solute', 'MinPartialCharge_solute', 'NumHBA_solute', 'NumHBD_solute',
        'MolLogP_solvent', 'MolMR_solvent', 'TPSA_solvent', 'MolWt_solvent', 'LabuteASA_solvent',
        'MaxPartialCharge_solvent', 'MinPartialCharge_solvent', 'NumHBA_solvent', 'NumHBD_solvent',
    ]

    for i in range(1, 15):
        GC_list.append(f'PEOE_VSA{i}_solute')
    for i in range(1, 15):
        GC_list.append(f'PEOE_VSA{i}_solvent')

    for UNIFAC_column in UNIFAC_column_list:
        GC_list.append(UNIFAC_column)

    GC_df = GC_df[GC_list]
    test_GC_df = test_solute_solvent_df[GC_list]

    ####### The training dataset:
    Joback_Hfus_list = []
    Joback_Hvap_list = []
    Joback_Tm_list = []
    Joback_Tb_list = []
    for solvent_smiles in list(GC_df["solute_smiles"]):
        Joback_Hfus_list.append(Joback.Hfus(Joback(solvent_smiles).counts))
        Joback_Hvap_list.append(Joback.Hvap(Joback(solvent_smiles).counts))
        Joback_Tm_list.append(Joback.Tm(Joback(solvent_smiles).counts))
        Joback_Tb_list.append(Joback.Tb(Joback(solvent_smiles).counts))
    GC_df.loc[0:, "Joback_Hfus_solute"] = Joback_Hfus_list
    GC_df.loc[0:, "Joback_Hvap_solute"] = Joback_Hvap_list
    GC_df.loc[0:, "Joback_Tm_solute"] = Joback_Tm_list
    GC_df.loc[0:, "Joback_Tb_solute"] = Joback_Tb_list

    Joback_Hfus_list = []
    Joback_Hvap_list = []
    Joback_Tm_list = []
    Joback_Tb_list = []
    for solvent_smiles in list(GC_df["solvent_smiles"]):
        Joback_Hfus_list.append(Joback.Hfus(Joback(solvent_smiles).counts))
        Joback_Hvap_list.append(Joback.Hvap(Joback(solvent_smiles).counts))
        Joback_Tm_list.append(Joback.Tm(Joback(solvent_smiles).counts))
        Joback_Tb_list.append(Joback.Tb(Joback(solvent_smiles).counts))

    GC_df.loc[0:, "Joback_Hfus_solvent"] = Joback_Hfus_list
    GC_df.loc[0:, "Joback_Hvap_solvent"] = Joback_Hvap_list
    GC_df.loc[0:, "Joback_Tm_solvent"] = Joback_Tm_list
    GC_df.loc[0:, "Joback_Tb_solvent"] = Joback_Tb_list


    Joback_Hfus_list = []
    Joback_Hvap_list = []
    Joback_Tm_list = []
    Joback_Tb_list = []
    for solvent_smiles in list(test_GC_df["solute_smiles"]):
        Joback_Hfus_list.append(Joback.Hfus(Joback(solvent_smiles).counts))
        Joback_Hvap_list.append(Joback.Hvap(Joback(solvent_smiles).counts))
        Joback_Tm_list.append(Joback.Tm(Joback(solvent_smiles).counts))
        Joback_Tb_list.append(Joback.Tb(Joback(solvent_smiles).counts))

    test_GC_df.loc[0:, "Joback_Hfus_solute"] = Joback_Hfus_list
    test_GC_df.loc[0:, "Joback_Hvap_solute"] = Joback_Hvap_list
    test_GC_df.loc[0:, "Joback_Tm_solute"] = Joback_Tm_list
    test_GC_df.loc[0:, "Joback_Tb_solute"] = Joback_Tb_list

    Joback_Hfus_list = []
    Joback_Hvap_list = []
    Joback_Tm_list = []
    Joback_Tb_list = []

    for solvent_smiles in list(test_GC_df["solvent_smiles"]):
        Joback_Hfus_list.append(Joback.Hfus(Joback(solvent_smiles).counts))
        Joback_Hvap_list.append(Joback.Hvap(Joback(solvent_smiles).counts))
        Joback_Tm_list.append(Joback.Tm(Joback(solvent_smiles).counts))
        Joback_Tb_list.append(Joback.Tb(Joback(solvent_smiles).counts))

    test_GC_df.loc[0:, "Joback_Hfus_solvent"] = Joback_Hfus_list
    test_GC_df.loc[0:, "Joback_Hvap_solvent"] = Joback_Hvap_list
    test_GC_df.loc[0:, "Joback_Tm_solvent"] = Joback_Tm_list
    test_GC_df.loc[0:, "Joback_Tb_solvent"] = Joback_Tb_list

    # Removing the molecules fow which the feature was unable to be obtained:
    GC_df.dropna(inplace=True)

    GC_df.to_csv(f"Datasets/BigSolDB_Datasets_Processed/GC_BigSolDB_{tolerance}.csv")
    test_GC_df.to_csv(f"Datasets/BigSolDB_Datasets_Processed/Dataset_for_Predictions/"
                      f"Prediction_GC_BigSolDB_{tolerance}.csv")
    #########################################################################################################

    #########################################################################################################
    # Getting MACCS:

    # First the a new dataframe needs to be created that is essentially the initial GC_df.
    # It will have both the training molecules and the molecules of interest for this to work
    # because the funciton later would otherwise delete the molecules of interest from the solute_solvent_df_bare
    GC_df_concat = pd.concat([GC_df, test_GC_df])

    # Making sure that solute_solvent_bare has the same molecules as present in gc_df after
    # features had been obtained for them. For some molecules, not all feature were able to be obtained:
    solute_solvent_df_bare = solute_solvent_df_bare.reset_index(drop=True)
    GC_df_concat.reset_index(inplace=True)
    GC_df_concat['combined_key'] = GC_df_concat['solute_InChIKey'] + '_' + GC_df_concat[
        'solvent_InChIKey']
    solute_solvent_df_bare['combined_key'] = solute_solvent_df_bare['solute_InChIKey'] + '_' + solute_solvent_df_bare[
        'solvent_InChIKey']
    valid_keys = set(GC_df_concat['combined_key'])
    solute_solvent_df_bare = solute_solvent_df_bare[solute_solvent_df_bare['combined_key'].isin(valid_keys)]
    solute_solvent_df_bare = solute_solvent_df_bare.set_index('combined_key').loc[
        GC_df_concat['combined_key']].reset_index()
    solute_solvent_df_bare = solute_solvent_df_bare.drop(columns=['combined_key'])
    solute_solvent_df_bare = solute_solvent_df_bare.reset_index(drop=True)
    ###### @@@@

    MACCS_df = solute_solvent_df_bare[
        ['solute_smiles', 'solvent_smiles', 'solute_InChIKey', 'solvent_InChIKey', 'experimental_logX']].copy()
    fingerprint_length = len(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles("O")))

    def calculate_fingerprints(row):
        fingerprints = np.zeros(2 * fingerprint_length, dtype=int)  # Double length to accommodate both fingerprints
        # Generate fingerprint for the solute
        mol_solute = Chem.MolFromSmiles(row["solute_smiles"])
        if mol_solute:
            fingerprint_solute = np.array(MACCSkeys.GenMACCSKeys(mol_solute), dtype=int)
            fingerprints[:fingerprint_length] = fingerprint_solute

        # Generate fingerprint for the solvent
        mol_solvent = Chem.MolFromSmiles(row["solvent_smiles"])
        if mol_solvent:
            fingerprint_solvent = np.array(MACCSkeys.GenMACCSKeys(mol_solvent), dtype=int)
            fingerprints[fingerprint_length:] = fingerprint_solvent

        return fingerprints

    # Apply the function to each row
    MACCS_df['Fingerprints'] = MACCS_df.apply(calculate_fingerprints, axis=1)
    fingerprint_data = np.stack(MACCS_df['Fingerprints'].values)
    column_names = [f"SoluteFingerprint{i}" for i in range(fingerprint_length)] + [f"SolventFingerprint{i}" for i in
                                                                                   range(fingerprint_length)]
    fingerprint_df = pd.DataFrame(fingerprint_data, columns=column_names)
    MACCS_df.drop(columns=['Fingerprints'], inplace=True)
    MACCS_df = pd.concat([MACCS_df, fingerprint_df], axis=1)

    # Ensuring that the training set does not contain the test set - molecules of interest
    test_MACCS_df = MACCS_df[MACCS_df["solute_InChIKey"].isin(test_set_InChIKey_list)]
    MACCS_df = MACCS_df[~MACCS_df["solute_InChIKey"].isin(test_set_InChIKey_list)]

    # So the MACCS_df works but the test_MACCS is wrong. Let's see it
    MACCS_df.to_csv(f"Datasets/BigSolDB_Datasets_Processed/MACCS_BigSolDB_{tolerance}.csv", index=False)
    test_MACCS_df.to_csv(f"Datasets/BigSolDB_Datasets_Processed/Dataset_for_Predictions/"
                      f"Prediction_MACCS_BigSolDB_{tolerance}.csv")
    #########################################################################################################

    #########################################################################################################
    # Lastly, the GC_MACCS:
    GC_MACCS_df = pd.merge(GC_df, MACCS_df, how='left')
    test_GC_MACCS_df = pd.merge(test_GC_df, test_MACCS_df, how='left')

    GC_MACCS_df.to_csv(f"Datasets/BigSolDB_Datasets_Processed/GC_MACCS_BigSolDB_{tolerance}.csv")
    test_GC_MACCS_df.to_csv(f"Datasets/BigSolDB_Datasets_Processed/Dataset_for_Predictions/"
                      f"Prediction_GC_MACCS_BigSolDB_{tolerance}.csv")
    #########################################################################################################
