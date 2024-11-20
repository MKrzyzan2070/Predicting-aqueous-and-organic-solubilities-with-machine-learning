from Code import Kendall, BigSolDB_dataset_creation_and_analysis, BigSolDB_molecule_solubility_prediction
import numpy as np
from sklearn.metrics import mean_squared_error
import shutil
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

gc_maccs_df = pd.read_csv("Datasets/BigSolDB_Datasets_Processed/GC_MACCS_BigSolDB_5.0.csv")

#   I N F O R M A T I O N !!!!
# Because this code generates a new dataset for each of the case-study molecules, one must regenerate the datasets when
# performing the train/test/validation analysis. Basically the reason behind is similar to when making the pipeline
# predictions. The molecule must be excluded from the dataset. The difference is that I am not excluding all the 30
# molecules from the dataset, because that would remove too many datapoints. Instead, a new dataset is created for
# each case study molecule/.

def calculate_descriptors(mol):
    total_atoms = mol.GetNumHeavyAtoms()

    aliphatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.IsInRing() == False and atom.GetAtomicNum() != 1)
    aliphatic_ratio = aliphatic_atoms / total_atoms if total_atoms > 0 else 0
    return {
        'NonHAtomCount': total_atoms,
        'AliphaticRatio': aliphatic_ratio,
        }

def remove_fragments(row):
    smiles = row["solute_smiles"]
    if "." in smiles:
        return False
    else:
        return True

gc_maccs_df['check'] = gc_maccs_df.apply(remove_fragments, axis=1)
gc_maccs_df = gc_maccs_df[gc_maccs_df["check"] == True]

def count_atoms(row):
    smiles = row["solute_smiles"]
    mol = Chem.MolFromSmiles(smiles)
    total_atoms = mol.GetNumHeavyAtoms()
    return total_atoms

gc_maccs_df["NonHAtomCount"] = gc_maccs_df.apply(count_atoms, axis=1)
gc_maccs_df = gc_maccs_df[gc_maccs_df['NonHAtomCount'] <= 15]

def get_aliphatic_ratio(row):
    smiles = row["solute_smiles"]
    mol = Chem.MolFromSmiles(smiles)
    total_atoms = mol.GetNumHeavyAtoms()
    aliphatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.IsInRing() == False and atom.GetAtomicNum() != 1)
    aliphatic_ratio = aliphatic_atoms / total_atoms if total_atoms > 0 else 0
    return aliphatic_ratio

gc_maccs_df['AliphaticRatio'] = gc_maccs_df.apply(get_aliphatic_ratio, axis=1)

accepted_smiles_list = []
for smiles in list(gc_maccs_df["solute_smiles"]):
    the_number = len(gc_maccs_df[gc_maccs_df["solute_smiles"] == smiles])
    if the_number < 15:
        accepted_smiles_list.append(smiles)

def filter_solute_solvent_pairs(row):
    smiles = row["solute_smiles"]
    if smiles in accepted_smiles_list:
        return True
    else:
        return False
gc_maccs_df["Check"] = gc_maccs_df.apply(filter_solute_solvent_pairs, axis=1)
gc_maccs_df = gc_maccs_df[gc_maccs_df["Check"] == True]

def has_long_aliphatic_chain(mol, chain_length=6):
    for atom in mol.GetAtoms():
        if not atom.IsInRing():
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, chain_length, atom.GetIdx())
            if len(env) >= chain_length:
                submol = Chem.PathToSubmol(mol, env)
                if all(atom.GetDegree() <= 2 for atom in submol.GetAtoms()):
                    return True
    return False

def count_atoms_and_check_chain(row):
    smiles = row["solute_smiles"]
    mol = Chem.MolFromSmiles(smiles)
    total_atoms = mol.GetNumHeavyAtoms()
    has_long_chain = has_long_aliphatic_chain(mol)
    return pd.Series({'NonHAtomCount': total_atoms, 'HasLongAliphaticChain': has_long_chain})

gc_maccs_df[['NonHAtomCount', 'HasLongAliphaticChain']] = gc_maccs_df.apply(count_atoms_and_check_chain, axis=1)

gc_maccs_df = gc_maccs_df[(gc_maccs_df['NonHAtomCount'] <= 15) & (~gc_maccs_df['HasLongAliphaticChain'])]

gc_maccs_df = gc_maccs_df.drop_duplicates(subset=['solute_InChIKey'])
gc_maccs_df = gc_maccs_df.reset_index(drop=True)

gc_maccs_df['MolLogP_solute_binned'] = pd.qcut(gc_maccs_df['MolLogP_solute'], q=2, duplicates='drop')
gc_maccs_df['AliphaticRatio_binned'] = pd.qcut(gc_maccs_df['AliphaticRatio'], q=4, duplicates='drop')

gc_maccs_df['strata'] = (gc_maccs_df['MolLogP_solute_binned'].astype(str) + '_' +
                         gc_maccs_df['AliphaticRatio_binned'].astype(str))

# I prefer to have the indices from the train/test splitting:
indices_list = np.arange(gc_maccs_df.shape[0])

train_idx, test_idx, y_train, y_test = train_test_split(
    indices_list,
    gc_maccs_df['strata'],
    test_size=30,
    stratify=gc_maccs_df['strata'],
    random_state=1
)
# Getting the "test_df":
stratified_sample_df = gc_maccs_df.iloc[test_idx]

smiles_list = list(stratified_sample_df["solute_smiles"])
mols = [Chem.MolFromSmiles(smile) for smile in smiles_list if smile is not None]

img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200), legends=[Chem.MolToSmiles(mol) for mol in mols])
img.save("Case-Study molecules.png")
############################################################################


metrics_df = pd.DataFrame({'InChIKey': [],
                           'Feature_Type': [],
                           'Kendall_Distance': [],
                           'RMSE': []})
################ @@@@@@@
feature_type_list = ["GC_MACCS"]
for feature_type in feature_type_list:
    for smiles in smiles_list:
        if feature_type == "GC-MACCS":
            feature_type = "GC_MACCS"
        ###########################################################################################################
        ########## @@@@@
        temperature = 298.15
        tolerance = 5.0
        make_pickle = False
        make_BigSolDB_dataset = True
        model = "LightGBM"
        do_PCA = False
        dataset = "BigSolDB"
        test_set_InChIKey_list = []

        gc_maccs_df = pd.read_csv("Datasets/BigSolDB_Datasets_Processed/GC_MACCS_BigSolDB_5.0.csv")
        exp_inchikey = str(gc_maccs_df[gc_maccs_df["solute_smiles"] == smiles]["solute_InChIKey"].iloc[0])
        BigSolDB_dataset_creation_and_analysis.dataset_creation_and_analysis(exp_inchikey, temperature,
                                                                             tolerance, make_pickle,
                                                                             make_BigSolDB_dataset,
                                                                             model, do_PCA, dataset,
                                                                             test_set_InChIKey_list, do_analysis=True)
        ########## @@@@@

        ########## @@@@@
        make_pickle = True
        tolerance = 5.0
        model = "LightGBM"
        dataset_name_1 = "BigSolDB"
        dataset_name_2 = "BigSolDB"
        BigSolDB_molecule_solubility_prediction.molecule_solubility_prediction(make_pickle, tolerance, model,
                                                                               feature_type_list, dataset_name_1)
        ########## @@@@@

        shutil.copy(f"Datasets/BigSolDB_Datasets_Processed/{feature_type}_BigSolDB_5.0.csv",
                    f"Experimental comparison datasets/Training/Training_{feature_type}_"
                    f"BigSolDB_5.0_{exp_inchikey}.csv")
        shutil.copy(f"Datasets/BigSolDB_Datasets_Processed/Experimental_DF.csv",
                    f"Experimental comparison datasets/Experimental/Experimental_DF_{exp_inchikey}.csv")
        shutil.copy(f"Pipeline Predictions/BigSolDB/Pipeline_{feature_type}_BigSolDB_LightGBM_5.0.csv",
                    f"Experimental comparison datasets/Pipeline/Pipeline_{feature_type}_"
                    f"BigSolDB_LightGBM_5.0_{exp_inchikey}.csv")
        ###############################################################################################################

        exp_BigSolDB_df = pd.read_csv(f"Experimental comparison datasets/Experimental/Experimental_DF_{exp_inchikey}.csv")
        exp_BigSolDB_df.drop(columns="Unnamed: 0", inplace=True)
        pred_BigSolDB_df = pd.read_csv(f"Experimental comparison datasets/"
                                       f"Pipeline/Pipeline_{feature_type}_BigSolDB_LightGBM_5.0_{exp_inchikey}.csv")

        exp_BigSolDB_df = exp_BigSolDB_df[exp_BigSolDB_df["solute_InChIKey"] == exp_inchikey]
        mol_exp_solubility_values = list(exp_BigSolDB_df["experimental_logX"])
        smiles_exp_solvents = list(exp_BigSolDB_df["solvent_smiles"])
        inchikey_exp_solvents = list(exp_BigSolDB_df["solvent_InChIKey"])

        exp_BigSolDB_df.drop(columns="temperature", inplace=True)
        exp_BigSolDB_df.rename(columns={'solute_smiles': 'Molecule_smiles', 'solvent_smiles': 'Solvent_smiles',
                                        'solute_InChIKey': 'Molecule_InChIKey', 'solvent_InChIKey': 'Solvent_InChIKey'},
                               inplace=True)

        solvent_count_df = pd.DataFrame({"Solvent_smiles": smiles_exp_solvents})
        count_list = []
        for smiles_exp_solvent in smiles_exp_solvents:
            training_df = pd.read_csv(f"Experimental comparison datasets/Training/"
                                      f"Training_{feature_type}_BigSolDB_5.0_{exp_inchikey}.csv")
            count_num = len(training_df[training_df["solvent_smiles"] == smiles_exp_solvent])
            count_list.append(count_num)
        solvent_count_df["Solvent count"] = count_list
        solvent_count_df.to_csv(f"Experimental comparison datasets/Solvent_count/Solvent_count_"
                                f"train_dataset_{exp_inchikey}.csv")

        pred_BigSolDB_df = pred_BigSolDB_df[pred_BigSolDB_df["Molecule_InChIKey"] == exp_inchikey]
        pred_BigSolDB_df = pred_BigSolDB_df[pred_BigSolDB_df["Solvent_InChIKey"].isin(inchikey_exp_solvents)]
        pred_BigSolDB_df['Solvent_InChIKey'] = pd.Categorical(pred_BigSolDB_df['Solvent_InChIKey'],
                                                              categories=inchikey_exp_solvents,
                                                              ordered=True)
        pred_BigSolDB_df = pred_BigSolDB_df.sort_values('Solvent_InChIKey')
        pred_BigSolDB_df.drop(columns="Molecule_smiles", inplace=True)

        exp_pred_BigSolDB_df = pd.merge(exp_BigSolDB_df, pred_BigSolDB_df, how="inner",
                                        on=["Molecule_InChIKey", "Solvent_InChIKey", "Solvent_smiles"])
        ##### Now making the image:
        pred_BigSolDB_df.sort_values(by="Solubility Prediction", ascending=False, inplace=True)
        pred_solvent_smiles_list = list(pred_BigSolDB_df["Solvent_smiles"])
        prediction_solubility_list = list(pred_BigSolDB_df["Solubility Prediction"])

        exp_BigSolDB_df.sort_values(by="experimental_logX", ascending=False, inplace=True)
        exp_solvent_smiles_list = list(exp_BigSolDB_df["Solvent_smiles"])
        experimental_solubility_list = list(exp_BigSolDB_df["experimental_logX"])

        # Calculating the Kendall Tau distance:
        exp_solubility_rank = [1 + i for i in range(len(exp_solvent_smiles_list))]
        pred_solubility_rank = []
        for exp_solvent_smiles in exp_solvent_smiles_list:
            pred_index = pred_solvent_smiles_list.index(exp_solvent_smiles)
            pred_solubility_rank.append(pred_index + 1)

        kendall = round(Kendall.kendall_top_k(np.array(exp_solubility_rank), np.array(pred_solubility_rank)), 3)



        ###########################################################
        ########### Creating the figures:

        ###### The full figure @@@@@@
        ####################################################################################################
        ####################################################################################################
        fig = plt.figure(figsize=(len(pred_solvent_smiles_list) * 2, 9))  # Adjust figure height for title space
        width_ratios = [1 for _ in range(len(pred_solvent_smiles_list))]
        width_ratios.extend([0.2, 2.3])
        grid = plt.GridSpec(2, len(pred_solvent_smiles_list) + 2, width_ratios=width_ratios, wspace=0.2,
                            hspace=0.11, figure=fig)

        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03)

        rmse = round(np.sqrt(mean_squared_error(experimental_solubility_list, prediction_solubility_list)), 3)

        # Add titles for "Experimental Values" and "Predicted Values"
        fig.text(0.4, 0.88, "Experimental Values", fontsize=26, ha='center', color='#8B0000', fontweight='bold')
        fig.text(0.4, 0.43, "Predicted Values", fontsize=26, ha='center', color='#8B0000', fontweight='bold')

        for i, (solvent_smiles, solubility) in enumerate(zip(pred_solvent_smiles_list, experimental_solubility_list)):
            ax = fig.add_subplot(grid[0, i])
            mol = Chem.MolFromSmiles(solvent_smiles)
            img = Draw.MolToImage(mol, size=(400, 400))
            ax.imshow(img)
            ax.set_title(f"{solubility:.3f}", fontsize=18)
            ax.axis('off')

        for i, (solvent_smiles, prediction) in enumerate(zip(exp_solvent_smiles_list, prediction_solubility_list)):
            ax = fig.add_subplot(grid[1, i])
            mol = Chem.MolFromSmiles(solvent_smiles)
            img = Draw.MolToImage(mol, size=(400, 400))
            ax.imshow(img)
            ax.set_title(f"{prediction:.3f}", fontsize=18)
            ax.axis('off')

        if feature_type == "GC_MACCS":
            feature_type = "GC-MACCS"
        representative_mol_smiles = list(exp_BigSolDB_df["Molecule_smiles"])[0]
        representative_mol = Chem.MolFromSmiles(representative_mol_smiles)
        ax = fig.add_subplot(grid[0:2, -1])  # Spanning both the experimental and prediction rows
        rep_img = Draw.MolToImage(representative_mol, size=(600, 600))
        ax.imshow(rep_img)
        ax.set_title(f"Kendall distance: {kendall} \nRMSE: {rmse}", fontsize=23, y=0.94)
        ax.axis('off')

        plt.savefig(f"Experimental comparison datasets/Images/{feature_type}_{exp_inchikey}.png")
        plt.close()
        ####################################################################################################
        ####################################################################################################


        #### THE ABRIDGED FIGURE @@@@
        ####################################################################################################
        ####################################################################################################
        if feature_type == "GC-MACCS":
            feature_type = "GC_MACCS"
        # I know it's a bit tedious:
        exp_BigSolDB_df = pd.read_csv(f"Experimental comparison datasets/Experimental/"
                                      f"Experimental_DF_{exp_inchikey}.csv")
        exp_BigSolDB_df.drop(columns="Unnamed: 0", inplace=True)
        pred_BigSolDB_df = pd.read_csv(f"Experimental comparison datasets/Pipeline/"
                                       f"Pipeline_{feature_type}_BigSolDB_LightGBM_5.0_{exp_inchikey}.csv")

        exp_BigSolDB_df = exp_BigSolDB_df[exp_BigSolDB_df["solute_InChIKey"] == exp_inchikey]
        mol_exp_solubility_values = list(exp_BigSolDB_df["experimental_logX"])
        smiles_exp_solvents = list(exp_BigSolDB_df["solvent_smiles"])
        inchikey_exp_solvents = list(exp_BigSolDB_df["solvent_InChIKey"])

        exp_BigSolDB_df.drop(columns="temperature", inplace=True)
        exp_BigSolDB_df.rename(columns={'solute_smiles': 'Molecule_smiles', 'solvent_smiles': 'Solvent_smiles',
                                        'solute_InChIKey': 'Molecule_InChIKey', 'solvent_InChIKey': 'Solvent_InChIKey'},
                               inplace=True)

        solvent_count_df = pd.DataFrame({"Solvent_smiles": smiles_exp_solvents})
        count_list = []
        for smiles_exp_solvent in smiles_exp_solvents:
            training_df = pd.read_csv(f"Experimental comparison datasets/Training/"
                                      f"Training_{feature_type}_BigSolDB_5.0_{exp_inchikey}.csv")
            count_num = len(training_df[training_df["solvent_smiles"] == smiles_exp_solvent])
            count_list.append(count_num)
        solvent_count_df["Solvent count"] = count_list
        solvent_count_df.to_csv(f"Experimental comparison datasets/Solvent_count/"
                                f"Solvent_count_train_dataset_{exp_inchikey}.csv")

        pred_BigSolDB_df = pred_BigSolDB_df[pred_BigSolDB_df["Molecule_InChIKey"] == exp_inchikey]
        pred_BigSolDB_df = pred_BigSolDB_df[pred_BigSolDB_df["Solvent_InChIKey"].isin(inchikey_exp_solvents)]
        pred_BigSolDB_df['Solvent_InChIKey'] = pd.Categorical(pred_BigSolDB_df['Solvent_InChIKey'],
                                                              categories=inchikey_exp_solvents,
                                                              ordered=True)
        pred_BigSolDB_df = pred_BigSolDB_df.sort_values('Solvent_InChIKey')
        pred_BigSolDB_df.drop(columns="Molecule_smiles", inplace=True)

        exp_pred_BigSolDB_df = pd.merge(exp_BigSolDB_df, pred_BigSolDB_df, how="inner",
                                        on=["Molecule_InChIKey", "Solvent_InChIKey", "Solvent_smiles"])

        ##### Now making the image:
        exp_BigSolDB_df = pd.read_csv(
            f"Experimental comparison datasets/Experimental/Experimental_DF_{exp_inchikey}.csv")
        exp_BigSolDB_df.drop(columns="Unnamed: 0", inplace=True)

        pred_BigSolDB_df = pd.read_csv(
            f"Experimental comparison datasets/Pipeline/Pipeline_{feature_type}_BigSolDB_LightGBM_5.0_{exp_inchikey}.csv")
        exp_BigSolDB_df = exp_BigSolDB_df[exp_BigSolDB_df["solute_InChIKey"] == exp_inchikey]
        mol_exp_solubility_values = list(exp_BigSolDB_df["experimental_logX"])
        smiles_exp_solvents = list(exp_BigSolDB_df["solvent_smiles"])
        inchikey_exp_solvents = list(exp_BigSolDB_df["solvent_InChIKey"])
        exp_BigSolDB_df.drop(columns="temperature", inplace=True)
        exp_BigSolDB_df.rename(columns={'solute_smiles': 'Molecule_smiles', 'solvent_smiles': 'Solvent_smiles',
                                        'solute_InChIKey': 'Molecule_InChIKey', 'solvent_InChIKey': 'Solvent_InChIKey'},
                               inplace=True)

        solvent_count_df = pd.DataFrame({"Solvent_smiles": smiles_exp_solvents})
        count_list = []
        for smiles_exp_solvent in smiles_exp_solvents:
            training_df = pd.read_csv(
                f"Experimental comparison datasets/Training/Training_{feature_type}_BigSolDB_5.0_{exp_inchikey}.csv")
            count_num = len(training_df[training_df["solvent_smiles"] == smiles_exp_solvent])
            count_list.append(count_num)
        solvent_count_df["Solvent count"] = count_list
        solvent_count_df.to_csv(
            f"Experimental comparison datasets/Solvent_count/Solvent_count_train_dataset_{exp_inchikey}.csv")

        pred_BigSolDB_df = pred_BigSolDB_df[pred_BigSolDB_df["Molecule_InChIKey"] == exp_inchikey]
        pred_BigSolDB_df = pred_BigSolDB_df[pred_BigSolDB_df["Solvent_InChIKey"].isin(inchikey_exp_solvents)]
        pred_BigSolDB_df['Solvent_InChIKey'] = pd.Categorical(pred_BigSolDB_df['Solvent_InChIKey'],
                                                              categories=inchikey_exp_solvents, ordered=True)
        pred_BigSolDB_df = pred_BigSolDB_df.sort_values('Solvent_InChIKey')
        pred_BigSolDB_df.drop(columns="Molecule_smiles", inplace=True)
        exp_pred_BigSolDB_df = pd.merge(exp_BigSolDB_df, pred_BigSolDB_df, how="inner",
                                        on=["Molecule_InChIKey", "Solvent_InChIKey", "Solvent_smiles"])

        # Now making the image:
        pred_BigSolDB_df.sort_values(by="Solubility Prediction", ascending=False, inplace=True)
        pred_solvent_smiles_list = list(pred_BigSolDB_df["Solvent_smiles"])[0:5]
        prediction_solubility_list = list(pred_BigSolDB_df["Solubility Prediction"])[0:5]

        exp_BigSolDB_df.sort_values(by="experimental_logX", ascending=False, inplace=True)
        exp_solvent_smiles_list = list(exp_BigSolDB_df["Solvent_smiles"])[0:5]
        experimental_solubility_list = list(exp_BigSolDB_df["experimental_logX"])[0:5]

        fig = plt.figure(figsize=(len(pred_solvent_smiles_list) * 2, 5))
        width_ratios = [1 for _ in range(len(pred_solvent_smiles_list))]
        width_ratios.extend([0.2, 2.3])
        grid = plt.GridSpec(2, len(pred_solvent_smiles_list) + 2, width_ratios=width_ratios, wspace=0.2, hspace=0.1,
                            figure=fig)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.06)

        fig.text(0.4, 0.91, "Experimental Values", fontsize=18, ha='center', color='#8B0000', fontweight='bold')
        fig.text(0.4, 0.47, "Predicted Values", fontsize=18, ha='center', color='#8B0000', fontweight='bold')

        for i, (solvent_smiles, solubility) in enumerate(zip(pred_solvent_smiles_list, experimental_solubility_list)):
            ax = fig.add_subplot(grid[0, i])
            mol = Chem.MolFromSmiles(solvent_smiles)
            img = Draw.MolToImage(mol, size=(400, 400))
            ax.imshow(img)
            ax.set_title(f"{solubility:.3f}", fontsize=16)
            ax.axis('off')

        for i, (solvent_smiles, prediction) in enumerate(zip(exp_solvent_smiles_list, prediction_solubility_list)):
            ax = fig.add_subplot(grid[1, i])
            mol = Chem.MolFromSmiles(solvent_smiles)
            img = Draw.MolToImage(mol, size=(400, 400))
            ax.imshow(img)
            ax.set_title(f"{prediction:.3f}", fontsize=16)
            ax.axis('off')

        if feature_type == "GC_MACCS":
            feature_type = "GC-MACCS"
        representative_mol_smiles = list(exp_BigSolDB_df["Molecule_smiles"])[0]
        representative_mol = Chem.MolFromSmiles(representative_mol_smiles)
        ax = fig.add_subplot(grid[0:2, -1])  # Spanning both the experimental and prediction rows
        rep_img = Draw.MolToImage(representative_mol, size=(600, 600))
        ax.imshow(rep_img)
        ax.set_title(f"Kendall distance: {kendall} \nRMSE: {rmse}", fontsize=19, y=0.94)

        ax.axis('off')
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.05, pos.width, pos.height * 0.95])
        plt.savefig(f"Experimental comparison datasets/"
                    f"Images/Abridged_{feature_type}_{exp_inchikey}.png", bbox_inches='tight')
        plt.close()
        ####################################################################################################
        ####################################################################################################

        new_row = pd.DataFrame({
            'InChIKey': [exp_inchikey],
            'Feature_Type': [feature_type],
            'Kendall_Distance': [kendall],
            'RMSE': [rmse]
        })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

metrics_df.to_csv("Organic solubility trends comparison.csv", index=False)


