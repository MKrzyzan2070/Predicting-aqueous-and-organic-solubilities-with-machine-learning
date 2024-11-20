from Code import BigSolDB_analysis
import cirpy
import pandas as pd


def make_bare_dataset_for_AqSolDB(solvent_list):

    test_set_InChIKey_list = []
    best_solvent_list = []
    ID_list = []
    Name_list = []
    InChI_list = []
    Solubility_list = []
    Ocurrences_list = []
    Group_list = []
    SD_list = []

    best_solv_dict = {}


    # Number one
    # Defining which molecules should be excluded from the dataset:
    for solvent_smile in solvent_list:
        inchi_key = cirpy.resolve(solvent_smile, 'inchikey')
        inchi_key = inchi_key.split("=")[1]
        best_solvent_list.append(solvent_smile)
        test_set_InChIKey_list.append(inchi_key)

        ID_list.append(0)
        Name_list.append(0)
        InChI_list.append(0)
        Ocurrences_list.append(0)
        Group_list.append(0)
        SD_list.append(0)
        Solubility_list.append(0)

    df = pd.DataFrame({"ID": ID_list, "Name": Name_list, "InChI": InChI_list, "InChIKey": test_set_InChIKey_list,
                       "SMILES": best_solvent_list, "Solubility": Solubility_list, "SD": SD_list,
                       "Ocurrences": Ocurrences_list, "Group": Group_list
                       })
    df.to_csv("Datasets/AqSolDB_Datasets_Processed/Dataset_for_Predictions/Bare Prediction Dataset.csv")
    ####################################################################################
