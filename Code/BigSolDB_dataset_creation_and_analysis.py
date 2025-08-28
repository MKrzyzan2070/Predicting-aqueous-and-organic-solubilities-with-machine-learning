from Code import BigSolDB_dataset_maker, Mini_Code_Creating_JSON_file
from Code import BigSolDB_analysis

def dataset_creation_and_analysis(exp_inchikey, temperature,
    tolerance, make_pickle, make_BigSolDB_dataset, model, do_PCA, dataset, test_set_InChIKey_list, do_analysis):

    if exp_inchikey is not None:
        test_set_InChIKey_list.append(exp_inchikey)

    # Creating the list of the molar fractions. This is necessary for the UNIFAC features
    mole_fraction_list = [(0.1, 0.9), (0.2, 0.8), (0.8, 0.2), (0.9, 0.1)]

    ############################################ CREATING THE DATASETS ##################################################
    #The dataset will be created only if making the pickle for kfold validation is requested
    if make_BigSolDB_dataset is True:
        BigSolDB_dataset_maker.make_BigSolDB_datasets(test_set_InChIKey_list, mole_fraction_list,
                                                        temperature, tolerance)
        Mini_Code_Creating_JSON_file.Make_solvent_JSON()
    ######################################################################################################################
    ######################################################################################################################

    if do_analysis is True:
        ############################################ ANALYSIS/VISUALISATION ##################################################
        path_list = [
            "Datasets/BigSolDB_Datasets_Processed/GC_BigSolDB.csv",
            "Datasets/BigSolDB_Datasets_Processed/MACCS_BigSolDB.csv",
            "Datasets/BigSolDB_Datasets_Processed/GC_MACCS_BigSolDB.csv",
            "Datasets/BigSolDB_Datasets_Processed/UNIFAC_BigSolDB.csv"
                        ]

        # Using the same code for the analysis as the one used for CombiSolu:
        for path in path_list:
            BigSolDB_analysis.KFold_validation_test_validation(path, make_pickle, tolerance, model, do_PCA, dataset)
            BigSolDB_analysis.Plot(path, tolerance, model, make_pickle, dataset)
        ######################################################################################################################
        ######################################################################################################################
