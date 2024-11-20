from Code import AqSolDB_dataset_maker
from Code import AqSolDB_analysis


def dataset_creation_and_analysis(make_AqSolu_dataset, make_pickle, model, do_PCA, test_set_InChIKey_list, do_analysis):
    mole_fraction_list = [(0.1, 0.9), (0.2, 0.8), (0.8, 0.2), (0.9, 0.1)]

    ##################################################################################################################
    ##################################################################################################################

    ############################################ CREATING THE DATASETS ###############################################
    #The dataset will be created only if making the pickle for kfold validation is requested
    if make_AqSolu_dataset is True:
        AqSolDB_dataset_maker.make_AqSolDB_datasets(test_set_InChIKey_list, mole_fraction_list)
    ###################################################################################################################

    if do_analysis is True:
        ############################################ ANALYSIS/VISUALISATION ########################################
        path_list = [
            "Datasets/AqSolDB_Datasets_Processed/GC_AqSolDB.csv",
            "Datasets/AqSolDB_Datasets_Processed/MACCS_AqSolDB.csv",
            "Datasets/AqSolDB_Datasets_Processed/GC_MACCS_AqSolDB.csv"
                        ]

        for path in path_list:
            AqSolDB_analysis.KFold_validation_test_validation(path, make_pickle, model, do_PCA)
            AqSolDB_analysis.Plot(path, model, make_pickle)
        #############################################################################################################