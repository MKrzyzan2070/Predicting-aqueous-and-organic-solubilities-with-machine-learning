from Code import BigSolDB_dataset_creation_and_analysis, BigSolDB_molecule_solubility_prediction

#   I N F O R M A T I O N !!!!
# It is very important that the whenever a new molecule is added to the list for making predcitions i.e. for identifying
# the organic co-solvents a new organic solubility and aquatic solubility dataset must be made. The only thing that can
# be set to False is the first make_pickle variable which corresponds to the train/test/validation analysis which is not
# relevant to identifying the organic co-solvents

#    I N F O R M A T I O N !!!!
# THE REPORTED PERFORMANCE METRIC VALUES REPORTED IN THE PAPER WERE OBTAINED FOR THE EMPTY LIST OF MOLECULES OF INTEREST!
# this was done by setting exclude_mols to False. In that case, the make_AqSolDB is also being set to true so that
# # the dataset is being reset.

# So, for the train/test/validation, the settings are:
# make_pickle_train_test_validation = True
# exclude_mols = False
# make_BigSolDB_dataset = True
# make_pickle_prediction = False

# The following settings if it is just reading the pickle files:
# make_pickle_train_test_validation = False
# exclude_mols = False
# make_BigSolDB_dataset = False
# make_pickle_prediction = False

# For the organic co-solvent identification (pipeline), the settings are:
# make_pickle_train_test_validation = False
# exclude_mols = True
# make_BigSolDB_dataset = True
# make_pickle_prediction = True

########################################################################################################################
temperature = 298.15
tolerance = 5.0

### @@@@@@@@
make_pickle_train_test_validation = False
exclude_mols = True
make_BigSolDB_dataset = True
make_pickle_prediction = True
### @@@@@@@@

model = "LightGBM" # Either RF or LightGBM
do_PCA = False
dataset = "BigSolDB" # This is a relict from when CombiSolu dataset was tried. Still, it will be left in case
                     # analysis will done on CombiSolu dataset at some time

exp_inchikey = None
if exclude_mols is True:
    # The list of the molecules of interest which are to be excluded from the dataset to avoid data leakage:
    molecules_of_interest = ['HUMNYLRZRPPJDN-UHFFFAOYSA-N', # Benzaldehyde
                             'XMGQYMWWDOXHJM-UHFFFAOYSA-N'  # Limonene
                             ]
    test_set_InChIKey_list = []
    for mol in molecules_of_interest:
        test_set_InChIKey_list.append(mol)
else:
    test_set_InChIKey_list = []

# This prevents from unnecessarily doing the analysis:
if exclude_mols is True:
    do_analysis = False
else:
    do_analysis = True
BigSolDB_dataset_creation_and_analysis.dataset_creation_and_analysis(exp_inchikey, temperature,
                                                                     tolerance, make_pickle_train_test_validation,
                                                                     make_BigSolDB_dataset, model,
                                                                     do_PCA, dataset, test_set_InChIKey_list, do_analysis)
########################################################################################################################


########################################################################################################################
# This basically triggers the pipeline predictions for the molecules of interest:
if exclude_mols is True:
    tolerance = 5.0
    model = "LightGBM"
    feature_type_list = ["GC_MACCS"]
    dataset_name_1 = "BigSolDB"
    dataset_name_2 = "BigSolDB" # Those two names are also a relict when the code was primarly written for a different
                                # dataset

    BigSolDB_molecule_solubility_prediction.molecule_solubility_prediction(make_pickle_prediction, tolerance, model,
                                                                           feature_type_list, dataset_name_1)

    for feature_type in feature_type_list:
        BigSolDB_molecule_solubility_prediction.analyse(dataset_name_1, dataset_name_2, feature_type, model, tolerance)
########################################################################################################################
