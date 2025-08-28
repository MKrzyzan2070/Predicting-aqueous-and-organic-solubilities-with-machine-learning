from Code import AqSolDB_dataset_creation_and_analysis
from Code import AqSolDB_molecule_solubility_prediction
import json


#   I N F O R M A T I O N !!!!
# It is very important that whenever a new molecule is added to the list for making predictions, i.e., for identifying
# the organic co-solvents a new organic solubility and aquatic solubility dataset must be made. The only thing that can
# be set to False is the first make_pickle variable which corresponds to the train/test/validation analysis and is not
# relevant to identifying the organic co-solvents

#   I N F O R M A T I O N !!!!
# THE REPORTED PERFORMANCE METRIC VALUES REPORTED IN THE PAPER WERE OBTAINED FOR THE EMPTY LIST OF MOLECULES OF INTEREST!
# This was done by setting exclude_mols to False. In that case, the make_AqSolDB is also being set to true so that
# the dataset is being reset.

# So, for the train/test/validation, the settings are:
# make_pickle_train_test_validation = True
# exclude_mols = False
# make_AqSolDB_dataset = True
# make_pickle_prediction = False

# The following settings if it is just reading the pickle files:
# make_pickle_train_test_validation = False
# exclude_mols = False
# make_AqSolDB_dataset = False
# make_pickle_prediction = False

# For the organic co-solvent identification (pipeline), the settings are:
# make_pickle_train_test_validation = False
# exclude_mols = True
# make_AqSolDB_dataset = True
# make_pickle_prediction = True

########################################################################################################################
### @@@@@@@@
make_pickle_train_test_validation = True
exclude_mols = True
make_AqSolDB_dataset = True
make_pickle_prediction = False
### @@@@@@@@

# This model will be used for the train/test not the pipeline predictions
model = "LightGBM" # Either RF or LightGBM
# This model will be for the pipeline
model_pipeline = "LightGBM"
do_PCA = False
# The train/test analysis encompasses all the feature type cases: MACCS, GC, and GC-MACCS.
# However, for the pipeline, a specific feature-type can be selected.
# The list of the desired feature types is the input, and it will generate the predictions for them:
feature_type_list_pipeline = ["GC_MACCS"]

if exclude_mols is True:
    with open("Code/JSON_files/Solvent_InChIKey_list_BigSolDB.json", 'r') as file:
        test_set_InChIKey_list = json.load(file)

    # It is not necessary to exclude benzaldehyde and limonene from the AqSolDB, since the potential solvents are being
    # removed from the dataset, as it is the solubility in water of those solvents that is being predicted.
    molecules_of_interest = []
    for mol in molecules_of_interest:
        test_set_InChIKey_list.append(mol)
else:
    test_set_InChIKey_list = []

# This prevents from unnecessarily doing the analysis:
if exclude_mols is True:
    do_analysis = False
else:
    do_analysis = True
AqSolDB_dataset_creation_and_analysis.dataset_creation_and_analysis(make_AqSolDB_dataset,
                                                                    make_pickle_train_test_validation, model,
                                                                    do_PCA, test_set_InChIKey_list, do_analysis)
########################################################################################################################

# This basically triggers the pipeline predictions for the molecules of interest:
if exclude_mols is True:
    ########################################################################################################################
    make_pickle = True
    model = model_pipeline
    feature_type_list = feature_type_list_pipeline
    dataset_name = "AqSolDB"

    AqSolDB_molecule_solubility_prediction.molecule_solubility_prediction(make_pickle_prediction, model, feature_type_list,
                                                                          dataset_name)
    ########################################################################################################################