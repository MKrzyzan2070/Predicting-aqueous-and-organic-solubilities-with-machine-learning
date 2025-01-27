from sklearn.ensemble import RandomForestRegressor
import json
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import numpy as np
np.random.seed(1)


#########################
import logging
import warnings
logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
#########################
def Train_RF_model(kf, x_data_train_split, y_data_train_split_binned, name,
                   y_data_train_split, dataset_name, tolerance=None):

    ################################################################################################
    # The  GridSearch. A GridSearch with more params was used before, but so as not to waste time when
    # training the model, the number of params was highly minimized based on the previous findings.:
    param_grid = {
        'random_state': [1],
        'oob_score': [True],
        'max_features': ['sqrt', None],
        'n_estimators': [300],
        'max_depth': [None],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [2, 4],
    }
    ################################################################################################

    # Performing the gridsearch
    grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid,
                               cv=kf.split(x_data_train_split,  y_data_train_split_binned), scoring='r2')
    grid_search.fit(x_data_train_split, y_data_train_split)

    # Saving the best parameters
    best_params = grid_search.best_params_
    params_dict = dict(best_params)
    if tolerance is not None:
        with open(f"Hyperparameters/{dataset_name}/{name}_RF_{tolerance}.json", 'w') as f:
            json.dump(params_dict, f, indent=4)
    else:
        with open(f"Hyperparameters/{dataset_name}/{name}_RF.json", 'w') as f:
            json.dump(params_dict, f, indent=4)

    # Testing the model on the x_data_test_split:
    random_forest_model = RandomForestRegressor(**best_params)
    random_forest_model_fit = random_forest_model.fit(x_data_train_split, y_data_train_split)

    return random_forest_model_fit, grid_search.best_score_


def Train_LGBM_model(kf, x_data_train_split, y_data_train_split_binned, name,
                     y_data_train_split, dataset_name, tolerance=None):


    ################################################################################################
    # The  GridSearch:
    if dataset_name == "BigSolDB":
        param_grid = {
            'random_state': [1],
            'verbosity': [-1],
            'colsample_bytree': [0.8],
            'subsample': [0.8],
            'learning_rate': [0.05, 0.08],
            'max_depth': [10, 15],
            'min_child_samples': [20, 30],
            'n_estimators': [600],
            'num_leaves': [20, 30],
            'reg_alpha': [0.5]
        }
    else:
        param_grid = {
            'random_state': [1],
            'verbosity': [-1],
            'colsample_bytree': [0.8],
            'subsample': [0.8],
            'learning_rate': [0.05, 0.08],
            'max_depth': [10, 15],
            'min_child_samples': [20, 30],
            'n_estimators': [600],
            'num_leaves': [20, 30],
            'reg_alpha': [0.5]
        }
    ################################################################################################

    grid_search = GridSearchCV(estimator=lgb.LGBMRegressor(),
                               param_grid=param_grid, cv=kf.split(x_data_train_split,  y_data_train_split_binned), scoring='r2')
    grid_search.fit(x_data_train_split, y_data_train_split)

    # Saving the best parameters
    best_params = grid_search.best_params_
    params_dict = dict(best_params)
    if tolerance is not None:
        with open(f"Hyperparameters/{dataset_name}/{name}_LightGBM_{tolerance}.json", 'w') as f:
            json.dump(params_dict, f, indent=4)
    else:
        with open(f"Hyperparameters/{dataset_name}/{name}_LightGBM.json", 'w') as f:
            json.dump(params_dict, f, indent=4)

    lgbm_model = lgb.LGBMRegressor(**best_params)

    lgbm_model_fit = lgbm_model.fit(x_data_train_split, y_data_train_split)

    return lgbm_model_fit, grid_search.best_score_
