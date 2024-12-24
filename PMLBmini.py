from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import argparse
from pathlib import Path
import os
import pickle
import time
import itertools

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import tabmini

from models.base import LogisticRegression
from models.gridsearch_wrapper import SKLearnWrapper
from models.xgboost_wrapper import XGBoostRegressorWrapper
from models.end2end import End2EndMLPResNet
from models.random_feature_representation_boosting import GradientRFRBoostClassifier





##################################################
############# Grid Search wrapper    #############
############# for custom estimators  #############
##################################################


class WrapperGridSearch(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 param_grid:Dict[str, List], 
                 verbose=3, 
                 scaler = MinMaxScaler(), # alternative is StandardScaler()
                 seed = 42,
                 scoring:Literal["roc_auc", "accuracy", "neg_log_loss"]="roc_auc"
                 ):
        self.param_grid = param_grid
        self.verbose = verbose
        self.scaler = scaler
        self.seed = seed
        self.scoring = scoring


    def fit(self, X, y):
        """
        Performs a stratified 3-fold CV for hyperparameter tuning
        based on self.param_grid, and fits the best model on the whole dataset
        """
        # MinMaxScaler and convert to torch
        self.classes_ = np.unique(y)
        N, D = X.values.shape
        X = self.scaler.fit_transform(X.values)
        X = torch.tensor(X).float()
        y = torch.tensor(y.values)[..., None].float()

        # Perform grid search with k-fold cross-validation
        param_grid = {**self.param_grid, **{"seed": [self.seed]}, **{"in_dim": [D]}}
        if self.param_grid["modelClass"][0] == End2EndMLPResNet:
            param_grid["batch_size"] = [max(int(N*8/15-1), self.param_grid["batch_size"][0])] # otherwise we can get a batch size of 1, error with batch norm
            param_grid["out_dim"] = [1]
        else:
            param_grid["n_classes"] = [2]

        estimator = SKLearnWrapper()
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5),
            verbose=self.verbose,
            scoring=self.scoring,   # scoring=accuracy   # scoring="neg_log_loss"  # scoring="roc_auc"
            error_score = -1e6 if scoring=="neg_log_loss" else 0,
        )

        # fit model
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        print("Best params:", grid_search.best_params_)
        best_model.set_model_eval()
        self.model = best_model
        return self


    def predict_proba(self, X):
        X = self.scaler.transform(X.values)
        X = torch.tensor(X).float()
        return self.model.predict_proba(X)


    def decision_function(self, X):
        X = self.scaler.transform(X.values)
        X = torch.tensor(X).float()
        return self.model.decision_function(X)
    

##################################################
###### run model on PMLBmini datasets ############
##################################################


def test_on_PMLBmini(
    estimator: BaseEstimator,
    estimator_name: str, 
    dataset_indices: List[int],
    save_dir: str, # = Config.save_dir / 'PMLBmini_dataset.pkl',
    other_saved_methods = {}, #{'XGBoost'},
    ):

    #download dataset, cache it
    dataset_save_path = save_dir + 'PMLBmini_dataset.pkl'
    if not os.path.exists(dataset_save_path):
        print("Dataset not found, downloading")
        dataset = tabmini.load_dataset(reduced=False)
        os.makedirs(save_dir, exist_ok=True)
        with open(dataset_save_path, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        print("Dataset found, loading")
        with open(dataset_save_path, 'rb') as f:
            dataset = pickle.load(f)

    # Select the dataset
    sorted_items = sorted(dataset.items())
    dataset = {k: v for i, (k, v) in enumerate(sorted_items) if i in dataset_indices}

    # Perform the comparison
    test_results, train_results = tabmini.compare(
        estimator_name,
        estimator,
        dataset,
        working_directory = Path(save_dir),
        scoring_method="roc_auc",
        methods= other_saved_methods,
        time_limit=3600,
        n_jobs=1,
    )

    # save results as CSV 
    dataset_indices_str = "_".join(map(str, dataset_indices))
    os.makedirs(save_dir + f"{estimator_name}/", exist_ok=True)
    test_results.to_csv(save_dir + f'{estimator_name}/test_{dataset_indices_str}.csv')
    train_results.to_csv(save_dir + f'{estimator_name}/train_{dataset_indices_str}.csv')
    return train_results, test_results


##################################################
######## param grid getters for models ###########
##################################################


def End2End_param_grid():
    param_grid = {
        'modelClass': [End2EndMLPResNet],
        'lr': np.logspace(-5, -1, 5),
        'hidden_dim': [512],
        'bottleneck_dim': [512],
        'n_blocks': [1, 2, 3],
        'loss': ["bce"],
        'n_epochs': [20, 30, 40, 50],
        'end_lr_factor': [0.01],
        'weight_decay': [0.0001],
        'batch_size': [32],
        'activation': [nn.ReLU()],
        }
    return param_grid



def RFNN_param_grid(
        upscale_type: Literal["identity", "SWIM", "iid"],
        activation: Literal["tanh", "relu"] = "tanh",
        hidden_dim: int = 512,
        ):
    param_grid = {
        'modelClass': [GradientRFRBoostClassifier],
        'n_layers': [0],
        'upscale_type': [upscale_type],
        'l2_cls': [10, 1, 0.1, 0.01, 0.001, 0.0001],
        'hidden_dim': [hidden_dim],
        'activation': [activation],
        }
    return param_grid



def GRFRBoost_param_grid(
        feature_type: Literal["SWIM", "iid"],
        upscale_type: Literal["identity", "SWIM", "iid"],
        activation: Literal["tanh", "relu"] = "tanh",
        use_batchnorm: bool = False,
        do_linesearch: bool = False, # find out if good or not
        freeze_top: bool = False,
        hidden_dim: int = 512,
        ghat_solver: Literal["solve", "ridgecv"] = "solve",
        ):
    param_grid = {
        'modelClass': [GradientRFRBoostClassifier],
        'l2_cls': [10, 1, 0.1, 0.01, 0.001, 0.0001],    # 1 to -4
        'l2_ghat': ([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] if ghat_solver == "solve"   #0 to -5
                    else [None]),
        'boost_lr': [10, 1.0, 0.1, 0.01],
        'n_layers': [1],
        'randfeat_xt_dim': [hidden_dim],
        'randfeat_x0_dim': [hidden_dim],
        'hidden_dim': [hidden_dim],
        'upscale_type': [upscale_type],
        'feature_type': [feature_type],
        'use_batchnorm': [use_batchnorm],
        'activation': [activation],
        'do_linesearch': [do_linesearch],
        'freeze_top_at_t': [0 if freeze_top else None],
        }
    return param_grid


######################################################  |
#####  command line argument to run experiments  #####  |
######################################################  V


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with different models and datasets.")
    parser.add_argument(
        "--models", 
        nargs='+', 
        type=str, 
        default=["LogisticRegression"], 
        help="List of model names to run."
    )
    parser.add_argument(
        "--dataset_indices", 
        nargs='+', 
        type=int, 
        default=[i for i in range(44)], 
        help="List of dataset IDs to run."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/nikita/Code/random-feature-boosting/save/PMLBmini/",
        help="Directory where the results files will be saved."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for all randomness."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Run experiments
    for model_name in args.models:
        #relu or tanh
        activation = "relu" if "relu" in model_name else "tanh"
        #gridsearchcv scorer
        if "crossentropy" in model_name:
            scoring = "neg_log_loss"
        elif "acc" in model_name:
            scoring = "accuracy"
        else:
            scoring = "roc_auc"
        
        #end2end
        if "E2E_MLP_ResNet" in model_name:
            param_grid = End2End_param_grid()

        #RFNN
        elif "RFNN_iid" in model_name:
            param_grid = RFNN_param_grid("iid", activation)
        elif "RFNN" in model_name:
            param_grid = RFNN_param_grid("SWIM", activation)

        #logistic
        elif "Logistic(ours)" in model_name:
            param_grid = RFNN_param_grid("identity")

        #GRFRBoost
        if "upSWIM" in model_name:
            up = "SWIM"
        elif "upidentity" in model_name:
            up = "identity"
        elif "upiid" in model_name:
            up = "iid"
        feat = "SWIM" if "featSWIM" in model_name else "iid"
        linesearch = True if "linesearchTrue" in model_name else False
        freeze = True if "freezeTrue" in model_name else False
        if "GRFRBoost" in model_name:
            param_grid = GRFRBoost_param_grid(upscale_type=up, 
                                                feature_type=feat, 
                                                do_linesearch=linesearch,
                                                freeze_top=freeze,
                                                activation=activation)

        #run model
        estimator = WrapperGridSearch(param_grid, seed=args.seed, scoring=scoring)
        train_results, test_results = test_on_PMLBmini(estimator, model_name, args.dataset_indices, args.save_dir)