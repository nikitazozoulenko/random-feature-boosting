from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor

from models import LogisticRegression, GradientRandomFeatureBoostingClassification, XGBoostClassifierWrapper
from optuna_kfoldCV import evaluate_pytorch_model_kfoldcv


##############################################################  |
##### Create "evalute_MODELHERE" function for each model #####  |
##############################################################  V


def evaluate_LogisticRegression(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    ModelClass = LogisticRegression
    get_optuna_params = lambda trial : {
        "in_dim": trial.suggest_categorical("in_dim", [X.shape[1]]),    # Fixed value
        "out_dim": trial.suggest_categorical("out_dim", [y.size(1)]),   # Fixed value

        "l2_reg": trial.suggest_float("l2_reg", 1e-8, 0.1, log=True),
        "max_iter": trial.suggest_int("max_iter", 20, 200),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device,
    )


def evaluate_XGBoostClassifier(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    ModelClass = XGBoostClassifierWrapper
    get_optuna_params = lambda trial : {
        "objective": trial.suggest_categorical("objective", ["multi:softmax"]), # Fixed param
        "num_class": trial.suggest_categorical("num_class", [y.size(1)]),       # Fixed param

        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device
    )