from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor

from models.models import RidgeCVModule, RidgeModule, XGBoostRegressorWrapper
from models.models import GradientRandFeatBoostRegression, End2EndMLPResNet
from optuna_kfoldCV import evaluate_pytorch_model_kfoldcv


##############################################################  |
##### Create "evalute_MODELHERE" function for each model #####  |
##############################################################  V


def evaluate_GRFBoost(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    ModelClass = GradientRandFeatBoostRegression
    get_optuna_params = lambda trial : {
        "out_dim": trial.suggest_categorical("out_dim", [y.size(1)]),  # Fixed value
        "feature_type": trial.suggest_categorical("feature_type", ["SWIM"]),    # Fixed value
        "upscale": trial.suggest_categorical("upscale", ["dense"]),             # Fixed value

        "hidden_dim": trial.suggest_int("hidden_dim", 32, 128, log=True),
        "bottleneck_dim": trial.suggest_int("bottleneck_dim", 64, 128, log=True),
        "n_layers": trial.suggest_int("n_layers", 1, 50, log=True),
        "l2_reg": trial.suggest_float("l2_reg", 1e-6, 0.1, log=True),
        "boost_lr": trial.suggest_float("boost_lr", 0.1, 1.0, log=True),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device,
    )



def evaluate_RidgeCV(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    ModelClass = RidgeCVModule
    get_optuna_params = lambda trial : {
        "lower_alpha": trial.suggest_float("lower_alpha", 1e-7, 0.1, log=True),
        "upper_alpha": trial.suggest_float("upper_alpha", 1e-7, 0.1, log=True),
        "n_alphas": trial.suggest_int("n_alphas", 10, 50),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device,
    )



def evaluate_Ridge(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    ModelClass = RidgeModule
    get_optuna_params = lambda trial : {
        "l2_reg": trial.suggest_float("l2_reg", 1e-7, 0.1, log=True),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device,
    )


def evaluate_XGBoostRegressor(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    ModelClass = XGBoostRegressorWrapper
    get_optuna_params = lambda trial : {
        "objective": trial.suggest_categorical("objective", ["reg:squarederror"]),   # Fixed value

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



# def evaluate_End2End(
#         X: Tensor,
#         y: Tensor,
#         k_folds: int,
#         cv_seed: int,
#         regression_or_classification: str,
#         n_optuna_trials: int,
#         device: Literal["cpu", "cuda"],
#         ):
#     ModelClass = End2EndMLPResNet
#     get_optuna_params = lambda trial : {
#         "hidden_dim": trial.suggest_int("hidden_dim", X.size(1), 128, log=True),
#         "n_layers": trial.suggest_int("n_layers", 1, 50, log=True),
#         "l2_reg": trial.suggest_float("l2_reg", 1e-6, 0.1, log=True),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#     }

#     return evaluate_pytorch_model_kfoldcv(
#         ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
#         regression_or_classification, n_optuna_trials, device,
#         )


################################################################ |
########### Run a specified model on OpenML datasets ########### |
################################################################ V

if __name__ == "__main__":
    pass