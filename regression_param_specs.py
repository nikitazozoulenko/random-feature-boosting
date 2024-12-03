from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor

from models.models import RidgeCVModule, RidgeModule, XGBoostRegressorWrapper
from models.models import End2EndMLPResNet
from models.models import GradientRFBoostRegressor, GreedyRFBoostRegressor, GreedyRFBoostRegressor_ScalarDiagDelta
from optuna_kfoldCV import evaluate_pytorch_model_kfoldcv


#############################################################/8  |
##### Create "evalute_MODELHERE" function for each model #####  |
##############################################################  V


def get_GradientRFBoost_eval_fun(
        feature_type: Literal["dense", "SWIM"] = "SWIM",
        upscale : Literal["dense", "SWIM", "identity"] = "dense",
        ):
    """Returns a function that evaluates the GradientRFBoost model
    with the specified number of layers"""
    def evaluate_GRFBoost(
            X: Tensor,
            y: Tensor,
            k_folds: int,
            cv_seed: int,
            regression_or_classification: str,
            n_optuna_trials: int,
            device: Literal["cpu", "cuda"],
            early_stopping_patience: int,
            ):
        ModelClass = GradientRFBoostRegressor
        get_optuna_params = lambda trial : {   
            "feature_type": trial.suggest_categorical("feature_type", [feature_type]),  # Fixed value
            "upscale": trial.suggest_categorical("upscale", [upscale]),                 # Fixed value

            "n_layers": trial.suggest_int("n_layers", 1, 40, log=True),
            "hidden_dim": (
                trial.suggest_int("hidden_dim", 16, 144, step=32)
                if upscale != "identity"
                else trial.suggest_categorical("hidden_dim", [X.size(1)])
            ),
            "randfeat_xt_dim": trial.suggest_int("randfeat_xt_dim", 128, 512, step=128),
            "randfeat_x0_dim": trial.suggest_int("randfeat_x0_dim", 128, 512, step=128),
            "boost_lr": trial.suggest_float("boost_lr", 0.3, 1.001, step=0.1),
        }

        return evaluate_pytorch_model_kfoldcv(
            ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device, early_stopping_patience
        )
    return evaluate_GRFBoost



def get_GreedyRFBoost_eval_fun(
        feature_type: Literal["dense", "SWIM"] = "SWIM",
        upscale: Literal["dense", "SWIM", "identity"] = "dense",
        ):
    """Returns a function that evaluates the GreedyRFBoost model
    with the specified number of layers"""
    def evaluate_GreedyRFBoost(
            X: Tensor,
            y: Tensor,
            k_folds: int,
            cv_seed: int,
            regression_or_classification: str,
            n_optuna_trials: int,
            device: Literal["cpu", "cuda"],
            early_stopping_patience: int,
            ):
        
        ModelClass = GreedyRFBoostRegressor
        get_optuna_params = lambda trial : {
            "feature_type": trial.suggest_categorical("feature_type", [feature_type]), # Fixed value
            "upscale": trial.suggest_categorical("upscale", [upscale]), # Fixed value                # Fixed value

            "n_layers": trial.suggest_int("n_layers", 1, 40, log=True),
            "hidden_dim": (
                trial.suggest_int("hidden_dim", 16, 144, step=32)
                if upscale != "identity"
                else trial.suggest_categorical("hidden_dim", [X.size(1)])
            ),
            "randfeat_xt_dim": trial.suggest_int("randfeat_xt_dim", 128, 512, step=128),
            "randfeat_x0_dim": trial.suggest_int("randfeat_x0_dim", 128, 512, step=128),
            "boost_lr": trial.suggest_float("boost_lr", 0.3, 1.001, step=0.1),
            "l2_reg_sandwich": trial.suggest_float("l2_reg_sandwich", 1e-6, 1, log=True),
        }

        return evaluate_pytorch_model_kfoldcv(
            ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device, early_stopping_patience
        )
    return evaluate_GreedyRFBoost



def get_GreedyRFBoostDiagScalar_eval_fun(
        feature_type: Literal["dense", "SWIM"] = "SWIM",
        upscale: Literal["dense", "SWIM", "identity"] = "dense",
        sandwich_solver: Literal["dense", "diag", "scalar"] = "diag",
        ):
    """Returns a function that evaluates the GreedyRFBoost model
    with the specified number of layers"""
    def evaluate_GreedyRFBoost(
            X: Tensor,
            y: Tensor,
            k_folds: int,
            cv_seed: int,
            regression_or_classification: str,
            n_optuna_trials: int,
            device: Literal["cpu", "cuda"],
            early_stopping_patience: int,
            ):
        
        ModelClass = GreedyRFBoostRegressor_ScalarDiagDelta
        get_optuna_params = lambda trial : {
            "feature_type": trial.suggest_categorical("feature_type", [feature_type]), # Fixed value
            "upscale": trial.suggest_categorical("upscale", [upscale]), # Fixed value
            "sandwich_solver": trial.suggest_categorical("sandwich_solver", [sandwich_solver]), # Fixed value

            "n_layers": trial.suggest_int("n_layers", 1, 40, log=True),
            "hidden_dim": trial.suggest_int("hidden_dim", 16, 512, log=True),
            "boost_lr": trial.suggest_float("boost_lr", 0.5, 1.001, step=0.1),
            "l2_reg_sandwich": trial.suggest_float("l2_reg_sandwich", 1e-8, 1e-2, log=True),
        }
        return evaluate_pytorch_model_kfoldcv(
            ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device, early_stopping_patience
        )
    return evaluate_GreedyRFBoost



def evaluate_RidgeCV(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        early_stopping_patience: int,
        ):
    ModelClass = RidgeCVModule
    get_optuna_params = lambda trial : {
        "lower_alpha": trial.suggest_float("lower_alpha", 1e-8, 1e-5, log=True),
        "upper_alpha": trial.suggest_float("upper_alpha", 0.001, 0.1, log=True),
        "n_alphas": trial.suggest_int("n_alphas", 10, 20),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device, early_stopping_patience
    )



def evaluate_Ridge(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        early_stopping_patience: int,
        ):
    ModelClass = RidgeModule
    get_optuna_params = lambda trial : {
        "l2_reg": trial.suggest_float("l2_reg", 1e-7, 0.1, log=True),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device, early_stopping_patience
    )



def evaluate_XGBoostRegressor(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        early_stopping_patience: int,
        ):
    ModelClass = XGBoostRegressorWrapper
    get_optuna_params = lambda trial : {
        "objective": trial.suggest_categorical("objective", ["reg:squarederror"]),   # Fixed value

        "alpha": trial.suggest_float("alpha", 1e-3, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-3, 100.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        #"subsample": trial.suggest_float("subsample", 0.5, 1.0),
        #"colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device, early_stopping_patience
    )



def evaluate_End2End(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        early_stopping_patience: int,
        ):
    
    ModelClass = End2EndMLPResNet
    get_optuna_params = lambda trial : {
        "in_dim": trial.suggest_categorical("in_dim", [X.size(1)]),         # Fixed value
        "out_dim": trial.suggest_categorical("out_dim", [y.size(1)]),       # Fixed value
        "loss": trial.suggest_categorical("loss", ["mse"]),# Fixed value
        
        "n_blocks": trial.suggest_int("n_blocks", 1, 5),
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 128, step=32),
        "bottleneck_dim": trial.suggest_int("bottleneck_dim", 32, 128, step=32),
        "lr": trial.suggest_float("lr", 0.0001, 0.1, log=True),
        "end_lr_factor": trial.suggest_float("end_lr_factor", 0.01, 1.0, log=True),
        "n_epochs": trial.suggest_int("n_epochs", 5, 20, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 0.01, log=True),
        "batch_size": trial.suggest_int("batch_size", 32, 128, step=32),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device, early_stopping_patience
        )


################################################################ |
########### Run a specified model on OpenML datasets ########### |
################################################################ V


######################################################  |
#####  command line argument to run experiments  #####  |
######################################################  V

# start by using all datasets. I can change this later
def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with different models and datasets.")
    parser.add_argument(
        "--models", 
        nargs='+', 
        type=str, 
        default=["GradientRFBoost", "GreedyRFBoost"], 
        help="List of model names to run."
    )
    parser.add_argument(
        "--dataset_indices", 
        nargs='+', 
        type=int, 
        default=[i for i in range(len(openML_reg_ids_noCat))], 
        help="List of datasets to run."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/nikita/Code/random-feature-boosting/save/OpenMLRegression/",
        help="Directory where the results json will be saved to file."
    )
    parser.add_argument(
        "--n_optuna_trials",
        type=int,
        default=100,
        help="Number of optuna trials in the inner CV hyperparameter loop."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device to run the experiments on."
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of inner and outer CV folds."
    )
    parser.add_argument(
        "--cv_seed",
        type=int,
        default=42,
        help="Seed for all randomness."
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=25,
        help="Number of trials before early stopping in Optuna early stopping callback."
    )
    return parser.parse_args()


from optuna_kfoldCV import run_all_openML_with_model, openML_reg_ids_noCat

if __name__ == "__main__":
    args = parse_args()

    # Run experiments
    for model_name in args.models:
        # baseline models
        if model_name == "End2End":
            eval_fun = evaluate_End2End
        elif model_name == "Ridge":
            eval_fun = evaluate_Ridge
        elif model_name == "RidgeCV":
            eval_fun = evaluate_RidgeCV
        elif model_name == "XGBoostRegressor":
            eval_fun = evaluate_XGBoostRegressor
        # random feature boosting models
        elif model_name == "GradientRFBoost":
            eval_fun = get_GradientRFBoost_eval_fun("SWIM", "dense")
        elif model_name == "GradientRFBoostID":
            eval_fun = get_GradientRFBoost_eval_fun("SWIM", "identity")
        elif model_name == "GreedyRFBoostDense":
            eval_fun = get_GreedyRFBoost_eval_fun("SWIM", "dense")
        elif model_name == "GreedyRFBoostDiag":
            eval_fun = get_GreedyRFBoostDiagScalar_eval_fun("SWIM", "dense", "diag")
        elif model_name == "GreedyRFBoostScalar":
            eval_fun = get_GreedyRFBoostDiagScalar_eval_fun("SWIM", "dense", "scalar")
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # run the experiments
        run_all_openML_with_model(
            dataset_ids = openML_reg_ids_noCat[args.dataset_indices],
            evaluate_model_func = eval_fun,
            name_model = model_name,
            k_folds = args.k_folds,
            cv_seed = args.cv_seed,
            regression_or_classification="regression",
            n_optuna_trials = args.n_optuna_trials,
            device = args.device,
            save_dir = args.save_dir,
            early_stopping_patience = args.early_stopping_patience,
        )