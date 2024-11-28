from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ucimlrepo import fetch_ucirepo 

from optuna_kfoldCV import evaluate_pytorch_model_kfoldcv, evaluate_dataset_with_model
from models.models import GreedyRandFeatBoostRegression, GradientRandFeatBoostRegression, RidgeModule, End2EndMLPResNet
from regression_param_specs import evaluate_Ridge





########################################  |
#####    Code for downloading data #####  |
########################################  V


def normalize_data(X: Tensor):
    """Normalize the features and target variable"""
    X_mean, X_std = X.mean(dim=0, keepdim=True), X.std(dim=0, keepdim=True)
    eps = 1e-6
    X = (X - X_mean) / (X_std+eps)
    return X



def pytorch_load_forestfire(device='cpu'):
    """Downloads and preprocesses the Forest Fire UCI dataset"""
    forest_fires = fetch_ucirepo(id=162) 
    X = forest_fires.data.features 
    y = forest_fires.data.targets 

    # Preprocess the dataset
    month_to_int = {
        "jan": 1., "feb": 2., "mar": 3., "apr": 4., "may": 5., "jun": 6.,
        "jul": 7., "aug": 8., "sep": 9., "oct": 10., "nov": 11., "dec": 12.
    }
    day_to_int = {
        "sun": 1., "mon": 2., "tue": 3., "wed": 4., "thu": 5., "fri": 6., "sat": 7.
    }
    X.loc[:, 'month'] = X['month'].map(month_to_int)
    X.loc[:, 'day'] = X['day'].map(day_to_int)
    y = np.log1p(y)  # Log-transform the target variable

    # make into torch tensors
    X = torch.tensor(X.astype(float).values, dtype=torch.float32, device=device)
    y = torch.tensor(y.values, dtype=torch.float32, device=device)
    X = normalize_data(X)
    y = normalize_data(y)
    return X, y



def pytorch_load_abalone(device='cpu'):
    """Downloads and preprocesses the Abalone dataset"""
    abalone = fetch_ucirepo(id=1) 
    X = abalone.data.features 
    y = abalone.data.targets

    # Preprocess the dataset
    X = pd.get_dummies(X, columns=['Sex'], prefix='', prefix_sep='')

    # make into torch tensors
    X = torch.tensor(X.astype(float).values, dtype=torch.float32, device=device)
    y = torch.tensor(y.values, dtype=torch.float32, device=device)
    X = normalize_data(X)
    y = normalize_data(y)
    return X, y



def pytorch_load_wine_quality(device='cpu'):
    """Downloads and preprocesses the Wine Quality UCI dataset"""
    wine_quality = fetch_ucirepo(id=186) 
    full_data = wine_quality.data.original 


    # Preprocess the dataset
    full_data = full_data.drop_duplicates()
    full_data.loc[:, "color"] = (full_data["color"] == "red")
    X = full_data.drop(columns="quality")
    y = full_data["quality"]
    
    # make into torch tensors
    X = torch.tensor(X.astype(float).values, dtype=torch.float32, device=device)
    y = torch.tensor(y.values, dtype=torch.float32, device=device)[:, None]
    X = normalize_data(X)
    y = normalize_data(y)
    return X, y





def pytorch_load_yearpredictionmsd(device='cpu'):
    """Downloads and preprocesses the Music Year Prediction UCI dataset.

    NOTE fetch_ucirepo(id=203) not supported. Need to download manually"""


######################################################  |
#####  Evaluation/spec functions for each model  #####  |
######################################################  V


def get_GradientRFBoost_eval_fun(
        n_layers: int,
        feature_type: Literal["dense", "SWIM"] = "SWIM",
        upscale : Literal["dense", "SWIM"] = "dense",
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
            ):
        ModelClass = GradientRandFeatBoostRegression
        get_optuna_params = lambda trial : {
            "out_dim": trial.suggest_categorical("out_dim", [y.size(1)]),               # Fixed value
            "n_layers": trial.suggest_categorical("n_layers", [n_layers]),              # Fixed value
            "feature_type": trial.suggest_categorical("feature_type", [feature_type]),  # Fixed value
            "upscale": trial.suggest_categorical("upscale", [upscale]),                 # Fixed value

            "hidden_dim": trial.suggest_int("hidden_dim", 16, 512, log=True),
            "bottleneck_dim": trial.suggest_int("bottleneck_dim", 128, 512, log=True),
            "boost_lr": trial.suggest_float("boost_lr", 0.5, 1.0, log=True),
        }

        return evaluate_pytorch_model_kfoldcv(
            ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device,
        )
    return evaluate_GRFBoost



def get_GreedyRFBoost_eval_fun(
        n_layers: int,
        feature_type: Literal["dense", "SWIM"] = "SWIM",
        sandwich_solver: Literal["scalar", "diag", "dense"] = "dense",
        upscale: Literal["dense", "SWIM"] = "dense",
        ):
    """Returns a function that evaluates the GreedyRFBoost model
    with the specified number of layers"""
    def evaluate_GRFBoost(
            X: Tensor,
            y: Tensor,
            k_folds: int,
            cv_seed: int,
            regression_or_classification: str,
            n_optuna_trials: int,
            device: Literal["cpu", "cuda"],
            ):
        
        ModelClass = GreedyRandFeatBoostRegression
        get_optuna_params = lambda trial : {
            "out_dim": trial.suggest_categorical("out_dim", [y.size(1)]),  # Fixed value
            "n_layers": trial.suggest_categorical("n_layers", [n_layers]),         # Fixed value
            "feature_type": trial.suggest_categorical("feature_type", [feature_type]), # Fixed value
            "sandwich_solver": trial.suggest_categorical("sandwich_solver", [sandwich_solver]), # Fixed value
            "upscale": trial.suggest_categorical("upscale", [upscale]), # Fixed value

            "hidden_dim": trial.suggest_int("hidden_dim", 16, 512, log=True),
            "bottleneck_dim": trial.suggest_int("bottleneck_dim", 128, 512, log=True) if sandwich_solver=="dense" else trial.suggest_categorical("bottleneck_dim", [None]),
            "l2_reg": trial.suggest_float("l2_reg", 1e-6, 0.1, log=True),
            "boost_lr": trial.suggest_float("boost_lr", 0.5, 1.0, log=True),
        }

        return evaluate_pytorch_model_kfoldcv(
            ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device,
        )
    return evaluate_GRFBoost



from regression_param_specs import evaluate_Ridge, evaluate_XGBoostRegressor
def get_Ridge_eval_fun(
        n_layers: int,
        ):
    """Returns a function that evaluates the Ridge model.
    The number of layers is not used in this model."""
    return evaluate_Ridge



def get_XGBoost_eval_fun(
        n_layers: int,
        ):
    """Returns a function that evaluates the XGBoost model.
    The number of layers is not used in this model."""
    return evaluate_XGBoostRegressor



def get_End2End_eval_fun(
        n_layers: int,
        ):
    """Returns a function that evaluates the End2End model
    with the specified number of layers"""
    def evaluate_End2End(
            X: Tensor,
            y: Tensor,
            k_folds: int,
            cv_seed: int,
            regression_or_classification: str,
            n_optuna_trials: int,
            device: Literal["cpu", "cuda"],
            ):
        ModelClass = End2EndMLPResNet
        get_optuna_params = lambda trial : {
            "in_dim": trial.suggest_categorical("in_dim", [X.size(1)]),         # Fixed value
            "out_dim": trial.suggest_categorical("out_dim", [y.size(1)]),       # Fixed value
            "n_blocks": trial.suggest_categorical("n_blocks", [n_layers]),      # Fixed value
            "loss": trial.suggest_categorical("loss", ["mse"]),# Fixed value
            
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 128, log=True),         ## change to stepsize
            "bottleneck_dim": trial.suggest_int("bottleneck_dim", 32, 128, log=True), ## change to stepsize
            "lr": trial.suggest_float("lr", 0.0001, 0.1, log=True),
            "end_lr_factor": trial.suggest_float("end_lr_factor", 0.01, 1.0, log=True),
            "n_epochs": trial.suggest_int("n_epochs", 5, 30, log=True),              ## change to stepsize
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 0.01, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),  ## change to stepsize
        }

        return evaluate_pytorch_model_kfoldcv(
            ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device,
            )
    return evaluate_End2End




def get_RandomFeatureResNet_eval_fun(
        n_layers: int,
        feature_type: Literal["dense", "SWIM"] = "SWIM",
        upscale : Literal["dense", "SWIM"] = "dense",
        ):
    """Returns a function that evaluates the RandomFeatureResNet model
    with the specified number of layers"""
    #TODO



######################################################  |
#####  command line argument to run experiments  #####  |
######################################################  V


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
        "--datasets", 
        nargs='+', 
        type=str, 
        default=["forestfire", "abalone", "wine_quality", "yearpredictionmsd"], 
        help="List of datasets to run."
    )
    parser.add_argument(
        "--MAX_n_layers_for_each_dataset", 
        nargs='+',
        type=int, 
        default=[5, 5, 5, 10],
        help="Maximum number of layers to use for each dataset."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/nikita/Code/random-feature-boosting/save/UCI/",
        help="Directory where the results json will be saved to file."
    )
    parser.add_argument(
        "--n_optuna_trials",
        type=int,
        default=2,  # TODO change to 100
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
        "--start_n_layers_from",
        type=int,
        default=1,
        help="What number of layers to start counting from."
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    # Run experiments
    for MAX_n_layers, dataset_name in zip(args.MAX_n_layers_for_each_dataset, 
                                          args.datasets):
        #download dataset
        print("Downloading dataset", dataset_name)
        if dataset_name == "forestfire":
            X, y = pytorch_load_forestfire(args.device)
        elif dataset_name == "abalone":
            X, y = pytorch_load_abalone(args.device)
        elif dataset_name == "wine_quality":
            X, y = pytorch_load_wine_quality(args.device)
        elif dataset_name == "yearpredictionmsd":
            X, y = pytorch_load_yearpredictionmsd(args.device)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        print("Finished downloading dataset", dataset_name)
        
        # Run experiments for each model
        for model_name in args.models:
            if model_name == "End2End":
                get_eval_fun = lambda t : get_End2End_eval_fun(t)
            elif model_name == "Ridge":
                get_eval_fun = lambda t : get_Ridge_eval_fun(t)
            elif model_name == "GradientRFBoost": #TODO change name to Dense
                get_eval_fun = lambda t : get_GradientRFBoost_eval_fun(t)
            elif model_name == "GreedyRFBoostDiag":
                get_eval_fun = lambda t : get_GreedyRFBoost_eval_fun(t, sandwich_solver="diag")
            elif model_name == "GreedyRFBoostScalar":
                get_eval_fun = lambda t : get_GreedyRFBoost_eval_fun(t, sandwich_solver="scalar")
            elif model_name == "GreedyRFBoost":
                get_eval_fun = lambda t : get_GreedyRFBoost_eval_fun(t)
            elif model_name == "RandomFeatureResNet":
                get_eval_fun = lambda t : get_RandomFeatureResNet_eval_fun(t)
            elif model_name == "XGBoostRegressor":
                get_eval_fun = lambda t : get_XGBoost_eval_fun(t)
            else:
                raise ValueError(f"Unknown model name: {model_name}")
            
            # Run the evaluation function for each number of layers
            for t in range(args.start_n_layers_from, MAX_n_layers+1):
                new_model_name = model_name+f"_t{t}"
                print("t", t, "model_name", model_name, "dataset_name", dataset_name)
                eval_fun = get_eval_fun(t)
                # to not run models that do not depend on t multiple times
                json = evaluate_dataset_with_model(
                    X, y, dataset_name, eval_fun, new_model_name, args.k_folds, args.cv_seed, 
                    "regression", args.n_optuna_trials, args.device, args.save_dir
                    )
                print(json)
                print("Finished running experiments for model", new_model_name, "on dataset", dataset_name, "with", t, "layers.")
                print("Training time for inner folds:", json[dataset_name][new_model_name]["t_fit"])

                if model_name in ["Ridge", "XGBoostRegressor"]:
                    break