from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor

from models.models import RidgeCVModule, RidgeModule, XGBoostRegressorWrapper
from models.models import GradientRandFeatBoostReg, End2EndMLPResNet
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
    ModelClass = GradientRandFeatBoostReg   #TODO UPDATE PARAMS FOR X0Xt
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
        #"subsample": trial.suggest_float("subsample", 0.5, 1.0),
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


######################################################  |
#####  command line argument to run experiments  #####  |
######################################################  V

# # start by using all datasets. I can change this later
# def parse_args():
#     parser = argparse.ArgumentParser(description="Run experiments with different models and datasets.")
#     parser.add_argument(
#         "--models", 
#         nargs='+', 
#         type=str, 
#         default=["GradientRFBoost", "GreedyRFBoost"], 
#         help="List of model names to run."
#     )
#     # parser.add_argument(
#     #     "--datasets", 
#     #     nargs='+', 
#     #     type=str, 
#     #     default=["forestfire", "abalone", "wine_quality", "yearpredictionmsd"], 
#     #     help="List of datasets to run."
#     # )
#     parser.add_argument(
#         "--save_dir",
#         type=str,
#         default="/home/nikita/Code/random-feature-boosting/save/UCI/",
#         help="Directory where the results json will be saved to file."
#     )
#     parser.add_argument(
#         "--n_optuna_trials",
#         type=int,
#         default=100,
#         help="Number of optuna trials in the inner CV hyperparameter loop."
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda",
#         help="PyTorch device to run the experiments on."
#     )
#     parser.add_argument(
#         "--k_folds",
#         type=int,
#         default=5,
#         help="Number of inner and outer CV folds."
#     )
#     parser.add_argument(
#         "--cv_seed",
#         type=int,
#         default=42,
#         help="Seed for all randomness."
#     )
#     return parser.parse_args()



# if __name__ == "__main__":
#     args = parse_args()

#     # Run experiments
#     for MAX_n_layers, dataset_name in zip(args.MAX_n_layers_for_each_dataset, 
#                                           args.datasets):
#         #download dataset
#         print("Downloading dataset", dataset_name)
#         if dataset_name == "forestfire":
#             X, y = pytorch_load_forestfire(args.device)
#         elif dataset_name == "abalone":
#             X, y = pytorch_load_abalone(args.device)
#         elif dataset_name == "wine_quality":
#             X, y = pytorch_load_wine_quality(args.device)
#         elif dataset_name == "yearpredictionmsd":
#             X, y = pytorch_load_yearpredictionmsd(args.device)
#         else:
#             raise ValueError(f"Unknown dataset name: {dataset_name}")
#         print("Finished downloading dataset", dataset_name)
        
#         # Run experiments for each model
#         for model_name in args.models:
#             if model_name == "End2End":
#                 get_eval_fun = lambda t : get_End2End_eval_fun(t)
#             elif model_name == "Ridge":
#                 get_eval_fun = lambda t : get_Ridge_eval_fun(t)
#             elif model_name == "GradientRFBoost": #TODO change name to Dense
#                 get_eval_fun = lambda t : get_GradientRFBoost_eval_fun(t)
#             elif model_name == "GreedyRFBoostDiag":
#                 get_eval_fun = lambda t : get_GreedyRFBoost_eval_fun(t, sandwich_solver="diag")
#             elif model_name == "GreedyRFBoostScalar":
#                 get_eval_fun = lambda t : get_GreedyRFBoost_eval_fun(t, sandwich_solver="scalar")
#             elif model_name == "GreedyRFBoost":
#                 get_eval_fun = lambda t : get_GreedyRFBoost_eval_fun(t)
#             elif model_name == "RandomFeatureResNet":
#                 get_eval_fun = lambda t : get_RandomFeatureResNet_eval_fun(t)
#             elif model_name == "XGBoostRegressor":
#                 get_eval_fun = lambda t : get_XGBoost_eval_fun(t)
#             else:
#                 raise ValueError(f"Unknown model name: {model_name}")
            
#             # Run the evaluation function for each number of layers
#             for t in range(args.start_n_layers_from, MAX_n_layers+1):
#                 new_model_name = model_name+f"_t{t}"
#                 print("t", t, "model_name", model_name, "dataset_name", dataset_name)
#                 eval_fun = get_eval_fun(t)
#                 # to not run models that do not depend on t multiple times
#                 json = evaluate_dataset_with_model(
#                     X, y, dataset_name, eval_fun, new_model_name, args.k_folds, args.cv_seed, 
#                     "regression", args.n_optuna_trials, args.device, args.save_dir
#                     )
#                 print(json)
#                 print("Finished running experiments for model", new_model_name, "on dataset", dataset_name, "with", t, "layers.")
#                 print("Training time for inner folds:", json[dataset_name][new_model_name]["t_fit"])

#                 if model_name in ["Ridge", "XGBoostRegressor"]:
#                     break

# if __name__ == "__main__":
#     pass