from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import time
import json
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor
import pandas as pd
import openml
import optuna
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb


#########################  |
##### Dataset Code  #####  |
#########################  V


openML_reg_ids_noCat = [ 
    44957,44959,44960,44963,44964,44965,44969,44970,
    44971,44972,44973,44975,44976,44977,44978,44980,
    44981,44983,44994,45402
    ]
openML_cls_ids_nFeatsLess500_noCat_noMissing = [
    6,11,12,14,16,18,22,28,32,37,44,54,182,458,1049,
    1050,1063,1067,1068,1462,1464,1475,1487,1489,1494,
    1497,1501,1510,4538,23517,40499,40979,40982,40983,
    40984,40994,41027
    ] # https://www.openml.org/search?type=study&study_type=task&id=99&sort=runs_included


def np_load_openml_dataset(
        dataset_id: int, 
        regression_or_classification: str = "regression",
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downloads the openML dataset and normalizes it.
    For regression, it also normalizes the targets.
    For classification, it makes one-hot encodings.
    
    Returns X (shape N,D), and y (shape N,d) for regression or 
    one-hot (N,C) for classification.
    """
    # Fetch dataset from OpenML by its ID
    dataset = openml.datasets.get_dataset(dataset_id)
    df, _, categorical_indicator, attribute_names = dataset.get_data()
    y = np.array(df.pop(dataset.default_target_attribute))
    X = np.array(df).astype(np.float32)

    #normalize
    X = X - X.mean(axis=0, keepdims=True)
    X = X / (X.std(axis=0, keepdims=True) + 1e-5)
    X = np.clip(X, -3, 3)
    if regression_or_classification == "regression":
        y = y - y.mean()
        y = y / (y.std() + 1e-5)
        y = np.clip(y, -3, 3)
        y = y.astype(np.float32)
        if y.ndim == 1:
            y = y[:, None]
    else:
        y = pd.get_dummies(y).values.astype(np.float32)

    return X, y



def pytorch_load_openml_dataset(
        dataset_id: int, 
        regression_or_classification: Literal["classification", "regression"],
        device: str = "cpu",
        ) -> Tuple[Tensor, Tensor]:
    """
    See 'np_load_openml_dataset' for preprocessing details.
    Converts arrays to PyTorch tensors and moves them to the device.
    """
    X, y = np_load_openml_dataset(dataset_id, regression_or_classification)
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y).to(device)
    
    return X, y


###################################################################  |
#####  Boilerplate code for tabular PyTorch model evaluation  #####  |
#####  with Optuna hyperparameter tuning inner kfoldcv        #####  |
###################################################################  V


def get_pytorch_optuna_cv_objective(
        trial,
        ModelClass: Callable,
        get_optuna_params: Callable,
        X_train: Tensor, 
        y_train: Tensor, 
        k_folds: int,
        cv_seed: int,
        regression_or_classification: Literal["classification", "regression"],
        ):
    """The objective to be minimized in Optuna's 'study.optimize(objective, n_trials)' function."""
    
    params = get_optuna_params(trial)

    inner_cv = KFold(n_splits=k_folds, shuffle=True, random_state=cv_seed)
    scores = []
    for inner_train_idx, inner_valid_idx in inner_cv.split(X_train):
        X_inner_train, X_inner_valid = X_train[inner_train_idx], X_train[inner_valid_idx]
        y_inner_train, y_inner_valid = y_train[inner_train_idx], y_train[inner_valid_idx]

        np.random.seed(cv_seed)
        torch.manual_seed(cv_seed)
        torch.cuda.manual_seed(cv_seed)
        model = ModelClass(**params)
        model.fit(X_inner_train, y_inner_train)

        preds = model(X_inner_valid)
        if regression_or_classification == "classification":
            preds = torch.argmax(preds, dim=1)
            gt = torch.argmax(y_inner_valid, dim=1)
            acc = (preds == gt).float().mean()
            scores.append(-acc.item()) #score is being minimized in Optuna
        else:
            rmse = torch.sqrt(nn.functional.mse_loss(y_inner_valid, preds))
            scores.append(rmse.item())

    return np.mean(scores)


def evaluate_pytorch_model_single_fold(
        ModelClass : Callable,
        get_optuna_params : Callable,
        X_train: Tensor,
        X_test: Tensor,
        y_train: Tensor,
        y_test: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: Literal["classification", "regression"],
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    """
    Evaluates a PyTorch model on a specified Train and Test set.
    Hyperparameters are tuned using Optuna with an inner k-fold CV loop.

    Returns the train and test scores, the time to fit the model, 
    inference time, and the best hyperparameters.
    """
    #hyperparameter tuning with Optuna
    study = optuna.create_study(direction="minimize", )
    objective = lambda trial: get_pytorch_optuna_cv_objective(
        trial, ModelClass, get_optuna_params, X_train, y_train, 
        k_folds, cv_seed, regression_or_classification
        )
    study.optimize(objective, n_trials=n_optuna_trials)

    #fit model with optimal hyperparams
    np.random.seed(cv_seed)
    torch.manual_seed(cv_seed)
    torch.cuda.manual_seed(cv_seed)
    t0 = time.perf_counter()
    model = ModelClass(**study.best_params).to(device)
    model.fit(X_train, y_train)

    #predict
    t1 = time.perf_counter()
    preds_train = model(X_train)
    preds_test = model(X_test)
    t2 = time.perf_counter()

    #evaluate
    if regression_or_classification == "classification":
        preds_train = torch.argmax(preds_train, dim=1)
        gt_train = torch.argmax(y_train, dim=1)
        acc_train = (preds_train == gt_train).float().mean()
        score_train = -acc_train

        preds_test = torch.argmax(preds_test, dim=1)
        gt_test = torch.argmax(y_test, dim=1)
        acc_test = (preds_test == gt_test).float().mean()
        score_test = -acc_test
    else:
        preds_train = model(X_train)
        score_train = torch.sqrt(nn.functional.mse_loss(y_train, preds_train))

        preds_test = model(X_test)
        score_test = torch.sqrt(nn.functional.mse_loss(y_test, preds_test))
    
    return (score_train.item(), score_test.item(), t1-t0, t2-t1, study.best_params.copy())

    

def evaluate_pytorch_model_kfoldcv(
        ModelClass : Callable,
        get_optuna_params : Callable,
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: Literal["classification", "regression"],
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        ):
    """
    Evaluates a PyTorch model using k-fold cross-validation,
    with an inner Optuna hyperparameter tuning loop for each fold.
    The model is then trained on the whole fold train set and evaluated
    on the fold test set.

    Inner and outer kFoldCV use the same number of folds.

    Regression: RMSE is used as the evaluation metric.
    Classification: (negative) Accuracy is used as the evaluation metric.
    """
    outer_cv = KFold(n_splits=k_folds, shuffle=True, random_state=cv_seed)
    outer_train_scores = []
    outer_test_scores = []
    chosen_params = []
    fit_times = []
    inference_times = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        score_train, score_test, t_fit, t_inference, best_params = evaluate_pytorch_model_single_fold(
            ModelClass, get_optuna_params,
            X_train, X_test, y_train, y_test, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device
            )

        #save
        outer_train_scores.append(score_train)
        outer_test_scores.append(score_test)
        fit_times.append(t_fit)
        inference_times.append(t_inference)
        chosen_params.append(best_params)
    
    return (outer_train_scores,
            outer_test_scores,
            fit_times,
            inference_times,
            chosen_params,
            )


def save_experiments_json(
        experiments: Dict[str, Dict[str, Dict[str, Any]]],
        save_path: str,
        ):
    with open(save_path, 'w') as f:
        json.dump(experiments, f, indent=4)


def evaluate_dataset_with_model(
        X: Tensor,
        y: Tensor,
        name_dataset: str,
        evaluate_model_func: Callable,
        name_model: str,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: Literal["classification", "regression"],
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        save_dir: Optional[str] = None,
        ):
    """Evaluates a model on a given tabular dataset (X, y).
    Returns a json of the kfoldCV results.

    Args:
        X (Tensor): Shape (N, D) of the input features.
        y (Tensor): Shape (N, C) for classification (NOTE one-hot), or (N, d) 
                    for regression (NOTE y.dim==2 even if d==1).
        name_dataset (str): Name of the dataset.
        evaluate_model_func (Callable): Function that evaluates the model, 
                                        see e.g. 'evaluate_LogisticRegression'.
        name_model (str): Name of the model.
        k_folds (int): Number of folds in the outer and inner CV.
        cv_seed (int): Seed for all the randomness.
        regression_or_classification (str): Either 'classification' or 'regression'
        n_optuna_trials (int): Number of Optuna trials for hyperparameter tuning.
        device (str): PyTorch device.
        save_dir (Optional[str]): If not None, path to the save directory. 
    """
    np.random.seed(cv_seed)
    torch.manual_seed(cv_seed)
    torch.cuda.manual_seed(cv_seed)

    # Fetch and process each dataset
    results = evaluate_model_func(
        X, y, k_folds, cv_seed, regression_or_classification, n_optuna_trials, device
        )
    
    # store results in nested dict
    experiments = {}
    experiments[str(name_dataset)] = {}
    experiments[str(name_dataset)][name_model] = {
        "score_train": results[0],
        "score_test": results[1],
        "t_fit": results[2],
        "t_inference": results[3],
        "hyperparams": results[4],
    }

    # Save results if specified
    if save_dir is not None:
        path = os.path.join(
            save_dir, 
            f"{regression_or_classification}_{str(name_dataset)}_{name_model}.json"
            )
        save_experiments_json(experiments, path)

    return experiments



def run_all_openML_with_model(
        dataset_ids: List[int],
        evaluate_model_func: Callable,
        name_model: str,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: Literal["classification", "regression"],
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        save_dir: Optional[str] = None,
        save_experiments_individually: bool = False,
        ):
    """Evaluates a model on a list of OpenML datasets.

    Args:
        dataset_ids (List[int]): List of OpenML dataset IDs.
        evaluate_model_func (Callable): Function that evaluates the model, 
                                        see e.g. 'evaluate_LogisticRegression'.
        name_model (str): Name of the model.
        k_folds (int): Number of folds in the outer and inner CV.
        cv_seed (int): Seed for all the randomness.
        regression_or_classification (str): Either 'classification' or 'regression'
        n_optuna_trials (int): Number of Optuna trials for hyperparameter tuning.
        device (str): PyTorch device.
        save_dir (Optional[str]): If not None, path to the save directory.
        save_experiments_individually (bool): If True, saves each dataset experiment in a separate json file.
    """
    # Fetch and process each dataset
    experiments = {}
    for i, dataset_id in enumerate(dataset_ids):
        dataset_id = str(dataset_id)
        X, y = pytorch_load_openml_dataset(dataset_id, regression_or_classification)
        
        save_each_dir = save_dir if save_experiments_individually else None
        json = evaluate_dataset_with_model(
            X, y, dataset_id, evaluate_model_func, name_model, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device, save_each_dir
            )
        experiments[dataset_id] = json[dataset_id]
        print(f" {i+1}/{len(dataset_ids)} Processed dataset {dataset_id}")
    
    # Save results
    if save_dir is not None:
        path = os.path.join(save_dir, f"{regression_or_classification}_{name_model}.json")
        save_experiments_json(experiments, path)
    return experiments


# ###### usage example ######
# run_all_openML_with_model(
#     openML_cls_ids_nFeatsLess500_noCat_noMissing[0:2], 
#     evaluate_LogisticRegression,
#     name_model="LogisticRegression",
#     k_folds=5,
#     cv_seed=42,
#     regression_or_classification="classification",
#     n_optuna_trials=100,
#     device="cuda",
#     save_dir = "/home/nikita/Code/zephyrox/pytorch_based/SWIM/save/"
# )