from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import abc

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch import Tensor
import xgboost as xgb

from models.base import FittableModule


######################################
###### XGBoost Wrappers ##############
######################################


class XGBoostRegressorWrapper(FittableModule):
    def __init__(self, 
                 **kwargs,
                 ):
        super(XGBoostRegressorWrapper, self).__init__()
        self.model = xgb.XGBRegressor(
            **kwargs,
        )


    def fit(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        if y.dim() == 1:
            y_np = y_np[:, None]
        self.model.fit(X_np, y_np)
        return self


    def forward(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy()
        y_pred_np = self.model.predict(X_np)
        if y_pred_np.ndim == 1:
            y_pred_np = y_pred_np[:, None]
        return torch.tensor(y_pred_np, dtype=X.dtype, device=X.device)
    


class XGBoostClassifierWrapper(FittableModule):
    def __init__(self, 
                 **kwargs,
                 ):
        super(XGBoostClassifierWrapper, self).__init__()
        self.model = xgb.XGBClassifier(
            **kwargs,
        )


    def fit(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        X_np = X.detach().cpu().numpy()
        if y.dim() == 2:
            if y.size(1) > 2:  # Multiclass
                self.model.n_classes_ = y.size(1)
                self.model.set_params(num_class=y.size(1))
                y_np = torch.argmax(y, dim=1).detach().cpu().numpy()
            else:  # Binary
                y_np = y[:, 0].detach().cpu().numpy()
        else:
            y_np = y.detach().cpu().numpy()
        self.model.fit(X_np, y_np)
        return self


    def forward(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy()
        y_pred_np = self.model.predict_proba(X_np)
        
        # Convert probabilities to logits using the inverse of sigmoid/softmax
        eps = 1e-6
        if y_pred_np.shape[1] == 2:
            # Binary case: use inverse sigmoid (logit)
            proba = np.clip(y_pred_np[:, 1], eps, 1 - eps)  # Clip probabilities away from 0 and 1
            logits = np.log(proba / (1 - proba))
            logits = logits[:, None]  # add dimension to match shape [N, 1]
        else:
            # Multiclass case: use inverse softmax (log)
            proba = np.clip(y_pred_np, eps, 1.0)  # Clip probabilities away from 0
            logits = np.log(proba)
            
        return torch.tensor(logits, dtype=X.dtype, device=X.device)
            







from models.random_feature_representation_boosting import GradientRFRBoostRegressor


class GRFRBoostedXGBoostRegressor(FittableModule):
    def __init__(self, 
                 **kwargs,
                 ):
        super(GRFRBoostedXGBoostRegressor, self).__init__()
        xgboost_args = [
            "objective",
            "alpha",
            "lambda",
            "learning_rate",
            "n_estimators",
            "max_depth",
            "subsample",
            "colsample_bytree",
        ]
        grfr_args = [
            "in_dim",
            "out_dim",
            "upscale_type",
            "feature_type",
            "randfeat_xt_dim",
            "randfeat_x0_dim",
            "activation",
            "n_layers",
            "boost_lr",
            "l2_reg",
            "l2_ghat",
            "return_features",
            "use_batchnorm",
        ]
        self.xgb_wrapper = XGBoostRegressorWrapper(
            **{k: v for k, v in kwargs.items() if k in xgboost_args},
        )
        self.grfr = GradientRFRBoostRegressor(
            **{k: v for k, v in kwargs.items() if k in grfr_args},
        )


    def fit(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        boosted_representation = self.grfr.fit_transform(X, y)
        self.xgb_wrapper.fit(boosted_representation, y)
        return self


    def forward(self, X: Tensor) -> Tensor:
        boosted_representation = self.grfr(X)
        return self.xgb_wrapper(boosted_representation)
