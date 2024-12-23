from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import torch.nn as nn


class SKLearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, modelClass=None, **model_params,):
        self.modelClass = modelClass
        self.model_params = model_params
        self.seed = None
        self.model = None

    def fit(self, X, y):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.model = self.modelClass(**self.model_params)
        self.model.fit(X, y)
        self.classes_ = np.unique([0, 1])
        return self

    def predict(self, X):
        #binary classification
        logits = self.model(X)
        proba = torch.sigmoid(logits).detach().cpu().numpy()
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        #binary classification
        proba_0 = torch.nn.functional.sigmoid(self.model(X))
        return torch.cat((1 - proba_0, proba_0), dim=1).cpu().numpy()
    
    def decision_function(self, X):
        logits = self.model(X)
        return logits.detach().cpu().numpy()

    def set_params(self, **params):
        self.modelClass = params.pop('modelClass', self.modelClass)
        self.seed = params.pop('seed', self.seed)
        self.model_params.update(params)
        return self

    def get_params(self, deep=True):
        params = {'modelClass': self.modelClass}
        params.update(self.model_params)
        return params
    
    def score(self, X, y):
        logits = self.model(X)
        if y.size(1) == 1:
            y_true = y.detach().cpu().numpy()
            y_score = logits.detach().cpu().numpy()
            auc = roc_auc_score(y_true, y_score)
            return auc
        else:
            pred = torch.argmax(logits, dim=1)
            y = torch.argmax(y, dim=1)
            acc = (pred == y).float().mean()
            return acc.detach().cpu().item()
    
    def set_model_eval(self):
        self.model.eval()
    

#### usage example ####
if __name__ == "__main__":
    from models.random_feature_representation_boosting import GradientRFRBoostClassifier

    # synthetic data
    D = 10
    N = 100
    n_classes = 3
    X_train = torch.randn(N, D)
    y_train = torch.randint(0, n_classes, (N,))
    X_test = torch.randn(N, D)
    y_test = torch.randint(0, n_classes, (N,))

    # Define hyperparameter search space
    param_grid = {
            'modelClass': [GradientRFRBoostClassifier],
            'l2_cls': np.logspace(-4, 0, 5),
            'l2_ghat': np.logspace(-7, -3, 5),
            'in_dim': [2],
            'n_classes': [3],
            'hidden_dim': [2],
            'n_layers': [1, 2, 3],
            'randfeat_xt_dim': [512],
            'randfeat_x0_dim': [512],
            'feature_type': ["SWIM"],
            'upscale_type': ["iid"],
            'use_batchnorm': [True],
            'lbfgs_max_iter': [300],
            'lbfgs_lr': [1.0],
            'SWIM_scale': [1.0],
            'activation': ["swim"],
        }
    seed = 42

    # Perform grid search with k-fold cross-validation
    estimator = SKLearnWrapper()
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid= {**param_grid, **{"seed": [seed]}},
        cv=5,
    )
    grid_search.fit(X_train, y_train)

    # fit best model
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    best_model.set_model_eval()
    accuracy = best_model.score(X_test, y_test)