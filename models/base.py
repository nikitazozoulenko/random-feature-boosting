from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import abc

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch import Tensor

from models.ridge_ALOOCV import fit_ridge_ALOOCV


############################################################################
##### Base classes                                                     #####
##### - FittableModule: A nn.Module with .fit(X, y) support            #####
##### - Sequential: chaining together multiple FittableModules         #####
##### - make_fittable: turns type nn.Module into FittableModule        #####
############################################################################


class FittableModule(nn.Module):
    def __init__(self):
        """Base class that wraps nn.Module with a .fit(X, y) 
        and .fit_transform(X, y) method. Requires subclasses to
        implement .forward, as well as either .fit or .fit_transform.
        """
        super(FittableModule, self).__init__()
    

    def fit(self, X: Tensor, y: Tensor):
        """Fits the model given training data X and targets y.

        Args:
            X (Tensor): Training data, shape (N, D).
            y (Tensor): Training targets, shape (N, d).
        """
        self.fit_transform(X, y)
        return self
    

    def fit_transform(self, X: Tensor, y: Tensor) -> Tensor:
        """Fit the module and return the transformed data."""
        self.fit(X, y)
        return self(X)



class Sequential(FittableModule):
    def __init__(self, *layers: FittableModule):
        """
        Args:
            *layers (FittableModule): Variable length argument list of FittableModules to chain together.
        """
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)


    def fit_transform(self, X: Tensor, y: Tensor):
        for layer in self.layers:
            X  = layer.fit_transform(X, y)
        return X


    def forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X)
        return X
    


def make_fittable(module_class: Type[nn.Module]) -> Type[FittableModule]:
    class FittableModuleWrapper(FittableModule, module_class):
        def __init__(self, *args, **kwargs):
            FittableModule.__init__(self)
            module_class.__init__(self, *args, **kwargs)
        
        def fit(self, X: Tensor, y: Tensor):
            return self
    
    return FittableModuleWrapper


Tanh = make_fittable(nn.Tanh)
ReLU = make_fittable(nn.ReLU)
Identity = make_fittable(nn.Identity)
FittableSequential = make_fittable(nn.Sequential)


# class FittableSequential(FittableModule):
#     def __init__(self, *layers):
#         super().__init__()
#         self.sequential = nn.Sequential(*layers)

#     def fit(self, X: Tensor, y: Tensor):
#         return self
        
#     def forward(self, x: Tensor):
#         return self.sequential(x)


##########################################
#### Logistic Regression and RidgeCV  ####
####      classifiers/regressors      ####
##########################################


class RidgeCVModule(FittableModule):
    def __init__(self, 
                lower_alpha: float = 1e-6,
                upper_alpha: float = 10,
                n_alphas: int = 10,
                ):
        """Ridge Regression with optimal l2_reg optimization by
        approximate leave-one-out cross-validation (ALOOCV)"""
        super(RidgeCVModule, self).__init__()
        self.alphas = np.logspace(np.log10(lower_alpha), np.log10(upper_alpha), n_alphas)
        self.W = None
        self.b = None
        self._alpha = None

    def fit(self, X: Tensor, y: Tensor):
        """Fit the RidgeCV model with ALOOCV"""
        self.W, self.b, self._alpha = fit_ridge_ALOOCV(X, y, alphas=self.alphas)
        return self

    def forward(self, X: Tensor) -> Tensor:
        return X @ self.W + self.b
    

class RidgeModule(FittableModule):
    def __init__(self, l2_reg: float = 1e-3):
        super(RidgeModule, self).__init__()
        self.l2_reg = l2_reg
        self.W = None
        self.b = None
    
    def fit(self, X: Tensor, y: Tensor):
        """Fit the Ridge model with a fixed l2_reg"""
        X_mean = X.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        N = X.size(0)
        A = X_centered.T @ X_centered + self.l2_reg * N * torch.eye(X.size(1), dtype=X.dtype, device=X.device)
        B = X_centered.T @ y_centered
        self.W = torch.linalg.solve(A, B)
        self.b = y_mean - (X_mean @ self.W)
        return self
    
    def forward(self, X: Tensor) -> Tensor:
        return X @ self.W + self.b



class LogisticRegressionSGD(FittableModule):
    def __init__(self, 
                 batch_size = 512,
                 num_epochs = 30,
                 lr = 0.01,):
        super(LogisticRegressionSGD, self).__init__()
        self.model = None
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

    def fit(self, X: Tensor, y: Tensor):
        # Determine input and output dimensions
        input_dim = X.size(1)
        if y.dim() > 1 and y.size(1) > 1:
            output_dim = y.size(1)
            y_labels = torch.argmax(y, dim=1)
            criterion = nn.CrossEntropyLoss()
        else:
            output_dim = 1
            y_labels = y.squeeze()
            criterion = nn.BCEWithLogitsLoss()

        # Define the model
        self.model = nn.Linear(input_dim, output_dim)
        device = X.device
        self.model.to(device)

        # DataLoader
        dataset = torch.utils.data.TensorDataset(X, y_labels)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
        )

        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self(X), y

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)


class LogisticRegression(FittableModule):
    def __init__(self, 
                 in_dim: int,
                 out_dim: int = 10,
                 l2_reg: float = 0.001,
                 lr: float = 1.0,
                 max_iter: int = 100,
                 ):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.l2_reg = l2_reg
        self.lr = lr
        self.max_iter = max_iter

        if out_dim > 1:
            self.loss = nn.functional.cross_entropy #this is with logits
        else:
            self.loss = nn.functional.binary_cross_entropy_with_logits


    def fit(self, 
            X: Tensor, 
            y: Tensor,
            init_W_b: Optional[Tuple[Tensor, Tensor]] = None,
            ):
        
        # No onehot encoding
        if y.dim() > 1:
            y_labels = torch.argmax(y, dim=1)
        else:
            y_labels = y

        # Put model on device
        device = X.device
        self.to(device)

        # Initialize weights and bias
        if init_W_b is not None:
            W, b = init_W_b
            self.linear.weight.data = W
            self.linear.bias.data = b
        else:
            nn.init.kaiming_normal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        
        with torch.enable_grad():
            # Optimize
            optimizer = torch.optim.LBFGS(self.linear.parameters(), lr=self.lr, max_iter=self.max_iter)
            def closure():
                optimizer.zero_grad()
                logits = self.linear(X)
                loss = self.loss(logits, y_labels)
                loss += self.l2_reg * torch.linalg.norm(self.linear.weight)**2
                loss.backward()
                return loss
            optimizer.step(closure)
        return self


    def forward(self, X: Tensor) -> Tensor:
        return self.linear(X)