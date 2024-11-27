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
from sklearn.linear_model import RidgeClassifierCV
import xgboost as xgb

from models.ridge_ALOOCV import fit_ridge_ALOOCV
from models.sandwiched_least_squares import sandwiched_LS_dense, sandwiched_LS_diag, sandwiched_LS_scalar



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


############################################################################
##### Layers                                                           #####
##### - Dense: Fully connected layer                                   #####
##### - SWIMLayer                                                      #####
##### - RidgeClassifierCV (currently just an sklearn wrapper)          #####
##### - LogisticRegressionSGD                                          #####
##### - LogisticRegression                                             #####
############################################################################

class Dense(FittableModule):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 activation: Optional[nn.Module] = None,
                 ):
        """Dense MLP layer with LeCun weight initialization,
        Gaussan bias initialization."""
        super(Dense, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = activation
    
    def fit(self, X:Tensor, y:Tensor):
        nn.init.normal_(self.dense.weight, mean=0, std=self.in_dim**-0.5)
        nn.init.normal_(self.dense.bias, mean=0, std=self.in_dim**-0.25)
        self.to(X.device)
        return self
    
    def forward(self, X):
        X = self.dense(X)
        if self.activation is not None:
            X = self.activation(X)
        return X
    


class SWIMLayer(FittableModule):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 activation: nn.Module = nn.Tanh(),
                 epsilon: float = 0.01,
                 sampling_method: Literal['uniform', 'gradient'] = 'gradient',
                 ):
        """Dense MLP layer with pair sampled weights (uniform or gradient-weighted).

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            activation (nn.Module): Activation function.
            epsilon (float): Small constant to avoid division by zero.
            sampling_method (str): Pair sampling method. Uniform or gradient-weighted.
        """
        super(SWIMLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.epsilon = epsilon
        self.sampling_method = sampling_method

        self.dense = nn.Linear(in_dim, out_dim)


    def fit(self, X: Tensor, y: Tensor):
        """Given forward-propagated training data X at the previous 
        hidden layer, and supervised target labels y, fit the weights
        iteratively by letting rows of the weight matrix be given by
        pairs of samples from X. See paper for more details.

        Args:
            X (Tensor): Forward-propagated activations of training data, shape (N, D).
            y (Tensor): Training labels, shape (N, p).
        """
        self.to(X.device)
        with torch.no_grad():
            N, D = X.shape
            dtype = X.dtype
            device = X.device
            EPS = torch.tensor(self.epsilon, dtype=dtype, device=device)

            #obtain pair indices
            n = 3*N
            idx1 = torch.arange(0, n, dtype=torch.int32, device=device) % N
            delta = torch.randint(1, N, size=(n,), dtype=torch.int32, device=device)
            idx2 = (idx1 + delta) % N
            dx = X[idx2] - X[idx1]
            dists = torch.linalg.norm(dx, axis=1, keepdims=True)
            dists = torch.maximum(dists, EPS)
            
            if self.sampling_method=="gradient":
                #calculate 'gradients'
                dy = y[idx2] - y[idx1]
                y_norm = torch.linalg.norm(dy, axis=1, keepdims=True) #NOTE 2023 paper uses ord=inf instead of ord=2
                grad = (y_norm / dists).reshape(-1) 
                p = grad/grad.sum()
            elif self.sampling_method=="uniform":
                p = torch.ones(n, dtype=dtype, device=device) / n
            else:
                raise ValueError(f"sampling_method must be 'uniform' or 'gradient'. Given: {self.sampling_method}")

            #sample pairs
            selected_idx = torch.multinomial(
                p,
                self.out_dim,
                replacement=True,
                )
            idx1 = idx1[selected_idx]
            dx = dx[selected_idx]
            dists = dists[selected_idx]

            #define weights and biases
            weights = dx / (dists**2)
            biases = -torch.einsum('ij,ij->i', weights, X[idx1]) - 0.5
            self.dense.weight.data = weights
            self.dense.bias.data = biases
            return self
    

    def forward(self, X):
        return self.activation(self.dense(X))



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
        
        A = X_centered.T @ X_centered + self.l2_reg * torch.eye(X.size(1), dtype=X.dtype, device=X.device)
        B = X_centered.T @ y_centered
        self.W = torch.linalg.solve(A, B)
        self.b = y_mean - (X_mean @ self.W)
        return self
    
    def forward(self, X: Tensor) -> Tensor:
        return X @ self.W + self.b



class RidgeClassifierCVModule(FittableModule):
    def __init__(self, alphas=np.logspace(-1, 3, 10)):
        """RidgeClassifierCV layer using sklearn's RidgeClassifierCV."""
        super(RidgeClassifierCVModule, self).__init__()
        self.ridge = RidgeClassifierCV(alphas=alphas)

    def fit(self, X: Tensor, y: Tensor):
        """Fit the sklearn ridge model."""
        # Make y categorical from one_hot NOTE assumees y one-hot
        y_cat = torch.argmax(y, dim=1)
        X_np = X.detach().cpu().numpy().astype(np.float64)
        y_np = y_cat.detach().cpu().squeeze().numpy().astype(np.float64)
        self.ridge.fit(X_np, y_np)
        return self

    def forward(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy().astype(np.float64)
        y_pred_np = self.ridge.predict(X_np)
        return torch.tensor(y_pred_np, dtype=X.dtype, device=X.device)



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
            self.loss = F.cross_entropy #this is with logits
        else:
            self.loss = F.binary_cross_entropy_with_logits


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


######################################
#####       Residual Block       #####
######################################


def create_layer(
        layer_name:str, 
        in_dim:int, 
        out_dim:int,
        activation: Optional[nn.Module],
        sampling_method: str = "gradient",
        ):
    if layer_name == "dense":
        return Dense(in_dim, out_dim, activation)
    elif layer_name == "SWIM":
        return SWIMLayer(in_dim, out_dim, activation, sampling_method=sampling_method)
    elif layer_name == "identity":
        return Identity()
    else:
        raise ValueError(f"layer_name must be one of ['dense', 'SWIM', 'identity']. Given: {layer_name}")



# class ResidualBlock(FittableModule):
#     def __init__(self, 
#                  in_dim: int,
#                  bottleneck_dim: int,
#                  layer1: str,
#                  layer2: str,
#                  activation: nn.Module = nn.Tanh(),
#                  residual_scale: float = 1.0,
#                  sampling_method: Literal['uniform', 'gradient'] = 'gradient',
#                  ):
#         """Residual block with 2 layers and a skip connection.
        
#         Args:
#             in_dim (int): Input dimension.
#             bottleneck_dim (int): Dimension of the bottleneck layer.
#             layer1 (str): First layer in the block. One of ["dense", "swim", "identity"].
#             layer2 (str): See layer1.
#             activation (nn.Module): Activation function.
#             residual_scale (float): Scale of the residual connection.
#             sampling_method (str): Pair sampling method for SWIM. One of ['uniform', 'gradient'].
#         """
#         super(ResidualBlock, self).__init__()
#         self.residual_scale = residual_scale
#         self.first = create_layer(layer1, in_dim, bottleneck_dim, None, sampling_method)
#         self.activation = activation
#         self.second = create_layer(layer2, bottleneck_dim, in_dim, None, sampling_method)


#     def fit(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
#         with torch.no_grad():
#             X0 = X
#             X, y = self.first.fit(X,y)
#             X = self.activation(X)
#             X, y = self.second.fit(X,y)
#         return X0 + X * self.residual_scale, y


#     def forward(self, X: Tensor) -> Tensor:
#         X0 = X
#         X = self.first(X)
#         X = self.activation(X)
#         X = self.second(X)
#         return X0 + X * self.residual_scale
    

#####################################
##### Residual Networks         #####
##### - ResNet                  #####
##### - NeuralEulerODE          #####
#####################################

# class ResNet(Sequential):
#     def __init__(self, 
#                  in_dim: int,
#                  hidden_size: int,
#                  bottleneck_dim: int,
#                  n_blocks: int,
#                  upsample_layer: Literal['dense', 'SWIM', 'identity'] = 'SWIM',
#                  upsample_activation: nn.Module = nn.Tanh(),
#                  res_layer1: str = "SWIM",
#                  res_layer2: str = "dense",
#                  res_activation: nn.Module = nn.Tanh(),
#                  residual_scale: float = 1.0,
#                  sampling_method: Literal['uniform', 'gradient'] = 'gradient',
#                  output_layer: Literal['ridge', 'dense', 'identity', 'logistic regression'] = 'ridge',
#                  ):
#         """Residual network with multiple residual blocks.
        
#         Args:
#             in_dim (int): Input dimension.
#             hidden_size (int): Dimension of the hidden layers.
#             bottleneck_dim (int): Dimension of the bottleneck layer.
#             n_blocks (int): Number of residual blocks.
#             upsample_layer (str): Layer before any residual connections. One of ['dense', 'SWIM', 'identity'].
#             upsample_activation (nn.Module): Activation function for the upsample layer.
#             res_layer1 (str): First layer in the block. One of ["dense", "swim", "identity"].
#             res_layer2 (str): See layer1.
#             res_activation (nn.Module): Activation function for the residual blocks.
#             residual_scale (float): Scale of the residual connection.
#             sampling_method (str): Pair sampling method for SWIM. One of ['uniform', 'gradient'].
#             output_layer (str): Output layer. One of ['ridge', 'ridge classifier', 'dense', 'identity', 'logistic regression'].
#         """
#         upsample = create_layer(upsample_layer, 
#                                 in_dim, 
#                                 hidden_size, 
#                                 upsample_activation, 
#                                 sampling_method)
#         residual_blocks = [
#             ResidualBlock(hidden_size, 
#                           bottleneck_dim, 
#                           res_layer1, 
#                           res_layer2, 
#                           res_activation, 
#                           residual_scale, 
#                           sampling_method)
#             for _ in range(n_blocks)
#         ]
#         if output_layer == 'dense':
#             out = Dense(hidden_size, 1, None)
#         elif output_layer == 'ridge':
#             out = RidgeCVModule()
#         elif output_layer == 'ridge classifier':
#             out = RidgeClassifierCVModule()
#         elif output_layer == 'identity':
#             out = Identity()
#         elif output_layer == 'logistic regression':
#             out = LogisticRegression()
#         else:
#             raise ValueError(f"output_layer must be one of ['ridge', 'ridge classifier', 'dense', 'identity', 'logistic regression']. Given: {output_layer}")
        
#         super(ResNet, self).__init__(upsample, *residual_blocks, out)



# class NeuralEulerODE(ResNet):
#     def __init__(self, 
#                  in_dim: int,
#                  hidden_size: int,
#                  n_layers: int,
#                  upsample_layer: Literal['dense', 'SWIM', 'identity'] = 'SWIM',
#                  upsample_activation: nn.Module = nn.Tanh(),
#                  res_layer: str = "SWIM",
#                  res_activation: nn.Module = nn.Tanh(),
#                  residual_scale: float = 1.0,
#                  sampling_method: Literal['uniform', 'gradient'] = 'gradient',
#                  output_layer: Literal['ridge', 'dense'] = 'dense',
#                  ):
#         """Euler discretization of Neural ODE."""
#         super(NeuralEulerODE, self).__init__(in_dim, hidden_size, None,
#                                              n_layers, upsample_layer, upsample_activation,
#                                              res_layer, "identity", res_activation,
#                                              residual_scale, sampling_method, output_layer)


######################################
##### End2End MLPResNet       #####
######################################

class End2EndMLPResNet(FittableModule):
    def __init__(self, 
                 in_dim: int,
                 hidden_dim: int,
                 bottleneck_dim: int,
                 out_dim: int,
                 n_blocks: int,
                 activation: nn.Module = nn.ReLU(),
                 loss: Literal["mse", "cross_entropy"] = "mse",
                 lr: float = 1e-3,
                 end_lr_factor: float = 1e-2,
                 n_epochs: int = 10,
                 weight_decay: float = 1e-5,
                 batch_size: int = 64,
                 ):
        """End-to-end trainer for residual networks using Adam optimizer 
        with a CosineAnnealingLR scheduler with end_lr = lr * end_lr_factor.
        
        Args:
            in_dim (int): Input dimension.
            hidden_dim (int): Dimension of the hidden layers.
            bottleneck_dim (int): Dimension of the bottleneck layer.
            out_dim (int): Output dimension.
            n_blocks (int): Number of residual blocks.
            activation (nn.Module): Activation function.
            loss (nn.Module): Loss function.
            lr (float): Learning rate for Adam optimizer.
            end_lr_factor (float): Factor for the end learning rate in the scheduler.
            n_epochs (int): Number of training epochs.
            weight_decay (float): Weight decay for Adam optimizer.
            batch_size (int): Batch size for training.
        """
        super(End2EndMLPResNet, self).__init__()
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Define resnet with batch norm
        self.upsample = nn.Linear(in_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.activation = activation
        
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                activation,
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(n_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, out_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min = lr * end_lr_factor
            )
        
        # Loss (need to do it like this due to optuna)
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown value of loss argument. Given: {loss}")


    def fit(self, X: Tensor, y: Tensor):
        """Trains network end to end with Adam optimizer and a tabular data loader"""
        device = X.device
        self.to(device)

        # DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
        )

        # training loop
        for epoch in tqdm(range(self.n_epochs)):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        return self


    def forward(self, X: Tensor) -> Tensor:
        X = self.upsample(X)
        X = self.batch_norm(X)
        X = self.activation(X)
        for block in self.residual_blocks:
            X = X + block(X)
        X = self.output_layer(X)
        return X






###################################################
### Greedy Boosting special case for regression ###
###################################################


class GreedyRandFeatBoostRegression(FittableModule):
    def __init__(self, 
                 hidden_dim: int = 128,
                 bottleneck_dim: Optional[int] = 32, #if None, use hidden_dim
                 out_dim: int = 1,
                 n_layers: int = 5,
                 activation: nn.Module = nn.Tanh(),
                 l2_reg: float = 0.01,
                 feature_type = "SWIM", # "dense", identity
                 boost_lr: float = 1.0,
                 upscale: Optional[str] = "dense",
                 sandwich_solver : Literal["scalar", "diag", "dense"] = "dense"
                 #TODO add argument which specifies f(x_t) vs f(x_t, x_0)
                 ):
        super(GreedyRandFeatBoostRegression, self).__init__()
        if bottleneck_dim is None:
            bottleneck_dim = hidden_dim
            
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.activation = activation
        self.l2_reg = l2_reg
        self.feature_type = feature_type
        self.boost_lr = boost_lr
        self.upscale = upscale

        if sandwich_solver == "dense":
            self.sandwich_solver = sandwiched_LS_dense
            self.XDelta_op = self.XDelta_dense
        elif sandwich_solver == "diag":
            self.sandwich_solver = sandwiched_LS_diag
            self.XDelta_op = self.XDelta_diag
            self.bottleneck_dim = hidden_dim
        elif sandwich_solver == "scalar":
            self.sandwich_solver = sandwiched_LS_scalar
            self.XDelta_op = self.XDelta_scalar
            self.bottleneck_dim = hidden_dim
        

        self.W = None
        self.b = None
        self.layers = []
        self.deltas = []


    def fit(self, X: Tensor, y: Tensor):
        with torch.no_grad():
            #optional upscale
            if self.upscale == "dense":
                self.upscale = create_layer(self.upscale, X.shape[1], self.hidden_dim, None)
                X = self.upscale.fit_transform(X, y)
            elif self.upscale == "SWIM":
                self.upscale = create_layer(self.upscale, X.shape[1], self.hidden_dim, self.activation)
                X = self.upscale.fit_transform(X, y)

            # Create regressor W_0
            self.W, self.b, alpha = fit_ridge_ALOOCV(X, y)

            # Layerwise boosting
            for t in range(self.n_layers):
                # Step 1: Create random feature layer   
                layer = create_layer(self.feature_type, self.hidden_dim, self.bottleneck_dim, self.activation)
                F = layer.fit_transform(X, y)
                self.layers.append(layer)

                # Step 2: Greedily minimize R(W_t, Phi_t + Delta F)
                R = y - X @ self.W - self.b # residual
                Delta = self.sandwich_solver(R, self.W, F, self.l2_reg)
                self.deltas.append(Delta)

                # Step 3: Learn top level classifier W_t
                X = X + self.boost_lr * self.XDelta_op(F, Delta)
                self.W, self.b, alpha = fit_ridge_ALOOCV(X, y)

            return X @ self.W + self.b, y

    @staticmethod
    def XDelta_scalar(X: Tensor, Delta: Tensor) -> Tensor:
        return X * Delta
    
    @staticmethod
    def XDelta_diag(X: Tensor, Delta: Tensor) -> Tensor:
        return X * Delta[None, :]
    
    @staticmethod
    def XDelta_dense(X: Tensor, Delta: Tensor) -> Tensor:
        return X @ Delta

    def forward(self, X: Tensor) -> Tensor:
        with torch.no_grad():
            #upscale
            if self.upscale is not None:
                X = self.upscale(X)
            # Boosting
            for layer, Delta in zip(self.layers, self.deltas):
                X = X + self.boost_lr * self.XDelta_op(layer(X), Delta)
            # Top level regressor
            return X @ self.W + self.b

        




class GradientRandFeatBoostRegression(FittableModule):
    def __init__(self, 
                 hidden_dim: int = 128,
                 bottleneck_dim: int = 128,
                 out_dim: int = 1,
                 n_layers: int = 5,
                 activation: nn.Module = nn.Tanh(),
                #  l2_reg: float = 0.01,   #TODO ALOOCV or fixed l2_reg
                 feature_type = "SWIM", # "dense", identity
                 boost_lr: float = 1.0,
                 upscale: Optional[str] = "dense",
                 ):
        super(GradientRandFeatBoostRegression, self).__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.activation = activation
        # self.l2_reg = l2_reg
        self.feature_type = feature_type
        self.boost_lr = boost_lr
        self.upscale = upscale

        # save for now. for more memory efficient implementation, we can remove a lot of this
        self.W = None
        self.b = None
        self.layers = []
        self.deltas = []


    def fit(self, X: Tensor, y: Tensor):
        with torch.no_grad():
            #optional upscale
            if self.upscale == "dense":
                self.upscale = create_layer(self.upscale, X.shape[1], self.hidden_dim, None)
                X = self.upscale.fit_transform(X, y)
            elif self.upscale == "SWIM":
                self.upscale = create_layer(self.upscale, X.shape[1], self.hidden_dim, self.activation)
                X = self.upscale.fit_transform(X, y)

            # Create regressor W_0
            self.W, self.b, _ = fit_ridge_ALOOCV(X, y)

            # Layerwise boosting
            N = X.size(0)
            for t in range(self.n_layers):
                # Step 1: Create random feature layer   
                layer = create_layer(self.feature_type, self.hidden_dim, self.bottleneck_dim, self.activation)
                F = layer.fit_transform(X, y)

                # Step 2: Obtain activation gradient and learn Delta
                # X shape (N, D) --- ResNet neurons
                # F shape (N, p) --- random features
                # y shape (N, d) --- target
                # W shape (D, d) --- top level classifier
                # G shape (N, D) --- gradient of neurons
                # r shape (N, d) --- residual at currect boosting iteration

                r = y - X @ self.W - self.b
                G = r @ self.W.T
                
                # fit to negative gradient (finding functional direction)
                Delta, Delta_b, _ = fit_ridge_ALOOCV(F, G)
                Ghat = F @ Delta + Delta_b

                # Line search closed form risk minimization of R(W_t, Phi_{t+1})
                linesearch = sandwiched_LS_scalar(r, self.W, Ghat, 0.01)


                # Step 3: Learn top level classifier
                X = X + self.boost_lr * linesearch * Ghat
                self.W, self.b, alpha = fit_ridge_ALOOCV(X, y)

                #update Delta scale
                Delta = Delta * linesearch
                Delta_b = Delta_b * linesearch

                # store
                self.layers.append(layer)
                self.deltas.append((Delta, Delta_b))

            return X @ self.W + self.b, y


    def forward(self, X: Tensor) -> Tensor:
        with torch.no_grad():
            if self.upscale is not None:
                X = self.upscale(X)
            for layer, (Delta, Delta_b) in zip(self.layers, self.deltas):
                X = X + self.boost_lr * (layer(X) @ Delta + Delta_b)
            return X @ self.W + self.b
        







def line_search_cross_entropy(cls, X, y, G_hat):
    """Solves the line search risk minimizatin problem
    R(W, X + a * g) for mutliclass cross entropy loss"""
    # No onehot encoding
    if y.dim() > 1:
        y_labels = torch.argmax(y, dim=1)
    else:
        y_labels = y

    # Optimize
    with torch.enable_grad():
        alpha = torch.tensor([0.0], requires_grad=True, device=X.device, dtype=X.dtype)
        optimizer = torch.optim.LBFGS([alpha])
        def closure():
            optimizer.zero_grad()
            logits = cls(X + alpha * G_hat)
            loss = nn.functional.cross_entropy(logits, y_labels)
            loss.backward()
            print("linesearch loss", loss)
            return loss
        optimizer.step(closure)

    return alpha.detach().item()




class GradientRandomFeatureBoostingClassification(FittableModule):
    def __init__(self, 
                 hidden_dim: int = 128, # TODO
                 bottleneck_dim: int = 128,
                 out_dim: int = 10,
                 n_layers: int = 5,
                 activation: nn.Module = nn.Tanh(),
                 l2_reg: float = 1,
                 feature_type = "SWIM", # "dense", identity
                 boost_lr: float = 1.0,
                 upscale: Optional[str] = "dense",
                 ):
        super(GradientRandomFeatureBoostingClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.activation = activation
        self.l2_reg = l2_reg
        self.feature_type = feature_type
        self.boost_lr = boost_lr
        self.upscale = upscale

        # save for now. for more memory efficient implementation, we can remove a lot of this
        self.classifiers = []
        self.alphas = []
        self.layers = []
        self.deltas = []


    def fit(self, X: Tensor, y: Tensor):
        with torch.no_grad():

            #optional upscale
            if self.upscale == "dense":
                self.upscale = create_layer(self.upscale, X.shape[1], self.hidden_dim, None)
                X = self.upscale.fit_transform(X, y)
            elif self.upscale == "SWIM":
                self.upscale = create_layer(self.upscale, X.shape[1], self.hidden_dim, self.activation)
                X = self.upscale.fit_transform(X, y)

            # Create classifier W_0
            cls = LogisticRegression(
                in_dim = self.hidden_dim,
                out_dim = self.out_dim,
                l2_reg = self.l2_reg,
                max_iter = 100,
            ).to(X.device)
            cls.fit(X, y)
            self.classifiers.append(cls)

            # Layerwise boosting
            N = X.size(0)
            for t in range(self.n_layers):
                # Step 1: Create random feature layer   
                layer = create_layer(self.feature_type, self.hidden_dim, self.bottleneck_dim, self.activation)
                F = layer.fit_transform(X, y)

                # Step 2: Obtain activation gradient
                # X shape (N, D) --- ResNet neurons
                # F shape (N, p) --- random features
                # y shape (N, d) --- one-hot target
                # r shape (N, D) --- residual at currect boosting iteration
                # W shape (D, d) --- top level classifier
                # probs shape (N, d) --- predicted probabilities


                probs = nn.functional.softmax(cls(X), dim=1)
                G = (y - probs) @ cls.linear.weight #negative gradient TODO divide by N?

                # fit Least Squares to negative gradient (finding functional direction)
                Delta, Delta_b, _ = fit_ridge_ALOOCV(F, G)
                print("alpha", _)

                # Line search for risk minimization of R(W_t, Phi_t + linesearch * G_hat)
                G_hat = F @ Delta + Delta_b
                linesearch = line_search_cross_entropy(cls, X, y, G_hat)
                print("Linesearch", linesearch)
                print("Gradient hat norm", torch.linalg.norm(G_hat))

                # Step 3: Learn top level classifier
                X = X + self.boost_lr * linesearch * G_hat
                cls = LogisticRegression(
                    in_dim = self.hidden_dim,
                    out_dim = self.out_dim,
                    l2_reg = self.l2_reg,
                    max_iter = 20,
                ).to(X.device)
                cls.fit(
                    X, 
                    y, 
                    init_W_b = (cls.linear.weight.detach().clone(), cls.linear.bias.detach().clone()) #TODO do i want this? or start from scratch?
                )

                #update Delta scale
                Delta = Delta * linesearch
                Delta_b = Delta_b * linesearch

                # store
                self.layers.append(layer)
                self.deltas.append((Delta, Delta_b))
                self.classifiers.append(cls)

        return cls(X), y


    def forward(self, X: Tensor) -> Tensor:
        with torch.no_grad():
            if self.upscale is not None:
                X = self.upscale(X)
            for layer, (Delta, Delta_b) in zip(self.layers, self.deltas):
                X = X + self.boost_lr * (layer(X) @ Delta + Delta_b)
            return self.classifiers[-1](X)
        



# TODO do layerwise SGD training for the stagewise boosting








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
        return self(X), y

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
        # Make y categorical from one_hot NOTE assumees y one-hot
        X_np = X.detach().cpu().numpy()
        y_np = torch.argmax(y, dim=1).detach().cpu().squeeze().numpy()
        self.model.fit(X_np, y_np)
        return self(X), y

    def forward(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy()
        y_pred_np = self.model.predict_proba(X_np)
        return torch.tensor(y_pred_np, dtype=X.dtype, device=X.device)