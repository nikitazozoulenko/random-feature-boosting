from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import abc

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch import Tensor
import torchmin

from models.sandwiched_least_squares import sandwiched_LS_dense, sandwiched_LS_diag, sandwiched_LS_scalar
from models.swim import SWIMLayer
from models.base import FittableModule, RidgeModule, FittableSequential, Identity, LogisticRegression




############################################################################
################# Base classes for Random Feature Boosting #################
############################################################################

def create_layer(
        in_dim: int, 
        out_dim: int, 
        layer_type: Literal["iid", "SWIM", "identity"],
        #iid_factor = None
        #swim_epsilon = None
        #swim_c = None
        ):
    """Takes in the input and output dimensions and returns 
    a layer of the specified type."""
    if layer_type == "iid":
        layer = FittableSequential( nn.Linear(in_dim, out_dim), nn.Tanh() )
    elif layer_type == "SWIM":
        layer = SWIMLayer(in_dim, out_dim, activation=nn.Tanh())
    elif layer_type == "identity":
        layer = Identity()
    else:
        raise ValueError(f"Unknown upscale type {upscale_type}")
    return layer


class Upscale(FittableModule):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 upscale_type: Literal["iid", "SWIM", "identity"] = "iid",
                 ):
        self.upscale_type = upscale_type
        super(Upscale, self).__init__()
        self.upscale = create_layer(in_dim, hidden_dim, upscale_type)
    
    def fit(self, X: Tensor, y: Tensor):
        self.upscale.fit(X, y)
        return self

    def forward(self, X: Tensor):
        return self.upscale(X)



class RandomFeatureLayer(nn.Module):
    @abc.abstractmethod
    def fit_transform(self, Xt: Tensor, X0: Tensor, y: Tensor) -> Tensor:
        """Takes in both Xt and X0 and y and fits the random 
        feature layer and returns the random features"""

    @abc.abstractmethod
    def forward(self, Xt: Tensor, X0: Tensor) -> Tensor:
        """Takes in both Xt and X0 and returns the random features"""



class GhatBoostingLayer(nn.Module):
    @abc.abstractmethod
    def fit_transform(self, F: Tensor, Xt: Tensor, y: Tensor, top_level_module: FittableModule) -> Tensor:
        """Takes in the random features, resnet representations Xt, target y, 
        and the top level module and fits the boosting layer (functional gradient), 
        and returns the gradient estimates"""

    @abc.abstractmethod
    def forward(self, F: Tensor) -> Tensor:
        """Takes in the random features and returns the gradient estimates"""



class BaseGRFRBoost(FittableModule):
    def __init__(
            self, 
            upscale: Upscale,
            top_level_modules: List[FittableModule],
            random_feature_layers: List[RandomFeatureLayer],
            ghat_boosting_layers: List[GhatBoostingLayer],
            boost_lr: float = 1.0,
            use_batchnorm: bool = True,
            ):
        """
        Base class for Greedy Random Feature Boosting.
        NOTE that we currently store all intermediary classifiers/regressors,
        for simplicity. We only use the topmost one for prediction.
        """
        super(BaseGRFRBoost, self).__init__()
        self.boost_lr = boost_lr

        self.upscale = upscale # simple upscale layer, same for everyone
        self.top_level_modules = nn.ModuleList(top_level_modules) # either ridge, or multiclass logistic, or binary logistic
        self.random_feature_layers = nn.ModuleList(random_feature_layers) # random features, same for everyone
        self.ghat_boosting_layers = nn.ModuleList(ghat_boosting_layers) # functional gradient boosting layers
        if not use_batchnorm:
            self.batchnorms = nn.ModuleList([Identity() for _ in range(len(ghat_boosting_layers))])
        else:
            self.batchnorms = nn.ModuleList([nn.BatchNorm1d(ghat_boosting_layers[-1].hidden_dim,
                                                            momentum=1, affine=False,
                                                            track_running_stats=False) 
                                             for _ in range(len(ghat_boosting_layers))])
            

    def fit(self, X: Tensor, y: Tensor):
        """Fits the Random Feature Representation Boosting model.
        NOTE that in the classification case, y has to be onehot for the
        multiclass case, and (N, 1) for binary classification. For regression
        y has to be (N, d)

        Args:
            X (Tensor): Input data, shape (N, in_dim)
            y (Tensor): Targets, shape (N, d) for regression,
            or onehot (N, C) for multiclass classification, 
            or (N, 1) for binary classification.
        """
        with torch.no_grad():
            X0 = X

            # upscale
            X = self.upscale.fit_transform(X0, y)          

            # Create top level regressor or classifier W_0
            self.top_level_modules[0].fit(X, y)

            for t in range(self.n_layers):
                # Step 1: Create random feature layer
                F = self.random_feature_layers[t].fit_transform(X, X0, y)
                # Step 2: Greedily or Gradient boost to minimize R(W_t, Phi_t + Delta F)
                Ghat = self.ghat_boosting_layers[t].fit_transform(F, X, y, self.top_level_modules[t])
                X = X + self.boost_lr * Ghat
                X = self.batchnorms[t](X)
                # Step 3: Learn top level classifier W_t
                self.top_level_modules[t+1].fit(X, y, init_nnlinear=self.top_level_modules[t].linear)

        return self


    def forward(self, X: Tensor) -> Tensor:
        """Forward pass for random feature representation boosting.
        
        Args:
            X (Tensor): Input data shape (N, in_dim)"""
        with torch.no_grad():
            #upscale
            X0 = X
            X = self.upscale(X0)
            for randfeat_layer, ghat_layer, batchnorm in zip(self.random_feature_layers, 
                                                             self.ghat_boosting_layers,
                                                             self.batchnorms):
                F = randfeat_layer(X, X0)
                Ghat = ghat_layer(F)
                X = X + self.boost_lr * Ghat
                X = batchnorm(X)
            # Top level regressor
            return self.top_level_modules[-1](X)
        



############################################################################
#################    Random feature layer       #################
############################################################################

class RandomFeatureLayer(nn.Module, abc.ABC):
    @abc.abstractmethod
    def fit_transform(self, Xt: Tensor, X0: Tensor, y: Tensor) -> Tensor:
        """Takes in both Xt and X0 and y and fits the random 
        feature layer and returns the random features"""

    @abc.abstractmethod
    def forward(self, Xt: Tensor, X0: Tensor) -> Tensor:
        """Takes in both Xt and X0 and returns the random features"""



class RandFeatLayer(RandomFeatureLayer):
    def __init__(self, 
                 in_dim: int,
                 hidden_dim: int,
                 randfeat_xt_dim: int,
                 randfeat_x0_dim: int,
                 feature_type : Literal["iid", "SWIM"] = "SWIM",
                 ):
        self.hidden_dim = hidden_dim
        self.randfeat_xt_dim = randfeat_xt_dim
        self.randfeat_x0_dim = randfeat_x0_dim
        super(RandFeatLayer, self).__init__()
        
        if randfeat_xt_dim > 0:
            self.Ft = create_layer(hidden_dim, randfeat_xt_dim, feature_type)
        if randfeat_x0_dim > 0:
            self.F0 = create_layer(in_dim, randfeat_x0_dim, feature_type)


    def fit(self, Xt: Tensor, X0: Tensor, y: Tensor) -> Tensor:
        """Note that SWIM requires y to be onehot or binary"""
        if self.randfeat_xt_dim > 0:
            self.Ft.fit(Xt, y)
        if self.randfeat_x0_dim > 0:
            self.F0.fit(X0, y)
        return self


    def fit_transform(self, Xt, X0, y):
        self.fit(Xt, X0, y)
        return self.forward(Xt, X0)
    

    def forward(self, Xt: Tensor, X0: Tensor) -> Tensor:
        features = []
        if self.randfeat_xt_dim > 0:
            features.append(self.Ft(Xt))
        if self.randfeat_x0_dim > 0:
            features.append(self.F0(X0))

        return torch.cat(features, dim=1)




############################################################################
#################    Ghat layer, Gradient Boosting Regression       ########
############################################################################


class GhatGradientLayerMSE(GhatBoostingLayer):
    def __init__(self,
                 hidden_dim: int = 128,
                 l2_ghat: float = 0.01,
                 ):
        self.hidden_dim = hidden_dim
        self.l2_ghat = l2_ghat
        super(GhatGradientLayerMSE, self).__init__()
        self.ridge = RidgeModule(l2_ghat)


    def fit_transform(self, F: Tensor, Xt: Tensor, y: Tensor, auxiliary_reg: RidgeModule) -> Tensor:
        """Fits the functional gradient given features, resnet neurons, and targets,
        and returns the gradient predictions

        Args:
            F (Tensor): Features, shape (N, p)
            Xt (Tensor): ResNet neurons, shape (N, D)
            y (Tensor): Targets, shape (N, d)
            auxiliary_reg (RidgeModule): Auxiliary top level regressor.
        """
        # compute negative gradient, L_2(mu_N) normalized
        N = y.size(0)
        r = y - auxiliary_reg(Xt)
        G = r @ auxiliary_reg.W.T
        G = G / torch.norm(G) * N**0.5

        # fit to negative gradient (finding functional direction)
        Ghat = self.ridge.fit_transform(F, G)

        # line search closed form risk minimization of R(W_t, Phi_{t+1})
        self.linesearch = sandwiched_LS_scalar(r, auxiliary_reg.W, Ghat, 1e-5)
        return Ghat * self.linesearch
    

    def forward(self, F: Tensor) -> Tensor:
        return self.linesearch * self.ridge(F)
    


class GhatGreedyLayerMSE(GhatBoostingLayer):
    def __init__(self,
                 hidden_dim: int = 128,
                 l2_ghat: float = 0.01,
                 sandwich_solver: Literal["dense", "diag", "scalar"] = "dense",
                 ):
        self.hidden_dim = hidden_dim
        self.l2_ghat = l2_ghat
        self.sandwich_solver = sandwich_solver
        super(GhatGreedyLayerMSE, self).__init__()

        if sandwich_solver == "dense":
            self.sandwiched_LS = sandwiched_LS_dense
        elif sandwich_solver == "diag":
            self.sandwiched_LS = sandwiched_LS_diag
        elif sandwich_solver == "scalar":
            self.sandwiched_LS = sandwiched_LS_scalar
        else:
            raise ValueError(f"Unknown sandwich solver {sandwich_solver}")


    def fit(self, F: Tensor, Xt: Tensor, y: Tensor, auxiliary_reg: RidgeModule) -> Tensor:
        """Greedily solves the regularized sandwiched least squares problem
        argmin_Delta R(W_t, Phi_t + Delta F) for MSE loss.

        Args:
            F (Tensor): Features, shape (N, p)
            Xt (Tensor): ResNet neurons, shape (N, D)
            y (Tensor): Targets, shape (N, d)
            auxiliary_reg (RidgeModule): Auxiliary top level regressor.
        """
        # greedily minimize R(W_t, Phi_t + Delta F)
        r = y - auxiliary_reg(Xt)
        self.Delta = self.sandwiched_LS(r, auxiliary_reg.W, F, self.l2_ghat)

    
    def fit_transform(self, F: Tensor, Xt: Tensor, y: Tensor, auxiliary_reg: RidgeModule) -> Tensor:
        self.fit(F, Xt, y, auxiliary_reg)
        return self(F)
    

    def forward(self, F: Tensor) -> Tensor:
        if self.sandwich_solver == "scalar":
            return F * self.Delta
        elif self.sandwich_solver == "diag":
            return F * self.Delta[None, :]
        elif self.sandwich_solver == "dense":
            return F @ self.Delta


############################################################################
################# Random Feature Representation Boosting for Regression ###################
############################################################################





class GreedyRFRBoostRegressor(BaseGRFRBoost):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 128,
                 n_layers: int = 5,
                 randfeat_xt_dim: int = 128,
                 randfeat_x0_dim: int = 128,
                 l2_reg: float = 0.01,
                 l2_ghat: float = 0.01,
                 boost_lr: float = 1.0,
                 sandwich_solver: Literal["dense", "diag", "scalar"] = "dense",
                 feature_type : Literal["iid", "SWIM"] = "SWIM",
                 upscale_type: Literal["iid", "SWIM", "identity"] = "iid",
                 ridge_solver: Literal["iterative", "analytic"] = "analytic", # TODO not currently implemented
                 ):
        """
        Tabular Greedy Random Feaute Boosting.
        Concatenates two streams of random features [f(Xt), f(X0)].
        Assumes dense sandwich solver for Delta.

        If 'sandwich_solver' is 'diag' or 'scalar', the arguments
        randfeat_xt_dim and randfeat_x0_dim are ignored and the
        feature space is set to 'hidden_dim'.

        If 'upscale' is 'identity', the 'hidden_dim' argument is ignored.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.randfeat_xt_dim = randfeat_xt_dim
        self.randfeat_x0_dim = randfeat_x0_dim
        self.l2_reg = l2_reg
        self.l2_ghat = l2_ghat
        self.boost_lr = boost_lr
        self.feature_type = feature_type
        self.upscale_type = upscale_type

        # if no upscale, set hidden_dim to in_dim
        if upscale_type == "identity":
            self.hidden_dim = in_dim
            hidden_dim = in_dim
        upscale = Upscale(in_dim, hidden_dim, upscale_type)

        # top level regressors
        top_level_regs = [RidgeModule(l2_reg) for _ in range(n_layers+1)]

        # random feature layers
        if sandwich_solver != "dense":
            randfeat_xt_dim = hidden_dim
            randfeat_x0_dim = 0
        random_feature_layers = [
            RandFeatLayer(in_dim, hidden_dim, randfeat_xt_dim, randfeat_x0_dim, feature_type)
            for _ in range(n_layers)
        ]

        # ghat boosting layers
        ghat_boosting_layers = [
            GhatGreedyLayerMSE(hidden_dim, l2_ghat, sandwich_solver)
            for _ in range(n_layers)
        ]

        super(GreedyRFRBoostRegressor, self).__init__(
            upscale, top_level_regs, random_feature_layers, ghat_boosting_layers
        )



class GradientRFRBoostRegressor(BaseGRFRBoost):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 128,
                 n_layers: int = 5,
                 randfeat_xt_dim: int = 128,
                 randfeat_x0_dim: int = 128,
                 l2_reg: float = 0.01,
                 l2_ghat: float = 0.01,
                 boost_lr: float = 1.0,
                 feature_type : Literal["iid", "SWIM"] = "SWIM",
                 upscale_type: Literal["iid", "SWIM", "identity"] = "iid",
                 ghat_ridge_solver: Literal["iterative", "analytic"] = "analytic", #TODO not currently supported
                 ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.randfeat_xt_dim = randfeat_xt_dim
        self.randfeat_x0_dim = randfeat_x0_dim
        self.l2_reg = l2_reg
        self.l2_ghat = l2_ghat
        self.boost_lr = boost_lr
        self.feature_type = feature_type
        self.upscale_type = upscale_type

        # if no upscale, set hidden_dim to in_dim
        if upscale_type == "identity":
            self.hidden_dim = in_dim
            hidden_dim = in_dim
        upscale = Upscale(in_dim, hidden_dim, upscale_type)

        # top level regressors
        top_level_regs = [RidgeModule(l2_reg) for _ in range(n_layers+1)]

        # random feature layers
        random_feature_layers = [
            RandFeatLayer(in_dim, hidden_dim, randfeat_xt_dim, randfeat_x0_dim, feature_type)
            for _ in range(n_layers)
        ]

        # ghat boosting layers
        ghat_boosting_layers = [
            GhatGradientLayerMSE(hidden_dim, l2_ghat)
            for _ in range(n_layers)
        ]

        super(GradientRFRBoostRegressor, self).__init__(
            upscale, top_level_regs, random_feature_layers, ghat_boosting_layers
        )


############################################
############# End Regression ###############
############################################



#######################################################
############ START CLASSIFICATION #####################
#######################################################



def line_search_cross_entropy(n_classes, cls, X, y, G_hat):
    """Solves the line search risk minimizatin problem
    R(W, X + a * g) for mutliclass cross entropy loss"""
    # No onehot encoding
    if n_classes>2:
        y_labels = torch.argmax(y, dim=1)
    else:
        y_labels = y

    #loss function
    if n_classes > 2:
        loss_fn = nn.functional.cross_entropy #this is with logits
    else:
        loss_fn = nn.functional.binary_cross_entropy_with_logits

    with torch.enable_grad():
        alpha = torch.tensor([0.0], requires_grad=True, device=X.device, dtype=X.dtype)

        def closure(a):
            logits = cls(X + a * G_hat)
            loss = loss_fn(logits, y_labels)
            return loss

        result = torchmin.minimize(closure, alpha, method='newton-exact')
        #print(result)

        
        # optimizer = torch.optim.LBFGS([alpha], lr=1.0)
        # def closure():
        #     optimizer.zero_grad()
        #     logits = cls(X + alpha * G_hat)
        #     loss = loss_fn(logits, y_labels) #+ alpha**2
        #     loss.backward()
        #     return loss
        # optimizer.step(closure)



    return result.x.detach().item()



class GhatGradientLayerCrossEntropy(GhatBoostingLayer):
    def __init__(self,
                 n_classes: int,
                 hidden_dim: int,
                 l2_ghat: float,
                 ):
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.l2_ghat = l2_ghat
        super(GhatGradientLayerCrossEntropy, self).__init__()
        self.ridge = RidgeModule(l2_ghat)


    def fit_transform(self, F: Tensor, Xt: Tensor, y: Tensor, auxiliary_cls: LogisticRegression) -> Tensor:
        """Fits the functional gradient given features, resnet neurons, and targets,
        and returns the gradient predictions

        Args:
            F (Tensor): Features, shape (N, p)
            Xt (Tensor): ResNet neurons, shape (N, D)
            y (Tensor): Labels, onehot shape (N, C) or (N, 1) for binary classification
            auxiliary_reg (RidgeModule): Auxiliary top level regressor.
        """
        # compute negative gradient, L_2(mu_N) normalized
        if self.n_classes==2:
            probs = nn.functional.sigmoid(auxiliary_cls(Xt))
        else:
            probs = nn.functional.softmax(auxiliary_cls(Xt), dim=1)

        G = (y - probs) @ auxiliary_cls.linear.weight
        N = y.size(0)
        # print("G norm", torch.norm(G))
        # print("average G norm", torch.norm(G) / N)
        # print("G pre norm", G)
        #G = G / (torch.norm(G) / N**0.5).clamp(min=0.001)
        G = G / (torch.norm(G) / N).clamp(min=0.001)
        # print("G after norm", G)
        # print()

        # fit to negative gradient (finding functional direction)
        Ghat = self.ridge.fit_transform(F, G)

        # line search closed form risk minimization of R(W_t, Phi_{t+1})
        self.linesearch = line_search_cross_entropy(
            self.n_classes, auxiliary_cls, Xt, y, Ghat
            )

        #print("linesearch", self.linesearch)
        return Ghat * self.linesearch
    

    def forward(self, F: Tensor) -> Tensor:
        return self.linesearch * self.ridge(F)
    

class GradientRFRBoostClassifier(BaseGRFRBoost):
    def __init__(self,
                 in_dim: int,
                 n_classes: int,
                 hidden_dim: int = 128,
                 n_layers: int = 5,
                 randfeat_xt_dim: int = 128,
                 randfeat_x0_dim: int = 128,
                 l2_cls: float = 0.01,
                 l2_ghat: float = 0.01,
                 boost_lr: float = 1.0,
                 feature_type : Literal["iid", "SWIM"] = "SWIM",
                 upscale_type: Literal["iid", "SWIM", "identity"] = "iid",
                 ghat_ridge_solver: Literal["lbfgs", "analytic"] = "analytic", #TODO not currently supported
                 lbfgs_lr: float = 1.0,
                 lbfgs_max_iter: int = 100,
                 use_batchnorm: bool = True,
                 ):
        """TODO

        Args:
            in_dim (int): _description_
            n_classes (int): _description_
            hidden_dim (int, optional): _description_. Defaults to 128.
            n_layers (int, optional): _description_. Defaults to 5.
            randfeat_xt_dim (int, optional): _description_. Defaults to 128.
            randfeat_x0_dim (int, optional): _description_. Defaults to 128.
            l2_cls (float, optional): _description_. Defaults to 0.01.
            l2_ghat (float, optional): _description_. Defaults to 0.01.
            boost_lr (float, optional): _description_. Defaults to 1.0.
            feature_type (Literal[&quot;iid&quot;, &quot;SWIM&quot;], optional): _description_. Defaults to "SWIM".
            upscale_type (Literal[&quot;iid&quot;, &quot;SWIM&quot;, &quot;identity&quot;], optional): _description_. Defaults to "iid".
            ghat_ridge_solver (Literal[&quot;lbfgs&quot;, &quot;analytic&quot;], optional): _description_. Defaults to "analytic".
            lbfgs_max_iter (int, optional): _description_. Defaults to 100.
            use_batchnorm (bool, optional): _description_. Defaults to True.
        """
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.randfeat_xt_dim = randfeat_xt_dim
        self.randfeat_x0_dim = randfeat_x0_dim
        self.l2_cls = l2_cls
        self.l2_ghat = l2_ghat
        self.feature_type = feature_type
        self.upscale_type = upscale_type

        # if no upscale, set hidden_dim to in_dim
        if upscale_type == "identity":
            self.hidden_dim = in_dim
            hidden_dim = in_dim

        upscale = Upscale(in_dim, hidden_dim, upscale_type)

        # auxiliary classifiers
        top_level_classifiers = [LogisticRegression(n_classes, l2_cls, lbfgs_lr, lbfgs_max_iter) 
                                 for _ in range(n_layers+1)] 
        # TODO this is the one that needs a pointer to the previous class...
        # TODO i should make this work for both reg and cls, so i can implement a speedy iterative one too for reg.
        # leave for now. Either .fit takes in the previous model, or i pass a pointer in the constructor.

        # random feature layers
        random_feature_layers = [
            RandFeatLayer(in_dim, hidden_dim, randfeat_xt_dim, randfeat_x0_dim, feature_type)
            for _ in range(n_layers)
        ]

        # ghat boosting layers
        ghat_boosting_layers = [
            GhatGradientLayerCrossEntropy(n_classes, hidden_dim, l2_ghat)
            for t in range(n_layers)
        ]

        super(GradientRFRBoostClassifier, self).__init__(
            upscale, top_level_classifiers, random_feature_layers, ghat_boosting_layers, boost_lr, use_batchnorm
        )

