"""The module.
"""
from turtle import shape, window_height
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features,out_features))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features,0,"relu",**{"shape":(1,out_features)}))

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = ops.matmul(X , self.weight)
        if self.bias!=None:
            x_shape = X.shape
            weight_shape = self.weight.shape
            output = output + ops.broadcast_to(self.bias,(x_shape[0],weight_shape[1]))
        return output
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        size = 1
        for x in X.shape[1:]:
            size *= x
        return ops.reshape(X,(X.shape[0],size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = ops.relu(x)
        return output
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        leng = len(logits.shape)
        y = init.one_hot(logits.shape[leng-1],y)
        return ops.summation(ops.logsumexp(logits,(leng-1,)) - ops.summation(y * logits,(leng-1,))) / logits.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(*(dim,)))
        self.bias = Parameter(init.zeros(*(dim,)))
        self.running_mean = init.zeros(*(dim,))
        self.running_var = init.ones(*(dim,))
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # leng = len(x.shape)
        if self.training:
            shape_l = list(x.shape)
            shape_l = shape_l[1:]
            shape_l.insert(0,1)
            E_x = ops.summation(x,axes=(0,)) / x.shape[0]
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * E_x.data
            E_x = ops.broadcast_to(ops.reshape(E_x,tuple(shape_l)),x.shape)
            Var_x = ops.summation((x - E_x )**2,axes=(0,)) / x.shape[0]
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * Var_x.data
            Var_x = ops.broadcast_to(ops.reshape(Var_x,tuple(shape_l)),x.shape)
            return ops.broadcast_to(ops.reshape(self.weight,tuple(shape_l)),x.shape) * (x-E_x)/((Var_x +self.eps)**0.5) + ops.broadcast_to(ops.reshape(self.bias,tuple(shape_l)),x.shape)
        else:
            return self.weight * (x-self.running_mean)/((self.running_var +self.eps)**0.5) + self.bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(*(dim,)))
        self.bias = Parameter(init.zeros(*(dim,)))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        leng = len(x.shape)
        shape_l = list(x.shape)
        shape_l = shape_l[:-1]
        shape_l.append(1)
        E_x = ops.summation(x,axes=(leng-1,)) / self.dim
        E_x = ops.broadcast_to(ops.reshape(E_x,tuple(shape_l)),x.shape)
        Var_x = ops.summation((x - E_x )**2,axes=(leng-1,)) / self.dim
        Var_x = ops.broadcast_to(ops.reshape(Var_x,tuple(shape_l)),x.shape)

        shape_w = list(x.shape)
        for i in range(len(x.shape)-1):
            shape_w[i] = 1
        return ops.broadcast_to(ops.reshape(self.weight,tuple(shape_w)),x.shape) * (x-E_x)/((Var_x +self.eps)**0.5) + ops.broadcast_to(ops.reshape(self.bias,tuple(shape_w)),x.shape)

        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*(x.shape),p = 1 - self.p)
            return mask * x / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



