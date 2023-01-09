"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = []
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for x in self.params:
            x.grad.cached_data = x.grad.realize_cached_data().allreduce()
        if self.u==[]:
            for x in self.params:
                self.u.append(0 * x.grad.data)
        for i in range(len(self.u)):
            # print(self.u[i].dtype())
            self.u[i] = self.momentum * self.u[i] + ( 1 - self.momentum ) * (self.params[i].grad.data + self.weight_decay * self.params[i].data)
            self.params[i].data = self.params[i].data - self.lr * self.u[i]
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = []
        self.v = []

    def step(self):
        ### BEGIN YOUR SOLUTION
        for x in self.params:
            x.grad.cached_data = x.grad.realize_cached_data().allreduce()
        if self.m==[]:
            for x in self.params:
                self.m.append(0 * x.grad.data)
        if self.v==[]:
            for x in self.params:
                self.v.append(0 * x.grad.data)

        self.t += 1
        for i in range(len(self.params)):
            decay_weight = self.params[i].grad.data + self.weight_decay * self.params[i].data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * decay_weight
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * decay_weight * decay_weight
            m = self.m[i] / np.float32( 1 - pow(self.beta1, self.t))
            v = self.v[i] / np.float32( 1 - pow(self.beta2, self.t))
            # print(self.params[i].grad.data.dtype,self.params[i].data.dtype,decay_weight.dtype,self.m[i].dtype,m.dtype)
            self.params[i].data = self.params[i].data - self.lr * m / (pow(v,0.5) + self.eps) 
        
        ### END YOUR SOLUTION
