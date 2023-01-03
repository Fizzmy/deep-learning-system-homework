"""The module.
"""
from typing import List
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
        self.weight = Parameter(init.kaiming_uniform(in_features,out_features,device=device))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features,0,"relu",**{"shape":(1,out_features)},device=device))

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


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return init.ones_like(x,device=x.device) / (1.0 + ops.exp(-x))
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
        y = init.one_hot(logits.shape[leng-1],y,device=logits.device)
        return ops.summation(ops.logsumexp(logits,(leng-1,)) - ops.summation(y * logits,(leng-1,))) / logits.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(*(dim,),device=device))
        self.bias = Parameter(init.zeros(*(dim,),device=device))
        self.running_mean = init.zeros(*(dim,),device=device)
        self.running_var = init.ones(*(dim,),device=device)
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
            shape_l = list(x.shape)
            shape_l = shape_l[1:]
            shape_l.insert(0,1)
            return ops.broadcast_to(ops.reshape(self.weight,tuple(shape_l)),x.shape) * (x-ops.broadcast_to(ops.reshape(self.running_mean,tuple(shape_l)),x.shape))/((ops.broadcast_to(ops.reshape(self.running_var,tuple(shape_l)),x.shape) +self.eps)**0.5) + ops.broadcast_to(ops.reshape(self.weight,tuple(shape_l)),x.shape)
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(*(dim,),device=device))
        self.bias = Parameter(init.zeros(*(dim,),device=device))
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



class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_channels*kernel_size*kernel_size,fan_out=out_channels*kernel_size*kernel_size,shape=(kernel_size,kernel_size,in_channels,out_channels),device=device))
        if bias:
            std = 1.0/(in_channels * kernel_size**2)**0.5
            self.bias = Parameter(init.rand(*(out_channels,),low=-std,high=std,device=device))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        padding = (self.kernel_size-1) // 2
        x = ops.conv(ops.transpose(ops.transpose(x,(1,2)),(2,3)),self.weight,self.stride,padding)
        if self.bias is not None:
            shape = (1,1,1,self.out_channels)
            return ops.transpose(ops.transpose(x + ops.broadcast_to(ops.reshape(self.bias,shape),x.shape),(2,3)),(1,2))
        else:
            return ops.transpose(ops.transpose(x,(2,3)),(1,2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.
        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.
        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).
        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        std = 1 / hidden_size**0.5
        self.W_ih = Parameter(init.rand(*(input_size,hidden_size),low=-std,high=std,device=device))
        self.W_hh = Parameter(init.rand(*(hidden_size,hidden_size),low=-std,high=std,device=device))
        self.bias = bias
        if bias:
            self.bias_ih = Parameter(init.rand(*(hidden_size,),low=-std,high=std,device=device))
            self.bias_hh = Parameter(init.rand(*(hidden_size,),low=-std,high=std,device=device))
        if nonlinearity=='tanh':
            self.nonlinearity=ops.tanh
        else:
            self.nonlinearity=ops.relu
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.
        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch = X.shape[0]
        if h == None:
            h = init.zeros(*(batch,self.hidden_size),device=X.device)
        x = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            shape = (1,self.hidden_size)
            x = x + ops.broadcast_to(ops.reshape(self.bias_ih,shape),x.shape) + ops.broadcast_to(ops.reshape(self.bias_hh,shape),x.shape)
        return self.nonlinearity(x)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.
        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.
        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = []
        self.rnn_cells.append(RNNCell(input_size,hidden_size,bias,nonlinearity,device,dtype))
        for i in range(num_layers-1):
            self.rnn_cells.append(RNNCell(hidden_size,hidden_size,bias,nonlinearity,device,dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.
        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch = X.shape[0], X.shape[1]
        if h0 == None:
            h0 = init.zeros(*(self.num_layers,batch,self.hidden_size),device=X.device)
        
        x = ops.split(X,0)
        h = ops.split(h0,0)
        nh = []
        nX = []
        for j in range(self.num_layers):
            nh.append(h[j])
        for i in range(seq_len):
            nh[0] = self.rnn_cells[0](x[i],nh[0])
            for j in range(self.num_layers-1):
                nh[j+1] = self.rnn_cells[j+1](nh[j],nh[j+1])
            nX.append(nh[self.num_layers-1])
        return ops.stack(nX,0), ops.stack(nh,0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.
        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights
        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).
        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        std = 1 / hidden_size**0.5
        self.W_ih = Parameter(init.rand(*(input_size,4*hidden_size),low=-std,high=std,device=device))
        self.W_hh = Parameter(init.rand(*(hidden_size,4*hidden_size),low=-std,high=std,device=device))
        self.bias = bias
        if bias:
            self.bias_ih = Parameter(init.rand(*(4*hidden_size,),low=-std,high=std,device=device))
            self.bias_hh = Parameter(init.rand(*(4*hidden_size,),low=-std,high=std,device=device))
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.
        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch = X.shape[0]
        if h == None:
            h = init.zeros(*(batch,self.hidden_size),device=X.device),init.zeros(*(batch,self.hidden_size),device=X.device)

        x = X @ self.W_ih + h[0] @ self.W_hh
        if self.bias:
            shape = (1,4*self.hidden_size)
            x = x + ops.broadcast_to(ops.reshape(self.bias_ih,shape),x.shape) + ops.broadcast_to(ops.reshape(self.bias_hh,shape),x.shape)
        
        x = ops.split(ops.reshape(x,(batch,4,self.hidden_size)),1)

        i, f, g, o = Sigmoid()(x[0]), Sigmoid()(x[1]), Tanh()(x[2]), Sigmoid()(x[3])

        c = f * h[1] + i * g
        return o * Tanh()(c), c

        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.
        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = []
        self.lstm_cells.append(LSTMCell(input_size,hidden_size,bias,device,dtype))
        for i in range(num_layers-1):
            self.lstm_cells.append(LSTMCell(hidden_size,hidden_size,bias,device,dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.
        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch = X.shape[0], X.shape[1]
        if h == None:
            h = init.zeros(*(self.num_layers,batch,self.hidden_size),device=X.device),init.zeros(*(self.num_layers,batch,self.hidden_size),device=X.device)
        
        x = ops.split(X,0)
        h, c = ops.split(h[0],0), ops.split(h[1],0)
        nh = []
        nc = []
        nX = []
        for j in range(self.num_layers):
            nh.append(h[j])
            nc.append(c[j])
        for i in range(seq_len):
            nh[0],nc[0] = self.lstm_cells[0](x[i],(nh[0],nc[0]))
            for j in range(self.num_layers-1):
                nh[j+1],nc[j+1] = self.lstm_cells[j+1](nh[j],(nh[j+1],nc[j+1]))
            nX.append(nh[self.num_layers-1])
        return ops.stack(nX,0), (ops.stack(nh,0),ops.stack(nc,0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.
        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector
        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = init.randn(*(num_embeddings,embedding_dim),device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors
        Input:
        x of shape (seq_len, bs)
        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len,batch = x.shape
        x = init.one_hot(self.num_embeddings, x, device=x.device, dtype="float32")
        return ops.reshape((ops.reshape(x, (seq_len * batch,self.num_embeddings)) @ self.weight),(seq_len,batch,self.embedding_dim))
        ### END YOUR SOLUTION