"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** numpy.float32(self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return self.scalar * (a**(self.scalar-1)) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return ( out_grad / b, -a / (b * b) * out_grad)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / numpy.float32(self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        
        dim = len(a.shape)
        new_axis = list(range(dim))
        if self.axes:
            new_axis[self.axes[0]],new_axis[self.axes[1]] = new_axis[self.axes[1]],new_axis[self.axes[0]]
        else:
            new_axis[-2],new_axis[-1] = new_axis[-1], new_axis[-2]
        
        return a.permute(tuple(new_axis)) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad,self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return reshape(out_grad,a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        dim_a = len(a.shape)
        dim_out = len(self.shape)
        out_axes = []
        out_axes.extend([x for x in range(dim_out-dim_a)])
        out_axes.extend([x+dim_out-dim_a for x in range(dim_a) if  a.shape[x]==1])
        return reshape(summation(out_grad,tuple(out_axes)),a.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return array_api.summation(a)
        if type(self.axes) is int:
            self.axes = self.axes,
        for x in range(len(self.axes)-1,-1,-1):
            a = array_api.summation(a,self.axes[x])
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        out_shape = list(a.shape)
        if self.axes==None:
            for i in range(len(out_shape)):
                out_shape[i]=1
        else:
            if type(self.axes) is int:
                self.axes = self.axes,
            for x in self.axes:
                out_shape[x]=1
        out_shape=tuple(out_shape)
        return broadcast_to(reshape(out_grad,out_shape),a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a , b = node.inputs
        dim_a = len(a.shape)
        dim_b = len(b.shape)
        if dim_a == dim_b:
            return (matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad))
        else:
            if dim_a > dim_b:
                axes = tuple(range(dim_a-dim_b))
                return (matmul(out_grad, transpose(b)), summation(matmul(transpose(a), out_grad),axes))
            else:
                axes = tuple(range(dim_b-dim_a))
                return (summation(matmul(out_grad, transpose(b)),axes), matmul(transpose(a), out_grad))
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        mask = Tensor(a > 0 , device=a.device, dtype=out_grad.dtype)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        self.softmax = None

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxZ = Z.max(axis = self.axes, keepdims=True)
        maxZ_dim = Z.max(axis = self.axes)
        Z_fix = Z - maxZ.broadcast_to(Z.shape)
        self.softmax = Tensor(Z_fix,device=Z.device)
        return array_api.log(array_api.summation(array_api.exp(Z_fix),axis=self.axes))+maxZ_dim
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        out_shape = list(a.shape)
        if self.axes==None:
            for i in range(len(out_shape)):
                out_shape[i]=1
        else:
            if type(self.axes) is int:
                self.axes = self.axes,
            for x in self.axes:
                out_shape[x]=1
        out_shape=tuple(out_shape)

        exp_x = exp(self.softmax)
        return exp_x / broadcast_to(reshape(summation(exp_x,self.axes) / out_grad, out_shape), a.shape)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - power_scalar(node,2))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        num = len(args)
        new_shape = [num]
        pre_shape = args[0].shape
        tot_size = 1
        for x in pre_shape:
            tot_size *= x
        new_shape.append(tot_size)
        a = array_api.empty(tuple(new_shape),device = args[0].device)

        for i in range(num):
            a[i,:] =  args[i].reshape(new_shape = (tot_size,))
        
        new_shape = [num]
        new_shape.extend(list(pre_shape))
        a = a.reshape(tuple(new_shape))

        permute_axis = list(range(len(new_shape)))
        for i in range(self.axis):
            permute_axis[i],permute_axis[i+1] = permute_axis[i+1],permute_axis[i]
        
        # print(permute_axis,new_shape)
        return a.permute(tuple(permute_axis)).compact()

        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        
        new_shape = list(A.shape)
        num = new_shape[self.axis]
        ans = []
        
        permute_axis = list(range(len(new_shape)))
        del new_shape[self.axis]
        # print(new_shape)

        
        for i in range(self.axis):
            permute_axis[i],permute_axis[i+1] = permute_axis[i+1],permute_axis[i]

        tot_size = 1
        for x in new_shape:
            tot_size *= x

        _A = A.permute(tuple(permute_axis)).compact().reshape((num,tot_size))

        for i in range(num):
            a = array_api.empty(shape = (tot_size,),device = A.device)
            a[:] = _A[i,:]
            ans.append(a.reshape(new_shape))

        return ans

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = a.shape
        shape = list(shape)
        for i in self.axes:
            if i >= len(shape):
                continue
            shape[i] += self.dilation * shape[i]
        ans = NDArray.make(tuple(shape),device = a.device)
        ans.fill(0)
        idx = []
        for i in shape:
            idx.append(slice(0,i,1))
        for i in self.axes:
            if i >= len(shape):
                continue
            idx[i] = slice(0,shape[i],self.dilation+1)
        ans[tuple(idx)] = a
        return ans
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = a.shape
        shape = list(shape)
        for i in self.axes:
            if i >= len(shape):
                continue
            shape[i] //= self.dilation + 1
        ans = NDArray.make(tuple(shape),device = a.device)
        ans.fill(0)
        idx = []
        for i in a.shape:
            idx.append(slice(0,i,1))
        for i in self.axes:
            if i >= len(shape):
                continue
            idx[i] = slice(0,a.shape[i],self.dilation+1)
        ans = a[tuple(idx)]
        return ans
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.compact()
        B = B.compact()
        A = A.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        N,H,W,C = A.shape
        K,_,_,O = B.shape
        nH,nW = (H-K+1)//self.stride, (W-K+1)//self.stride
        _A = NDArray.make(shape=(N,nH,nW,K,K,C),strides = (H * W * C,self.stride * W * C,self.stride * C,W * C,C,1),handle=A._handle,device = A.device).compact().reshape((N*nH*nW,K*K*C))
        _B = B.reshape((K*K*C,O))
        return ( _A @ _B ).reshape((N,nH,nW,O))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X , W = node.inputs
        K = W.shape[0]
        grad = dilate(out_grad,(1,2),self.stride-1)
        # rH, rW = X.shape[1] - grad.shape[1], X.shape[2] - grad.shape[2]
        W_flip = flip(transpose(W,(2,3)),(0,1))

        X_t = transpose(X,(0,3))
        grad_t = transpose(transpose(grad,(0,1)),(1,2))

        return conv(grad,W_flip,1,K-self.padding-1), transpose(transpose(conv(X_t,grad_t,1,self.padding),(0,1)),(1,2))
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)