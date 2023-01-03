import sys
sys.path.append('./python')
import needle as ndl
import numpy as np
from needle import backend_ndarray as nd
import torch

l=2
_A = [np.random.randn(5, 5).astype(np.float32) for i in range(l)]
A = [ndl.Tensor(nd.array(_A[i]), device=ndl.cpu()) for i in range(l)]
A_t = [torch.Tensor(_A[i]) for i in range(l)]
out = ndl.stack(A, axis=1)
out_t = torch.stack(A_t, dim=1)

print(out.numpy(),out_t)