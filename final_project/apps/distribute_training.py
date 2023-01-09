import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time
from mpi4py import MPI
from random import Random



def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    np.random.seed(4)
    correct, total_loss = 0, 0
    totnum = 0
    if opt is None:
        model.eval()
        for batch in dataloader:
            X, y = batch
            X,y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out = model(X)
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            loss = loss_fn(out, y)
            total_loss += loss.data.numpy() * y.shape[0]
            totnum+=y.shape[0]
    else:
        model.train()
        for batch in dataloader:
            opt.reset_grad()
            X, y = batch
            X,y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out = model(X)
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            loss = loss_fn(out, y)
            total_loss += loss.data.numpy() * y.shape[0]
            totnum+=y.shape[0]
            loss.backward()
            opt.step()
    return correct/totnum,total_loss/totnum


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss()):

    np.random.seed(4)
    opt=optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for i in range(n_epochs):
        correct, total_loss = epoch_general_cifar10(dataloader, model, loss_fn, opt)
        print(i," correct:",correct," loss:",total_loss)




def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss()):
    np.random.seed(4)
    correct, total_loss = epoch_general_cifar10(dataloader, model, loss_fn)
    
    return correct,total_loss

if __name__ == "__main__":
    
    rank, size, device = ndl.ddp.init()

    dataset = ndl.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=True)

    batch_size = 512
    train_set, bsz = ndl.ddp.partition_dataset(
        dataset, batch_size , size, device=device, dtype='float32')
    
    model = ResNet9(device=device, dtype="float32")
    ndl.ddp.broadcast_parameters(model)


    begin = time.time()
    train_cifar10(model, train_set, n_epochs=1, optimizer=ndl.optim.Adam,
         lr=0.001, weight_decay=0.001)
    end = time.time()

    print("Time:",end-begin)


    correct, loss = evaluate_cifar10(model, train_set, loss_fn=nn.SoftmaxLoss())

    print(correct,loss)