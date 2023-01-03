import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time
from mpi4py import MPI
from random import Random

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset(dataset, batch_size, world_size, device, dtype):
    bsz = batch_size // world_size
    partition_sizes = [1.0 / size for _ in range(world_size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(MPI.COMM_WORLD.Get_rank())
    print(f'partitioned dataset length: {len(partition)}')
    train_set = ndl.data.DataLoader(
        dataset=partition, batch_size=bsz, shuffle=True, device=device, dtype=dtype)
    return train_set, bsz

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
    
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    device = ndl.cuda(rank)

    if rank==0:
        # device.set_device()
        vec = device.get_id()
    else:
        vec = None
    vec = comm.bcast(vec, root=0)

    device.init_nccl(vec,rank,size)
    dataset = ndl.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=True)

    train_set, bsz = partition_dataset(
        dataset, 64 * size , size, device=device, dtype='float32')
    
    model = ResNet9(device=device, dtype="float32")
    train_cifar10(model, train_set, n_epochs=10, optimizer=ndl.optim.Adam,
         lr=0.001, weight_decay=0.001)


    correct, loss = evaluate_cifar10(model, train_set, loss_fn=nn.SoftmaxLoss())

    print(correct,loss)