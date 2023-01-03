import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
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
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss()):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt=optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for i in range(n_epochs):
        correct, total_loss = epoch_general_cifar10(dataloader, model, loss_fn, opt)
        print(i," correct:",correct," loss:",total_loss)
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss()):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    correct, total_loss = epoch_general_cifar10(dataloader, model, loss_fn)
    
    return correct,total_loss
    # print("correct:",correct/totnum," loss:",total_loss/totnum)
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    nbatch, batch_size = data.shape
    tot_num = 0
    correct, total_loss = 0, 0
    h = None
    for i in range(0, nbatch-1, seq_len):
        x, target = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        tot_num += target.shape[0]
        x, h = model(x, h)
        if isinstance(h, tuple):
            h = h[0].detach(), h[1].detach()
        else:
            h = h.detach()
        correct += np.sum(np.argmax(x.numpy(), axis=1) == target.numpy())
        loss = loss_fn(x, target)
        total_loss += loss.data.numpy() * target.shape[0]
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

    return correct/tot_num,total_loss/tot_num
            


    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss(), clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt=optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for i in range(n_epochs):
        correct, total_loss = epoch_general_ptb(data, model, seq_len, loss_fn, opt, clip, device, dtype)
        print(i," correct:",correct," loss:",total_loss)
    return correct, total_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss(),
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.eval()
    correct, total_loss = epoch_general_ptb(data, model, seq_len, loss_fn, None, None, device, dtype)
    
    return correct,total_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
