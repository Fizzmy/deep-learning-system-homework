import sys
from turtle import hideturtle
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os


np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION

    return nn.Sequential(nn.Residual(nn.Sequential(nn.Linear(dim,hidden_dim),norm(hidden_dim),nn.ReLU(),nn.Dropout(drop_prob),nn.Linear(hidden_dim,dim),norm(dim))),nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    L = []
    L.append(nn.Linear(dim,hidden_dim))
    L.append(nn.ReLU())
    for i in range(num_blocks):
        L.append(ResidualBlock(hidden_dim,hidden_dim//2,norm,drop_prob))
    L.append(nn.Linear(hidden_dim,num_classes))

    return nn.Sequential(*tuple(L))
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
        acc = 0.0
        sum_loss = 0.0
        batch_num = 0
        num = 0
        for i, batch in enumerate(dataloader):
            batch_num += 1
            num += batch[1].shape[0]
            batch_x, batch_y = batch[0], batch[1]
            y = model(batch_x)
            loss = nn.SoftmaxLoss()(y,batch_y)
            loss.backward()
            predicted = np.argmax(y.numpy(),axis=1)
            acc += np.sum(predicted == batch_y.numpy())
            sum_loss += loss.numpy()
        # print(num)
        return 1-acc/num,sum_loss/batch_num
    else:
        model.train()
        acc = 0.0
        sum_loss = 0.0
        batch_num = 0
        num = 0
        for i, batch in enumerate(dataloader):
            batch_num += 1
            num += batch[1].shape[0]
            batch_x, batch_y = batch[0], batch[1]
            opt.reset_grad()
            y = model(batch_x)
            loss = nn.SoftmaxLoss()(y,batch_y)
            loss.backward()
            opt.step()
            predicted = np.argmax(y.numpy(),axis=1)
            acc += np.sum(predicted == batch_y.numpy())
            sum_loss += loss.numpy()
        # print(num)
        return 1-acc/num,sum_loss/batch_num
            

    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    test_dataset = ndl.data.MNISTDataset(\
            "./data/train-images-idx3-ubyte.gz",
            "./data/train-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=True)
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    ### BEGIN YOUR SOLUTION
    for i in range(epochs):
        acc, sum_loss = epoch(test_dataloader,model,opt)
        # print(acc,sum_loss)
    test_dataset = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=False)
    t_acc, t_loss = epoch(test_dataloader,model)
    return (acc, sum_loss, t_acc, t_loss)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
