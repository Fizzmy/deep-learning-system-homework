import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


def convBN(in_channels,out_channels,kernel_size,stride,device):
    return nn.Sequential(nn.Conv(in_channels, out_channels, kernel_size, stride,device=device),nn.BatchNorm2d(out_channels,device=device),nn.ReLU())

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        L = []
        L.append(convBN(3,16,7,4,device))
        L.append(convBN(16,32,3,2,device))
        L.append(nn.Residual(nn.Sequential(*(convBN(32,32,3,1,device),convBN(32,32,3,1,device)))))
        L.append(convBN(32,64,3,2,device))
        L.append(convBN(64,128,3,2,device))
        L.append(nn.Residual(nn.Sequential(*(convBN(128,128,3,1,device),convBN(128,128,3,1,device)))))
        L.append(nn.Flatten())
        L.append(nn.Linear(128,128,device=device))
        L.append(nn.ReLU())
        L.append(nn.Linear(128,10,device=device))
        self.model = nn.Sequential(*tuple(L))
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size,device=device,dtype=dtype)
        if seq_model=='rnn':
            self.nlp = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            self.nlp = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch = x.shape
        x = self.embedding(x)
        x, h = self.nlp(x,h)
        x = nn.ops.reshape(x, (seq_len * batch, self.hidden_size))
        x = self.linear(x)
        return x, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)