from contextlib import AsyncExitStack
import numpy as np
from .autograd import Tensor
import gzip
import struct

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

import os
import pickle
from needle import backend_ndarray as nd

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img,1)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        # print(shift_x,shift_y)
        ### BEGIN YOUR SOLUTION
        shape = img.shape
        # print(shape)
        img = np.pad(img,((self.padding,self.padding),(self.padding,self.padding),(0,0)),'constant' ,constant_values = 0)
        return img[self.padding+shift_x:self.padding+shift_x+shape[0],self.padding+shift_y:self.padding+shift_y+shape[1],:]

        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device = None,
        dtype = None
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.n = 0
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.n * self.batch_size >= len(self.dataset):
            raise StopIteration
        
        idx = self.ordering[self.n]
        L = []
        for j in range(len(self.dataset[0])):
            L.append(Tensor([self.dataset[i][j] for i in idx],device=self.device))
        self.n += 1
        return tuple(L)
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename,'rb') as image:
            magic, num, rows, cols = struct.unpack('>IIII',image.read(16))
            self.images = np.frombuffer(image.read(), dtype=np.uint8).reshape(num, 784)
        with gzip.open(label_filename,'rb') as label:
            magic, num = struct.unpack('>II',label.read(8))
            self.labels = np.frombuffer(label.read(), dtype=np.uint8)

        self.images = self.images.astype(np.float32)
        self.images = (self.images - np.min(self.images)) / (np.max(self.images) - np.min(self.images))
        self.images = np.reshape(self.images,(-1,784))
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION

        if isinstance(index,slice):
            start = index.start
            stop = index.stop
            x = []
            y = []
            for i in range(start,stop):
                img = self.images[i]
                img = img.reshape((28, 28, 1))
                img = self.apply_transforms(img)
                x.append(img.reshape(-1))
                y.append(self.labels[i])
            return np.array(x),np.array(y)
        else:
            img = self.images[index]
            img = img.reshape((28, 28, 1))
            img = self.apply_transforms(img)
            return (img.reshape(-1).astype(np.float32),self.labels[index])

        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION

        if train:
            self.X=[]
            self.y=[]
            for i in range(1,6):
                batch_path=base_folder+'/data_batch_%d'%(i)
                batch_dict=unpickle(batch_path)
                train_batch=batch_dict[b'data'].astype('float32').reshape(-1,3,32,32) / 255
                train_labels=np.array(batch_dict[b'labels'])
                self.X.append(train_batch)
                self.y.append(train_labels)
            self.X = np.concatenate(self.X)
            self.y = np.concatenate(self.y)
        else:
            testpath=base_folder+'/test_batch'
            test_dict=unpickle(testpath)
            self.X=test_dict[b'data'].astype('float').reshape(-1,3,32,32) / 255
            self.y=np.array(test_dict[b'labels'])
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if isinstance(index,slice):
            start = index.start
            stop = index.stop
            x = []
            y = []
            for i in range(start,stop):
                img = self.X[i]
                img = self.apply_transforms(img)
                x.append(img.reshape(-1))
                y.append(self.y[i])
            return np.array(x),np.array(y)
        else:
            img = self.X[index]
            img = self.apply_transforms(img)
            return (img,self.y[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])





class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word in self.word2idx:
            return self.word2idx[word]
        self.word2idx[word] = len(self.idx2word)
        self.idx2word.append(word)
        return len(self.idx2word)-1
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []
        with open(path,"r") as f:
            lines = f.readlines()
            if max_lines is not None:
                lines = lines[:max_lines]
            for l in lines:
                l = l.split()
                for word in l:
                    ids.append(self.dictionary.add_word(word))
                ids.append(self.dictionary.add_word('<eos>'))
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    leng = len(data)
    leng = leng // batch_size * batch_size
    data = data[:leng]
    return np.array(data).reshape(batch_size,-1).transpose()

    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    # print(batches.shape,i,bptt)
    if i+bptt+1 > batches.shape[0]:
        data = batches[i:batches.shape[0]-1]
        target = batches[i+1:batches.shape[0]].reshape(-1)
    else:
        data = batches[i:i+bptt]
        target = batches[i+1:i+bptt+1].reshape(-1)
    return Tensor(data,device=device,dtype=dtype),Tensor(target,device=device,dtype=dtype)
    ### END YOUR SOLUTION