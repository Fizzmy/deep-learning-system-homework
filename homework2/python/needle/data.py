from contextlib import AsyncExitStack
import numpy as np
from .autograd import Tensor
import gzip
import struct

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


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
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
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
            L.append(Tensor([self.dataset[i][j] for i in idx]))
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

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
