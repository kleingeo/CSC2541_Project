import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset

#from data.image_folder import make_dataset
from data.npy_folder import make_dataset

from PIL import Image

import numpy as np


class RandomFlip(object):
    def __init__(self, axis, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return np.flip(img, self.axis)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(axis={0}, p={1})'.format(self.axes, self.p)


class RandomCrop3D(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        wx, wy, wz = img.shape
        tx, ty, tz = output_size
        if wx == tx and wy == ty and wz == tz:
            return 0, 0, 0, wx, wy, wz
        i = random.randint(0, wx - tx)
        j = random.randint(0, wy - ty)
        k = random.randint(0, wz - tz)
        return i, j, k, tx, ty, tz

    def __call__(self, img):
        i, j, k, tx, ty, tz = self.get_params(img, self.size)
        return img[i:i+tx, j:j+ty, k:k+tz]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class Numpy3DToTensor(object):
    def __call__(self, x):
        if x.ndim == 3:
            x = np.expand_dims(x, axis=3)
        x = torch.from_numpy(x.transpose((3,0,1,2)).copy()).float()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize3D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Aligned3dDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        assert(self.A_size == self.B_size)

        # self.transform = get_transform(opt)
        self.transform = self.get_transform()

    def get_transform(self):
        transform_list = []
        # osize = [self.opt.fineSize, self.opt.fineSize, self.opt.fineSize]
        # transform_list.append(RandomCrop3D(osize))
        # if self.opt.isTrain and not self.opt.no_flip:
        #     transform_list.append(RandomFlip(0))
        #     transform_list.append(RandomFlip(1))
        #     transform_list.append(RandomFlip(2))
        transform_list += [Numpy3DToTensor(),
                           Normalize3D((0.5,) * self.opt.input_nc, (0.5,) * self.opt.input_nc)]
                    #      Normalize3D((0.5,), (0.5,))]
        return transforms.Compose(transform_list)

    def __getitem__(self, index):

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A = np.load(A_path)
        B = np.load(B_path)

        isize = [self.opt.loadSize] * 3
        osize = [self.opt.fineSize] * 3
        wc, wx, wy, wz = A.shape

        if wz < isize[2]:
            pad_width = isize[2] - wz
            pad_left = np.floor(pad_width / 2).astype('int')
            pad_right = np.ceil(pad_width / 2).astype('int')
            A = np.pad(A, ((0, 0), (0, 0), (0, 0), (pad_left, pad_right)), 'constant')
            B = np.pad(B, ((0, 0), (0, 0), (0, 0), (pad_left, pad_right)), 'constant')

        wc, wx, wy, wz = A.shape

        i = random.randint(0, wx - osize[0])
        j = random.randint(0, wy - osize[1])
        k = random.randint(0, wz - osize[2])
        A = A[:, i:i+osize[0], j:j+osize[1], k:k+osize[2]]
        B = B[:, i:i+osize[0], j:j+osize[1], k:k+osize[2]]

        A = A.transpose((1, 2, 3, 0))
        B = B.transpose((1, 2, 3, 0))

        if self.opt.isTrain and not self.opt.no_flip:
            if random.random() < 0.5:
                A = np.flip(A, 0)
                B = np.flip(B, 0)
            if random.random() < 0.5:
                A = np.flip(A, 1)
                B = np.flip(B, 1)
            #if random.random() < 0.5:
            #    A = np.flip(A, 2)
            #    B = np.flip(B, 2)

        # transform_list.append(RandomCrop3D(osize))
        # if self.opt.isTrain and not self.opt.no_flip:
        #     transform_list.append(RandomFlip(0))
        #     transform_list.append(RandomFlip(1))
        #     transform_list.append(RandomFlip(2))

        A = self.transform(A)
        B = self.transform(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'Aligned3dDataset'
