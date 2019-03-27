import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset

#from data.image_folder import make_dataset
from data.npy_folder import make_dataset

from PIL import Image

import numpy as np

import torchvision.transforms.functional as F


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
        x = torch.from_numpy(x.transpose((3, 0, 1, 2)).copy()).float()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Numpy2DToTensor(object):
    def __call__(self, x):
        if x.ndim == 2:
            x = np.expand_dims(x, axis=2)
        x = torch.from_numpy(x.transpose((2, 0, 1)).copy()).float()
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


class Aligned3dToImageDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        # list of number of slices for each file
        self.A_sizes = self.parse_paths(self.A_paths)
        self.B_sizes = self.parse_paths(self.B_paths)

        self.A_indices = np.cumsum(self.A_sizes)
        self.B_indices = np.cumsum(self.B_sizes)

        self.A_size = np.sum(self.A_sizes)
        self.B_size = np.sum(self.B_sizes)

        assert(self.A_size == self.B_size)

        # self.transform = get_transform(opt)
        self.transform = self.get_transform()

    def parse_paths(self, paths):
        # sizes = [np.load(path).shape[-1] for path in paths]
        sizes = [1 for path in paths]
        return sizes

    def get_transform(self):
        osize = [self.opt.loadSize, self.opt.loadSize]
        transform_list = []
        # transform_list.append(transforms.ToPILImage())
        # transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        # transform_list.append(transforms.RandomCrop(self.opt.fineSize))

#        if self.opt.isTrain and not self.opt.no_flip:
#            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5,) * self.opt.input_nc, (0.5,) * self.opt.input_nc)]
        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        """
        3D images stored in npy files - do some math to calculate
        which file and slice to load
        """
        index_A_file = np.argmax(self.A_indices > index)
        index_B_file = np.argmax(self.B_indices > index)

        A_path = self.A_paths[index_A_file]
        B_path = self.B_paths[index_B_file]

        if index_A_file > 0:
            A_slice = index - self.A_indices[index_A_file-1]
        else:
            A_slice = index

        if index_B_file > 0:
            B_slice = index - self.B_indices[index_B_file-1]
        else:
            B_slice = index

        A = np.load(A_path)
        B = np.load(B_path)

        A = A[:, :, :, A_slice]
        B = B[:, :, :, B_slice]

        A = A.transpose((1, 2, 0))
        B = B.transpose((1, 2, 0))

        osize = [self.opt.loadSize, self.opt.loadSize]
        #wx, wy = A.shape

        #A = np.expand_dims(A, 2)
        #B = np.expand_dims(B, 2)

        # pass 0 to 255 uint8 to ToPILImage()
        A = np.uint8(A * 255.0)
        B = np.uint8(B * 255.0)

        A = transforms.ToPILImage()(A)
        A = transforms.Resize(osize, Image.BICUBIC)(A)
        B = transforms.ToPILImage()(B)
        B = transforms.Resize(osize, Image.BICUBIC)(B)

        if self.opt.isTrain:
            w, h = A.size
            i = random.randint(0, w-self.opt.fineSize-1)
            j = random.randint(0, h-self.opt.fineSize-1)
        else:
            i = 0
            j = 0

        A = F.crop(A, i, j, self.opt.fineSize, self.opt.fineSize)
        B = F.crop(B, i, j, self.opt.fineSize, self.opt.fineSize)

        if self.opt.isTrain and not self.opt.no_flip:
            p = random.random()
            if p < 0.5:
                A = F.hflip(A)
                B = F.hflip(B)
            p = random.random()
            if p < 0.5:
                A = F.vflip(A)
                B = F.vflip(B)

            r = random.random() * 360 - 180
            RandomRotation = transforms.RandomRotation(degrees=(r, r), resample=Image.BICUBIC)
            A = RandomRotation(A)
            B = RandomRotation(B)

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
        # return len(self.A_paths)
        return self.A_size

    def name(self):
        return 'Aligned3dToImageDataset'
