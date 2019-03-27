import os.path
from data.base_dataset import BaseDataset, get_transform

#from data.image_folder import make_dataset
from data.npy_folder import make_dataset

from PIL import Image
import random

import numpy as np
import torchvision.transforms as transforms


class Unaligned2dToImageDataset(BaseDataset):
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

        self.transform = self.get_transform()

    def parse_paths(self, paths):
        sizes = [1 for path in paths]
        return sizes

    def get_transform(self):
        osize = [self.opt.loadSize, self.opt.loadSize]
        transform_list = []
        transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))

        if self.opt.isTrain and not self.opt.no_flip:
            transform_list.append(transforms.RandomVerticalFlip())
            #transform_list.append(transforms.RandomHorizontalFlip())
            #transform_list.append(transforms.RandomRotation(degrees=180))

#            transform_list.append(transforms.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10))

        if self.opt.isTrain:
            transform_list.append(transforms.RandomCrop(self.opt.fineSize))
        else:
            transform_list.append(transforms.CenterCrop(self.opt.fineSize))

        transform_list += [transforms.ToTensor()]

#                transforms.Normalize((0.5,) * self.opt.input_nc, (0.5,) * self.opt.input_nc)]

        return transforms.Compose(transform_list)

    def __getitem__(self, index):

        """
        3D images stored in npy files - do some math to calculate
        which file and slice to load
        """
        index_A = index % self.A_size
        index_B = index % self.B_size

        index_A_file = np.argmax(self.A_indices > index_A)
        index_B_file = np.argmax(self.B_indices > index_B)

        A_path = self.A_paths[index_A_file]
        B_path = self.B_paths[index_B_file]

        if index_A_file > 0:
            A_slice = index_A - self.A_indices[index_A_file-1]
        else:
            A_slice = index_A

        if index_B_file > 0:
            B_slice = index_B - self.B_indices[index_B_file-1]
        else:
            B_slice = index_B

        A = np.load(A_path)
        B = np.load(B_path)

        A = A[:, :, :, A_slice]
        B = B[:, :, :, B_slice]

        A = A.transpose((1, 2, 0))
        B = B.transpose((1, 2, 0))

        # pass 0 to 255 uint8 to ToPILImage()
        A = np.uint8(A * 255.0)
        B = np.uint8(B * 255.0)

        if A.shape[2] == 1:
            A = np.tile(A, (1, 1, self.opt.input_nc))
        if B.shape[2] == 1:
            B = np.tile(B, (1, 1, self.opt.output_nc))

        A = self.transform(A)
        B = self.transform(B)

        A_nc = A.shape[0]
        B_nc = B.shape[0]
        norm_A = transforms.Normalize((0.5,) * A_nc, (0.5,) * A_nc)
        norm_B = transforms.Normalize((0.5,) * B_nc, (0.5,) * B_nc)

#        norm_A = transforms.Normalize((0.5,) * self.opt.input_nc, (0.5,) * self.opt.input_nc)
#        norm_B = transforms.Normalize((0.5,) * self.opt.output_nc, (0.5,) * self.opt.output_nc)
        A = norm_A(A)
        B = norm_B(B)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'Unaligned2dToImageDataset'
