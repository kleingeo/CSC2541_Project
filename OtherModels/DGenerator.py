"""
2018-09-28
(c) A.Martel Lab Co.
author: G.Kuling
Description: This is a data generator that will be used to process batches
that are made to train a network to segment fat tissue and fibroglandular
tissue in multi weighted breast MRI.
WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
"""

import keras
import numpy as np
import os
np.random.seed(1)


class DGenerator(keras.utils.Sequence):
    def __init__(self,
                 data_dir=r'Y:\Grey\2018-10-12-FGTSeg-Data\Batches\2D\Training'
                          r'\data\\',
                 target_dir=r'Y:\Grey\2018-10-12-FGTSeg-Data\Batches\2D'
                            r'\Training'
                            r'\target\\',
                 batch_size=50,
                 mode='2Ch',
                 img_size=[512, 512],
                 shuffle=True,
                 num_classes=3):
        """
        Initialization for data generator
        :param data_dir: (dir of numpy arrays) Directory training data is saved.
            Individual samples in each npy array.
        :param target_dir: (dir of numpy arrays) Directory training targets are
        saved. Individual samples in each npy array.
        :param batch_size: (int) Size of batches to be produced each time
            __getitem__ is called
        :param mode: (str) mode of input type. Option=['2Ch', 'WOFS', 'FS']
        :param img_size: (list of ints) the dimensions of the input
        :param shuffle: (bool) whether to shuffle training examples between
            each epoch
        :param num_classes: (int) number of classes that are labeled in the
            target masks
        """

        self.data_dir = data_dir
        self.target_dir = target_dir
        self.modality = mode
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.num_classes = num_classes

        # list all data to be used
        self.data_files = os.listdir(data_dir)
        self.target_files = os.listdir(target_dir)
        assert len(self.data_files) == len(self.target_files), \
            'Sample list and target list are different sizes.'

        self.num_channels, self.batch_chnls = decide_chnls(mode)

        self.idxs = np.arange(len(self.data_files))
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(len(self.data_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.idxs[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Find list of samples and corresponding targets
        if isinstance(self.data_files, list) and \
                isinstance(self.target_files, list):
            sample_list_temp = [self.data_files[k] for k in indexes]
            target_list_temp = [self.target_files[k] for k in indexes]

        # Generate data
        batch_x_data, batch_y_data = self.data_generation(sample_list_temp,
                                                            target_list_temp)

        ### If you need to change data types, do this here
        batch_x_data = batch_x_data.astype(np.float32)
        batch_y_data = batch_y_data.astype(np.uint8)

        return batch_x_data, batch_y_data

    def data_generation(self, sample_list, target_list):
        """
        Generates the data into batches to be passed to the trainer.
        :param sample_list: (list) list of the training examples to be imported
        :param target_list: (list) corresponding targets for the training list
        :return: a bathc of training images and a batch of target segmentations
        """
        # Initialization
        batch_x_data = np.empty((self.batch_size,
                                 self.num_channels,
                                 self.img_size[0],
                                 self.img_size[1]
                                 ))

        batch_y_data = np.empty((self.batch_size,
                                 self.num_classes,
                                 self.img_size[0],
                                 self.img_size[1]
                                 ))

        # Generate data
        for i1 in range(len(sample_list)):
            for i2 in range(len(self.batch_chnls)):
                batch_x_data[i1, i2, ...] = \
                    np.load(self.data_dir + '/' +
                            sample_list[i1])[0, self.batch_chnls[i2], ...]

            batch_y_data[i1, ...] = np.load(self.target_dir + '/' +
                                            target_list[i1])[0,...]

        return batch_x_data, batch_y_data

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        """
        if self.shuffle is True:
            np.random.shuffle(self.idxs)


if __name__ == '__main__':
    a = DGenerator()

    b = a.__getitem__(0)

    print('done')
