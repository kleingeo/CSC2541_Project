import numpy as np
import keras
import nibabel as nib

from skimage.transform import resize

from Dataset.RandomAugmentation import RandomAugmentation


def pad_data(data, desired_shape):

    x_min_pad = 0
    x_max_pad = 0

    y_min_pad = 0
    y_max_pad = 0


    x_size, y_size = data.shape

    # Check is padding is necessary
    if x_size < desired_shape[0]:

        # Determine padding based on if shape is divisible by 2 or not
        if (desired_shape[0] - x_size) % 2 == 0:

            x_max_pad = int((desired_shape[0] - x_size) / 2)
            x_min_pad = int((desired_shape[0] - x_size) / 2)

        else:
            x_max_pad = int((desired_shape[0] - x_size) / 2) + 1
            x_min_pad = int((desired_shape[0] - x_size) / 2)


    if y_size < desired_shape[1]:

        if (desired_shape[1] - y_size) % 2 == 0:

            y_max_pad = int((desired_shape[1] - y_size) / 2)
            y_min_pad = int((desired_shape[1] - y_size) / 2)

        else:
            y_max_pad = int((desired_shape[1] - y_size) / 2) + 1
            y_min_pad = int((desired_shape[1] - y_size) / 2)


    data_padded = np.pad(data, ((x_min_pad, x_max_pad), (y_min_pad, y_max_pad)), mode='constant', constant_values=0)

    return data_padded


def crop_images_centered_over_label(t2_img, t1_img, flair_img, seg, sample_size):

    x_sample_size = sample_size[0]
    y_sample_size = sample_size[1]

    seg_idx = np.argwhere(seg > 0)

    (ymin, xmin), (ymax, xmax) = seg_idx.min(0), seg_idx.max(0) + 1

    xlength = xmax - xmin
    ylength = ymax - ymin

    # Recenter cropped images
    hx = int((x_sample_size - xlength) / 2)
    hy = int((y_sample_size - ylength) / 2)


    # Initial index for cropping
    crop_x0 = xmin - hx
    crop_y0 = ymin - hy

    # Check indicies for cropping to ensure they are within the image

    # Check to see if ending index is outside image
    if crop_x0 + x_sample_size > t2_img.shape[0]:

        # If crop ending index is outside the image, adjust initial index based on difference
        diff_x0 = crop_x0 - (t2_img.shape[0] - x_sample_size)

        crop_x0 = crop_x0 - diff_x0

    # Repeat for y direction
    if crop_y0 + y_sample_size > t2_img.shape[1]:
        diff_y0 = crop_y0 - (t2_img.shape[1] - y_sample_size)

        crop_y0 = crop_y0 - diff_y0

    # Ensure initial crop does not have a negative index based on adjustment. If so, set to zero
    if crop_x0 < 0:
        crop_x0 = 0

    if crop_y0 < 0:
        crop_y0 = 0

    crop_x1 = crop_x0 + x_sample_size
    crop_y1 = crop_y0 + y_sample_size

    t2_img_cropped = t2_img[crop_x0:crop_x1, crop_y0:crop_y1]
    seg_cropped = seg[crop_x0:crop_x1, crop_y0:crop_y1]

    if t1_img is not None:
        t1_img_cropped = t1_img[crop_x0:crop_x1, crop_y0:crop_y1]
    else:
        t1_img_cropped = None

    if flair_img is not None:
        flair_img_cropped = flair_img[crop_x0:crop_x1, crop_y0:crop_y1]
    else:
        flair_img_cropped = None

    return t2_img_cropped, t1_img_cropped, flair_img_cropped, seg_cropped


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self, t2_sample, seg_sample, t2_sample_main_path, seg_sample_main_paths, seg_slice_list,
                 batch_size=5, sample_size=(256, 256), real_or_fake='real', shuffle=False, augment_data=False,
                 t1_sample=None, flair_sample=None,
                 t1_sample_main_path=None, flair_sample_main_path=None):

        """
        Initialization
        """
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.seg_samples = seg_sample
        self.t2_samples = t2_sample

        self.shuffle = shuffle

        self.seg_slice_list = seg_slice_list

        self.t2_sample_main_path = t2_sample_main_path
        self.seg_sample_main_paths = seg_sample_main_paths
        self.augment_data = augment_data

        self.t1_samples = t1_sample
        self.flair_samples = flair_sample

        self.t1_sample_main_path = t1_sample_main_path
        self.flair_sample_main_path = flair_sample_main_path

        self.real_or_fake = real_or_fake


        self.n_channels = 1

        if (self.t1_samples is not None) and (self.real_or_fake is not None):
            self.n_channels = self.n_channels + 1

        if (self.flair_samples is not None) and (self.real_or_fake is not None):
            self.n_channels = self.n_channels + 1

        self.n_channels = 4

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.t2_samples) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of spine samples and corresponding segmentations

        if (isinstance(self.t2_samples, list)) and (isinstance(self.seg_samples, list)):

            t2_samples_temp = [self.t2_samples[k] for k in indexes]
            seg_samples_temp = [self.seg_samples[k] for k in indexes]

            seg_slice_list_temp = [self.seg_slice_list[k] for k in indexes]

            if (self.t1_samples is not None) and (self.real_or_fake is not None):
                t1_samples_temp = [self.t1_samples[k] for k in indexes]
            else:
                t1_samples_temp = [None for k in indexes]

            if (self.flair_samples is not None) and (self.real_or_fake is not None):
                flair_samples_temp = [self.flair_samples[k] for k in indexes]
            else:
                flair_samples_temp = [None for k in indexes]

        if (isinstance(self.t2_samples, np.ndarray)) and (isinstance(self.seg_samples, np.ndarray)):

            t2_samples_temp = [self.t2_samples[k] for k in indexes]
            seg_samples_temp = [self.seg_samples[k] for k in indexes]

            seg_slice_list_temp = [self.seg_slice_list[k] for k in indexes]

            if (self.t1_samples is not None) and (self.real_or_fake is not None):
                t1_samples_temp = [self.t1_samples[k] for k in indexes]
            else:
                t1_samples_temp = [None for k in indexes]

            if (self.flair_samples is not None) and (self.real_or_fake is not None):
                flair_samples_temp = [self.flair_samples[k] for k in indexes]
            else:
                flair_samples_temp = [None for k in indexes]


        if (isinstance(self.t2_samples, str)) and (isinstance(self.seg_samples, str)):

            t2_samples_temp = self.t2_samples
            seg_samples_temp = self.seg_samples
            seg_slice_list_temp = self.seg_slice_list


            if (self.t1_samples is not None) and (self.real_or_fake is not None):
                t1_samples_temp = self.t1_samples

            else:
                t1_samples_temp = None

            if (self.flair_samples is not None) and (self.real_or_fake is not None):
                flair_samples_temp = self.flair_samples
            else:
                flair_samples_temp = None



        # sample_list = np.hstack([np.array(t2_samples_temp).reshape((-1, 1)),
        #                          np.array(t1_samples_temp).reshape((-1, 1)),
        #                          np.array(flair_samples_temp).reshape((-1, 1)),
        #                          np.array(seg_samples_temp).reshape((-1, 1)),
        #                          np.array(seg_slice_list_temp).reshape((-1, 1))
        #                          ])

        sample_list = np.hstack([np.array(t2_samples_temp).reshape((-1, 1)),
                                 np.array(t1_samples_temp).reshape((-1, 1)),
                                 np.array(flair_samples_temp).reshape((-1, 1)),
                                 np.array(seg_samples_temp).reshape((-1, 1)),
                                 np.array(seg_slice_list_temp).reshape((-1, 1))
                                 ])

        # Generate data
        x_data, y_data = self.__data_generation(sample_list)

        return x_data, y_data

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.seg_samples))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sample_list):
        """
        Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels
        """
        # Initialization

        x_data_hold = np.empty((self.batch_size,) + self.sample_size + (self.n_channels,), dtype=np.float32)
        y_data_hold = np.empty((self.batch_size,) + self.sample_size + (1,), dtype=np.uint8)

        # Generate data
        for i, samples in enumerate(sample_list):

            # Store sample

            t2_sample = samples[0]
            t1_sample = samples[1]
            flair_sample = samples[2]
            seg_sample = samples[3]

            seg_slice = samples[4]


            volume_name = seg_sample.split('_seg.nii.gz')[0]
            seg_img_full = nib.load(self.seg_sample_main_paths + '/' + volume_name + '/' + seg_sample)
            seg_img = seg_img_full.get_fdata()[:, :, int(seg_slice)]

            t2_sample = volume_name + '_t2.nii.gz'
            t2_img_full = nib.load(self.seg_sample_main_paths + '/' + volume_name + '/' + t2_sample)
            t2_img = t2_img_full.get_fdata()[:, :, int(seg_slice)]

            # t2_img = np.load(self.t2_sample_main_path + '/' + t2_sample)
            #
            # if len(t2_img.shape) > 2:
            #     t2_img = t2_img[:, :, 0]

            t2_img = (t2_img - t2_img.min()) / (t2_img.max() - t2_img.min()) * 255


            # t1_sample = None
            # flair_sample = None
            t1ce_sample = True

            t1_img = None
            t1ce_img = None
            flair_img = None

            if t1_sample is not None:

                if self.real_or_fake == 'fake':

                    t1_img = np.load(self.t1_sample_main_path + '/' + t1_sample)

                elif self.real_or_fake == 'real':

                    t1_sample = volume_name + '_t1.nii.gz'
                    t1_img_full = nib.load(self.seg_sample_main_paths + '/' + volume_name + '/' + t1_sample)
                    t1_img = t1_img_full.get_fdata()[:, :, int(seg_slice)]

                if len(t1_img.shape) > 2:
                    t1_img = t1_img[:, :, 0]

                t1_img = (t1_img - t1_img.min()) / (t1_img.max() - t1_img.min()) * 255

            else:
                t1_img = None



            if flair_sample is not None:

                if self.real_or_fake == 'fake':

                    flair_img = np.load(self.flair_sample_main_path + '/' + flair_sample)

                elif self.real_or_fake == 'real':

                    flair_sample = volume_name + '_flair.nii.gz'
                    flair_img_full = nib.load(self.seg_sample_main_paths + '/' + volume_name + '/' + flair_sample)
                    flair_img = flair_img_full.get_fdata()[:, :, int(seg_slice)]

                if len(flair_img.shape) > 2:
                    flair_img = flair_img[:, :, 0]

                flair_img = (flair_img - flair_img.min()) / (flair_img.max() - flair_img.min()) * 255
            else:
                flair_img = None


            if t1ce_sample is not None:
                # t1_img = np.load(self.t1_sample_main_path + '/' + t1_sample)

                t1ce_sample = volume_name + '_t1ce.nii.gz'
                t1ce_img_full = nib.load(self.seg_sample_main_paths + '/' + volume_name + '/' + t1ce_sample)
                t1ce_img = t1ce_img_full.get_fdata()[:, :, int(seg_slice)]

                if len(t1_img.shape) > 2:
                    t1ce_img = t1ce_img[:, :, 0]

                t1ce_img = (t1ce_img - t1ce_img.min()) / (t1ce_img.max() - t1ce_img.min()) * 255
            else:
                t1ce_img = None



            # t2_img = resize(t2_img, output_shape=self.sample_size, order=3,
            #                 mode='reflect', anti_aliasing=True)
            # seg_img = resize(seg_img, output_shape=self.sample_size, order=0,
            #                  mode='reflect', anti_aliasing=True)
            #
            # if t1_sample is not None:
            #     t1_img = resize(t1_img, output_shape=self.sample_size, order=3,
            #                     mode='reflect', anti_aliasing=True)
            #
            # if flair_sample is not None:
            #     flair_img = resize(flair_img, output_shape=self.sample_size, order=3,
            #                        mode='reflect', anti_aliasing=True)

            if self.augment_data:

                rot_ang = (np.deg2rad(10))
                shear = (np.deg2rad(10))
                translate = True
                scale_factor = [0.8, 1.2]
                elastic = True


                t2_img, t1_img, t1ce_img, flair_img, seg_img = RandomAugmentation(
                    t2_img=t2_img, seg_img=seg_img,
                    t1_img=t1_img, t1ce_img=t1ce_img, flair_img=flair_img,
                    sample_size=self.sample_size,
                    rotation=rot_ang, scaling=scale_factor,
                    translation=translate, shearing=shear, elastic=elastic)



            # t2_img, t1_img, flair_img, seg_img = crop_images_centered_over_label(t2_img, t1_img, flair_img, seg_img,
            #                                                                      self.sample_size)
            #


            # Only care about tumour core (TC), no setting ET label to 0
            # seg_img[seg_img == 2] = 0
            # seg_img[seg_img == 1] = 0

            seg_img[seg_img > 1] = 1

            x_data = t2_img

            if t1_img is not None:
                x_data = np.stack([x_data, t1_img], axis=-1)

            if flair_img is not None:
                if len(x_data.shape) > 2:
                    flair_img = np.expand_dims(flair_img, axis=-1)
                    x_data = np.concatenate([x_data, flair_img], axis=-1)
                else:
                    x_data = np.stack([x_data, flair_img], axis=-1)


            if t1ce_img is not None:
                if len(x_data.shape) > 2:
                    t1ce_img = np.expand_dims(t1ce_img, axis=-1)
                    x_data = np.concatenate([x_data, t1ce_img], axis=-1)


            if (t1_img is None) and (flair_img is None):
                x_data = np.expand_dims(x_data, axis=-1)

            assert x_data.shape[2] == self.n_channels, 'Make sure the size of the input features have the correct' \
                                                       ' number of channels'

            y_data = np.expand_dims(seg_img, axis=-1)

            x_data_hold[i, ] = x_data
            y_data_hold[i, ] = y_data


        return x_data_hold, y_data_hold



if __name__ == '__main__':

    import pandas as pd

    df = pd.read_pickle('../Dataset/seg_slice_dataframe.pickle')

    t2_file_path = '/localdisk1/GeoffKlein/BRATS2018/T2_T1'
    seg_file_path = '/localdisk1/GeoffKlein/BRATS2018/MICCAI_BraTS_2018_Data_Training/HGG'
    t1_file_path = '/localdisk1/GeoffKlein/BRATS2018/T2_T1'
    flair_file_path = '/localdisk1/GeoffKlein/BRATS2018/T2_Flair'


    t2_filelist_train = df['t2_filename'].loc[df['train_val_test'] == 'train'].values
    t1_filelist_train = df['t1_filename'].loc[df['train_val_test'] == 'train'].values
    flair_filelist_train = df['flair_filename'].loc[df['train_val_test'] == 'train'].values

    t2_filelist_val = df['t2_filename'].loc[df['train_val_test'] == 'val'].values
    t1_filelist_val = df['t1_filename'].loc[df['train_val_test'] == 'val'].values
    flair_filelist_val = df['flair_filename'].loc[df['train_val_test'] == 'val'].values

    seg_filelist_train = df['seg_filename'].loc[df['train_val_test'] == 'train'].values
    seg_slice_train = df['slice_number'].loc[df['train_val_test'] == 'train'].values

    seg_filelist_val = df['seg_filename'].loc[df['train_val_test'] == 'val'].values
    seg_slice_val = df['slice_number'].loc[df['train_val_test'] == 'val'].values



    params_train_generator = {'sample_size': (256, 256),
                              'batch_size': 45,
                              'n_channels': 3,
                              'shuffle': False,
                              'augment_data': False}


    training_generator = DataGenerator(
        t2_sample=t2_filelist_train,
        seg_sample=seg_filelist_train,
        t2_sample_main_path=t2_file_path,
        seg_sample_main_paths=seg_file_path,
        seg_slice_list=seg_slice_train,
        t1_sample=t1_filelist_train,
        flair_sample=flair_filelist_train,
        t1_sample_main_path=t1_file_path,
        flair_sample_main_path=flair_file_path,
        **params_train_generator)


    training_generator.__getitem__(0)