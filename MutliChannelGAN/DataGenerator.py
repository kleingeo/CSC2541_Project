import numpy as np
import pandas as pd


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


def data_generator(img_filelist, label_filelist, file_path, batch_size, sample_size,
                   shuffle=False, augment=False):


    total_num_samples = len(img_filelist)

    if shuffle == True:
        file_index = np.arange(0, total_num_samples)
        np.random.shuffle(file_index)

        img_filelist = img_filelist[file_index]
        label_filelist = label_filelist[file_index]

    # Sample counter
    sample_counter = 0

    while(True):

        x_batch_hold = np.zeros((batch_size,) + sample_size)
        y_batch_hold = np.zeros((batch_size,) + sample_size)

        for idx in range(batch_size):

            # Once gone through all samples, restart
            if sample_counter >= total_num_samples:
                sample_counter = 0

            x = np.load(file_path + '/' + img_filelist[sample_counter])
            y = np.load(file_path + '/' + label_filelist[sample_counter])

            if augment==True:
                pass

            x = pad_data(x, sample_size)
            y = pad_data(y, sample_size)

            x_batch_hold[idx] = x
            y_batch_hold[idx] = y


        x_batch_hold = x_batch_hold.astype(np.float32)
        y_batch_hold = y_batch_hold.astype(np.uint8)

        yield x_batch_hold, y_batch_hold




if __name__ == '__main__':

    file_path = '../prostate_data'

    df = pd.read_pickle('../build_dataframe/dataframe_slice.pickle')

    img_filelist = df['image_filename'].loc[df['train_val_test'] == 'train'].values

    label_filelist = df['label_filename'].loc[df['train_val_test'] == 'train'].values

    gen = data_generator(img_filelist, label_filelist, file_path, batch_size=5, sample_size=(256, 256),
                         shuffle=True, augment=False)

    x, y = next(gen)