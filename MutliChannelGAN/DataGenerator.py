import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def crop_images_centered_over_label(img, label, sample_size):

    x_sample_size = sample_size[0]
    y_sample_size = sample_size[1]

    label_idx = np.argwhere(label > 0)

    (ymin, xmin), (ymax, xmax) = label_idx.min(0), label_idx.max(0) + 1

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
    if crop_x0 + x_sample_size > img.shape[0]:

        # If crop ending index is outside the image, adjust initial index based on difference
        diff_x0 = crop_x0 - (img.shape[0] - x_sample_size)

        crop_x0 = crop_x0 - diff_x0

    # Repeat for y direction
    if crop_y0 + y_sample_size > img.shape[1]:
        diff_y0 = crop_y0 - (img.shape[1] - y_sample_size)

        crop_y0 = crop_y0 - diff_y0

    # Ensure initial crop does not have a negative index based on adjustment. If so, set to zero
    if crop_x0 < 0:
        crop_x0 = 0

    if crop_y0 < 0:
        crop_y0 = 0

    crop_x1 = crop_x0 + x_sample_size
    crop_y1 = crop_y0 + y_sample_size

    cropped_img = img[crop_x0:crop_x1, crop_y0:crop_y1]
    cropped_label = label[crop_x0:crop_x1, crop_y0:crop_y1]

    return cropped_img, cropped_label

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
    total_skipped = 0
    x_mean_list = []



    while(True):

        img_batch_hold = np.zeros((batch_size,) + sample_size)
        label_batch_hold = np.zeros((batch_size,) + sample_size)

        for idx in range(batch_size):

            # Once gone through all samples, restart
            if sample_counter >= total_num_samples:
                sample_counter = 0
                total_skipped = 0

            img = np.load(file_path + '/' + img_filelist[sample_counter])
            label = np.load(file_path + '/' + label_filelist[sample_counter])

            # plt.imshow(x[:, :])
            # plt.show()
            #
            #
            # x_mean_list.append(x.mean())

            if img.min() == img.max():
                # total_skipped = total_skipped + 1
                # print()
                # print('skipped slice ,', label_filelist[sample_counter])
                # print()
                # print('Total skipped = ', total_skipped)
                # print()
                sample_counter = sample_counter + 1
                continue

            img = (img - img.min()) / (img.max() - img.min()) * 255


            if augment==True:
                pass

            img, label = crop_images_centered_over_label(img, label, sample_size)

            img = pad_data(img, sample_size)
            label = pad_data(label, sample_size)

            img_batch_hold[idx, :, :] = img
            label_batch_hold[idx, :, :] = label

            sample_counter = sample_counter + 1


        img_batch_hold = np.expand_dims(img_batch_hold, axis=-1)
        # label_batch_hold = np.expand_dims(label_batch_hold, axis=-1)
        #
        # img_batch_hold = img_batch_hold.astype(np.float32)
        # label_batch_hold = label_batch_hold.astype(np.uint8)
        #
        # multi_channel = np.stack([img_batch_hold, label_batch_hold], axis=-1)

        yield img_batch_hold




if __name__ == '__main__':

    file_path = '../prostate_data'

    df = pd.read_pickle('../build_dataframe/dataframe_slice.pickle')

    img_filelist = df['image_filename'].loc[df['train_val_test'] == 'train'].values

    label_filelist = df['label_filename'].loc[df['train_val_test'] == 'train'].values

    gen = data_generator(img_filelist, label_filelist, file_path, batch_size=5, sample_size=(256, 256),
                         shuffle=True, augment=False)

    img = next(gen)