import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
from nibabel.processing import resample_to_output


if __name__ == '__main__':

    # Initialize seed for randomization
    np.random.seed(1)

    file_path_main_images = 'D:/Geoff_Klein/BRATS18/MICCAI_BraTS_2018_Data_Training/HGG'

    save_data_directory = '../brats_data'



    df_vol = pd.DataFrame(columns=['volume_name', 't2_filename', 'seg_filename',
                                   'pixdim_x', 'pixdim_y', 'pixdim_z',
                                   'size_x', 'size_y', 'size_z',
                                   'image_path', 'label_path', 'train_val_test'])

    df_slice_sample = pd.DataFrame(columns=['slice_name', 'slice_number',
                                            'pixdim_x', 'pixdim_y',
                                            'size_x', 'size_y',
                                            'image_filename', 'label_filename', 'train_val_test'])

    folder_list = os.listdir(file_path_main_images)

    for folder in folder_list:

        file_list = os.listdir(file_path_main_images + '/' + folder)

        t2_filename = [file for file in file_list if file.endswith('t2.nii.gz')][0]
        seg_filename = [file for file in file_list if file.endswith('seg.nii.gz')][0]

        t2_img = nib.load(file_path_main_images + '/' + folder + '/' + t2_filename)
        seg_img = nib.load(file_path_main_images + '/' + folder + '/' + seg_filename)

        img_size = t2_img.shape

        img_pix_size = t2_img.header['pixdim'][1:4]

        train_val_test = 'hold'

        dict_vol = {'volume_name': folder,
                    't2_filename': t2_filename,
                    'seg_filename': seg_filename,

                    'pixdim_x': img_pix_size[0],
                    'pixdim_y': img_pix_size[1],
                    'pixdim_z': img_pix_size[2],

                    'size_x': img_size[0],
                    'size_y': img_size[1],
                    'size_z': img_size[2],

                    'image_path': file_path_main_images + '/' + folder + '/' + t2_filename,
                    'label_path': file_path_main_images + '/' + folder + '/' + seg_filename,

                    'train_val_test': train_val_test}

        df_vol = df_vol.append(dict_vol, ignore_index=True)

        img_data = t2_img.get_fdata()
        label_data = seg_img.get_fdata()

        for idx in range(img_data.shape[-1]):

            img_slice = img_data[:, :, idx]
            label_slice = label_data[:, :, idx]


            # Only care about tumour core (TC), no setting ET label to 0
            label_slice[label_slice == 2] = 0

            # Ignore slices of both the image and label that are only zero
            if img_slice.max() == img_slice.min():
                continue

            if label_slice.max() == label_slice.min():
                continue

            t2_slice_name = t2_filename.split('.')
            seg_slice_name = seg_filename.split('.')

            img_slice_filename = str(t2_slice_name[0]) + '_slice_' + str(idx) + '.npy'
            label_slice_filename = str(seg_slice_name[0]) + '_slice_' + str(idx) + '.npy'

            dict_slice = {'volume_name': folder,
                          'slice_number': idx,

                          'pixdim_x': img_pix_size[0],
                          'pixdim_y': img_pix_size[1],

                          'size_x': img_size[0],
                          'size_y': img_size[1],

                          'image_filename': img_slice_filename,
                          'label_filename': label_slice_filename,
                          'train_val_test': train_val_test}



            np.save(save_data_directory + '/' + img_slice_filename, img_slice.astype(np.float32))
            np.save(save_data_directory + '/' + label_slice_filename, label_slice.astype(np.uint8))

            df_slice_sample = df_slice_sample.append(dict_slice, ignore_index=True)




    # Separate into train, validate and testing

    total_slices = df_vol['size_z'].values.sum()

    N_train = int(total_slices * 0.7)
    N_val = int(total_slices * 0.15)
    N_test = int(total_slices * 0.15)

    N_total = [N_train, N_val, N_test]

    num_train = 0
    num_val = 0
    num_test = 0

    for idx, row in df_vol.iterrows():

        N_sample_slices = row['size_z']

        volume_name = row['volume_name']

        if (N_sample_slices + num_train) < N_train:
            train_val_test = 'train'
            num_train = num_train + N_sample_slices

        elif (N_sample_slices + num_val) < N_val:
            train_val_test = 'val'
            num_val = num_val + N_sample_slices

        else:
            train_val_test = 'test'
            num_test = num_test + N_sample_slices

        df_vol.loc[idx, 'train_val_test'] = train_val_test

        slice_idx = df_slice_sample.loc[df_slice_sample['volume_name'] == volume_name].index

        df_slice_sample.loc[slice_idx, 'train_val_test'] = train_val_test





    df_vol.to_csv('dataframe_brats_volume.csv', index=False)
    df_slice_sample.to_csv('dataframe_brats_slice.csv', index=False)

    df_vol.to_pickle('dataframe_brats_volume.pickle')
    df_slice_sample.to_pickle('dataframe_brats_slice.pickle')

