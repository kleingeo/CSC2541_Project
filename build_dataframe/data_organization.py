import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
from nibabel.processing import resample_to_output


if __name__ == '__main__':

    # Initialize seed for randomization
    np.random.seed(1)

    file_path_main_images = 'D:/Geoff_Klein/Prostate_Data/Task05_Prostate/imagesTr'
    file_path_main_labels = 'D:/Geoff_Klein/Prostate_Data/Task05_Prostate/labelsTr'

    save_data_directory = '../prostate_data'

    total_slices = 602

    N_train = int(total_slices * 0.7)
    N_val = int(total_slices * 0.15)
    N_test = int(total_slices * 0.15)

    N_total = [N_train, N_val, N_test]

    num_train = 0
    num_val = 0
    num_test = 0



    df_vol = pd.DataFrame(columns=['patient_number', 'pixdim_x', 'pixdim_y', 'pixdim_z',
                               'resampled_pixdim_x', 'resampled_pixdim_y', 'resampled_pixdim_z',
                               'size_x', 'size_y', 'size_z',
                               'resampled_size_x', 'resampled_size_y', 'resampled_size_z',
                               'filename', 'image_path', 'label_path', 'train_val_test'])

    df_slice_sample = pd.DataFrame(columns=['patient_number', 'slice_number',
                                            'resampled_pixdim_x', 'resampled_pixdim_y',
                                            'resampled_size_x', 'resampled_size_y',
                                            'image_filename', 'label_filename', 'train_val_test'])


    file_list = os.listdir(file_path_main_images)

    np.random.shuffle(file_list)

    for file in file_list:

        if file.startswith('prostate'):

            file_split = file.split('.')[0].split('_')

            img = nib.load(file_path_main_images + '/' + file)

            if os.path.exists(file_path_main_labels + '/' + file) is False:
                OSError('Label missing for: ', file)

            label = nib.load(file_path_main_labels + '/' + file)

            img, img_ADC = nib.funcs.four_to_three(img)

            img_size = img.shape

            N_sample_slices = img_size[2]

            if (N_sample_slices + num_train) < N_train:
                train_val_test = 'train'
                num_train = num_train + N_sample_slices

            elif (N_sample_slices + num_val) < N_val:
                train_val_test = 'val'
                num_val = num_val + N_sample_slices

            else:
                train_val_test = 'test'
                num_test = num_test + N_sample_slices

            img_pix_size = img.header['pixdim'][1:4]

            x_resample_size = 1
            y_resampled_size = 1

            # No need to resample slice axis only using 2D slices
            img_resampled = resample_to_output(img,
                                               voxel_sizes=[x_resample_size, y_resampled_size, img_pix_size[2]])

            label_resampled = resample_to_output(label,
                                                 voxel_sizes=[x_resample_size, y_resampled_size, img_pix_size[2]],
                                                 order=0)

            img_resampeld_size = img_resampled.shape
            img_resampeld_pix_size = img_resampled.header['pixdim'][1:4]


            dict_vol = {'patient_number': file_split[-1],
                        'pixdim_x': img_pix_size[0],
                        'pixdim_y': img_pix_size[1],
                        'pixdim_z': img_pix_size[2],

                        'resampled_pixdim_x': img_resampeld_pix_size[0],
                        'resampled_pixdim_y': img_resampeld_pix_size[1],
                        'resampled_pixdim_z': img_resampeld_pix_size[2],

                        'size_x': img_size[0],
                        'size_y': img_size[1],
                        'size_z': img_size[2],

                        'resampled_size_x': img_resampeld_size[0],
                        'resampled_size_y': img_resampeld_size[1],
                        'resampled_size_z': img_resampeld_size[2],

                        'filename': file,
                        'image_path': file_path_main_images + '/' + file,
                        'labe_path': file_path_main_labels + '/' + file,

                        'train_val_test': train_val_test}

            df_vol = df_vol.append(dict_vol, ignore_index=True)

            img_data = img_resampled.get_fdata()
            label_data = label_resampled.get_fdata()

            for idx in range(img_data.shape[-1]):

                img_slice = img_data[:, :, idx]
                label_slice = label_data[:, :, idx]


                # Ignore slices of both the image and label that are only zero
                if img_slice.max() == img_slice.min():
                    continue

                if label_slice.max() == label_slice.min():
                    continue

                img_slice_filename = 'prostate_' + str(file_split[-1]) + '_slice_' + str(idx) + '.npy'
                label_slice_filename = 'label_' + str(file_split[-1]) + '_slice_' + str(idx) + '.npy'

                dict_slice = {'patient_number': file_split[-1],
                              'slice_number': idx,

                              'resampled_pixdim_x': img_resampeld_pix_size[0],
                              'resampled_pixdim_y': img_resampeld_pix_size[1],

                              'resampled_size_x': img_resampeld_size[0],
                              'resampled_size_y': img_resampeld_size[1],

                              'image_filename': img_slice_filename,
                              'label_filename': label_slice_filename,
                              'train_val_test': train_val_test}



                np.save(save_data_directory + '/' + img_slice_filename, img_slice)
                np.save(save_data_directory + '/' + label_slice_filename, label_slice)

                df_slice_sample = df_slice_sample.append(dict_slice, ignore_index=True)




    # df_vol.to_csv('dataframe_volume.csv', index=False)
    # df_slice_sample.to_csv('dataframe_slice.csv', index=False)
    #
    # df_vol.to_pickle('dataframe_volume.pickle')
    # df_slice_sample.to_pickle('dataframe_slice.pickle')

