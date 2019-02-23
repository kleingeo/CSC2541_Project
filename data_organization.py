import pandas as pd
import numpy as np
import os
import nibabel as nib
from nibabel.processing import resample_to_output
from joblib import Parallel, delayed

def build_df(df, file, idx, file_path_main_images, file_path_main_labels):
    if file.startswith('prostate'):

        file_split = file.split('.')[0].split('_')

        img = nib.load(file_path_main_images + '/' + file)

        img_size = img.shape

        img_pix_size = img.header['pixdim'][1:4]

        if os.path.exists(file_path_main_labels + '/' + file) is False:
            OSError('Label missing for: ', file)

        # pixdim_x.append(img_pix_size[0])
        # pixdim_y.append(img_pix_size[1])
        # pixdim_z.append(img_pix_size[2])

        img_resampled = resample_to_output(img, voxel_sizes=[0.802734, 0.802734, 2.5])

        img_resampeld_size = img_resampled.shape

        img_resampeld_pix_size = img_resampled.header['pixdim'][1:4]

        dict_hold = {'patient_number': file_split[-1],
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
                     'labe_path': file_path_main_labels + '/' + file}

        df = df.append(dict_hold, ignore_index=True)
        print()
        print(idx)

    return df


if __name__ == '__main__':

    file_path_main_images = 'D:/Geoff_Klein/Prostate_Data/Task05_Prostate/imagesTr'
    file_path_main_labels = 'D:/Geoff_Klein/Prostate_Data/Task05_Prostate/labelsTr'

    df = pd.DataFrame(columns=['patient_number', 'pixdim_x', 'pixdim_y', 'pixdim_z',
                               'resampled_pixdim_x', 'resampled_pixdim_y', 'resampled_pixdim_z',
                               'size_x', 'size_y', 'size_z',
                               'resampled_size_x', 'resampled_size_y', 'resampled_size_z',
                               'filename', 'image_path', 'label_path'])

    pixdim_x = []
    pixdim_y = []
    pixdim_z = []


    resampled_x_size = []
    resampled_y_size = []
    resampled_z_size = []

    for idx, file in enumerate(os.listdir(file_path_main_images)):

        if file.startswith('prostate'):

            file_split = file.split('.')[0].split('_')


            img = nib.load(file_path_main_images + '/' + file)

            img, img_ADC = nib.funcs.four_to_three(img)

            img_size = img.shape

            img_pix_size = img.header['pixdim'][1:4]

            if os.path.exists(file_path_main_labels + '/' + file) is False:

                OSError('Label missing for: ', file)


            pixdim_x.append(img_pix_size[0])
            pixdim_y.append(img_pix_size[1])
            pixdim_z.append(img_pix_size[2])



            # img_resampled = resample_to_output(img, voxel_sizes=[0.802734, 0.802734, 2.5])

            img_resampled = resample_to_output(img, voxel_sizes=[0.6, 0.6, 3.6])

            img_resampeld_size = img_resampled.shape

            img_resampeld_pix_size = img_resampled.header['pixdim'][1:4]

            resampled_x_size.append(img_resampeld_size[0])
            resampled_y_size.append(img_resampeld_size[1])
            resampled_z_size.append(img_resampeld_size[2])

            dict_hold = {'patient_number': file_split[-1],
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
                         'labe_path': file_path_main_labels + '/' + file}

            df = df.append(dict_hold, ignore_index=True)
            print()
            print(idx)


    df.to_csv('dataframe.csv', index=False)

    pixdim_x = np.array(pixdim_x)
    pixdim_y = np.array(pixdim_y)
    pixdim_z = np.array(pixdim_z)

    resampled_x_size = np.array(resampled_x_size)
    resampled_y_size = np.array(resampled_y_size)
    resampled_z_size = np.array(resampled_z_size)