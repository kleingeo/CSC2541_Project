import pandas as pd
import numpy as np
import os
import nibabel as nib

def build_dataframe(image_path_t2_t1, image_path_t2_flair, seg_path_main):

    df = pd.DataFrame(columns=['volume_name',
                               't1_filename', 't2_filename', 'flair_filename',
                               't1_file_path', 't2_file_path', 'flair_file_path',
                               'seg_filename', 'seg_file_path',
                               'slice_number', 'train_val_test'])



    for image in os.listdir(image_path_t2_t1):

        if (image.endswith('real_A.npy')) or (image.endswith('real_A0.npy')) or (image.endswith('real_B.npy')):

            continue

        split_img_name = image.split('_')

        volume_name = split_img_name[1] + '_' + split_img_name[2] + '_' + split_img_name[3] + '_' + split_img_name[4]

        slice_number = int(split_img_name[-3])

        seg_filename = volume_name + '_seg.nii.gz'

        seg_file_path = seg_path_main + '/' + volume_name

        seg_img = nib.load(seg_file_path + '/' + seg_filename)

        seg_data = seg_img.get_fdata()[:, :, slice_number]

        seg_data[seg_data == 2] = 0

        # Ignore slices of both the image and label that are only zero
        if seg_data.max() == seg_data.min():
            continue

        train_val_test = 'hold'

        t2_filename = image.split('fake_B.npy')[0] + 'real_A.npy'
        t2_file_path = image_path_t2_t1

        flair_filename = image
        flair_file_path = image_path_t2_flair

        t1_filename = image
        t1_file_path = image_path_t2_t1

        dict_slice = {'volume_name': volume_name,
                      'slice_number': slice_number,

                      't1_filename': t1_filename,
                      't2_filename': t2_filename,
                      'flair_filename': flair_filename,

                      't1_file_path': t1_file_path,
                      't2_file_path': t2_file_path,
                      'flair_file_path': flair_file_path,

                      'seg_filename': seg_filename,
                      'seg_file_path': seg_file_path,

                      'train_val_test': train_val_test}

        df.append(dict_slice, ignore_index=True)

    total_slices = df.shape[0]

    N_train = int(total_slices * 0.7)
    N_val = int(total_slices * 0.15)
    N_test = int(total_slices * 0.15)


    num_train = 0
    num_val = 0
    num_test = 0

    volume_name_list, volume_name_slices = np.unique(df['volume_name'].values, return_counts=True)

    for idx, volume_name in enumerate(volume_name_list):

        N_sample_slices = volume_name_slices[idx]

        if (N_sample_slices + num_train) < N_train:
            train_val_test = 'train'
            num_train = num_train + N_sample_slices

        elif (N_sample_slices + num_val) < N_val:
            train_val_test = 'val'
            num_val = num_val + N_sample_slices

        else:
            train_val_test = 'test'
            num_test = num_test + N_sample_slices

        df['train_val_test'].loc[df['volume_name'] == volume_name] = train_val_test


    return df


if __name__ == '__main__':

    image_path_t2_t1 = 'D:/Geoff_Klein/BRATS18/T2_Flair'

    image_path_t2_flair = 'D:/Geoff_Klein/BRATS18/T2_T1'

    seg_path_main = 'D:/Geoff_Klein/BRATS18/MICCAI_BraTS_2018_Data_Training/HGG'

    df = build_dataframe(image_path_t2_t1, image_path_t2_flair, seg_path_main)

    df.to_csv('seg_slice_dataframe.csv', index=False)
    df.to_pickle('seg_slice_dataframe.pickle')