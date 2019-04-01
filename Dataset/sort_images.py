import os
import shutil
import nibabel as nib
import pandas as pd
import numpy as np

np.random.seed(25)

def sort_images(image_path_t2_t1, image_path_t2_flair, seg_path_main, destination):

    df = pd.DataFrame(columns=['volume_name',
                               't1_filename', 't2_filename', 'flair_filename',
                               't1_file_path', 't2_file_path', 'flair_file_path',
                               'seg_filename', 'seg_file_path',
                               'slice_number', 'train_val_test',
                               'NRC_1', 'ED_2', 'ET_4'])

    for image in os.listdir(image_path_t2_t1):

        if (image.endswith('real_A.npy')) or (image.endswith('real_A0.npy')) or (image.endswith('real_B.npy')):

            continue

        image_split = image.split('_')

        volume_name = '_'.join([image_split[1],
                               image_split[2],
                               image_split[3],
                               image_split[4]])

        slice_number = image_split[6]



        seg_filename = volume_name + '_seg.nii.gz'

        seg_file_path = seg_path_main + '/' + volume_name

        seg_img = nib.load(seg_file_path + '/' + seg_filename)

        seg_data = seg_img.get_fdata()[:, :, int(slice_number)]

        seg_data_labels = np.unique(seg_data)

        NRC_1 = False
        ED_2 = False
        ET_4 = False

        if 1 in seg_data_labels:
            NRC_1 = True

        if 2 in seg_data_labels:
            ED_2 = True

        if 4 in seg_data_labels:
            ET_4 = True

        train_val_test = 'hold'


        seg_filename_new = volume_name + '_seg_' + slice_number + '.npy'

        np.save(destination + '/' + seg_filename_new, seg_data)


        t2_filename_old = image.split('fake_B.npy')[0] + 'real_A.npy'
        t2_filename_new = volume_name + '_t2_' + slice_number + '.npy'

        t1_filename_new = volume_name + '_t1_' + slice_number + '.npy'
        flair_filename_new = volume_name + '_flair_' + slice_number + '.npy'

        shutil.copy(image_path_t2_t1 + '/' + t2_filename_old, destination + '/' + t2_filename_new)

        shutil.copy(image_path_t2_t1 + '/' + image, destination + '/' + t1_filename_new)
        shutil.copy(image_path_t2_flair + '/' + image, destination + '/' + flair_filename_new)

        dict_slice = {'volume_name': volume_name,
                      'slice_number': slice_number,

                      't1_filename': t1_filename_new,
                      't2_filename': t2_filename_new,
                      'flair_filename': flair_filename_new,

                      't1_file_path': destination,
                      't2_file_path': destination,
                      'flair_file_path': destination,

                      'seg_filename': seg_filename_new,
                      'seg_file_path': destination,

                      'NRC_1': NRC_1,
                      'ED_2': ED_2,
                      'ET_4': ET_4,

                      'train_val_test': train_val_test}

        df = df.append(dict_slice, ignore_index=True)

    return df




if __name__ == '__main__':

    image_path_t2_t1 = 'D:/BRATS18/T2_Flair'

    image_path_t2_flair = 'D:/BRATS18/T2_T1'

    seg_path_main = 'D:/BRATS18/MICCAI_BraTS_2018_Data_Training/HGG'

    destination = 'D:/BRATS18/Synthetic_Images'

    df = sort_images(image_path_t2_t1, image_path_t2_flair, seg_path_main, destination)

    df.to_csv('seg_slice_dataframe_complete.csv', index=False)
    df.to_pickle('seg_slice_dataframe_complete.pickle')

    df_TC = df.loc[(df['NRC_1'] == True) &
                   (df['ET_4'] == True)]



    total_slices = df_TC.shape[0]

    N_train = int(total_slices * 0.8)
    N_val = int(total_slices * 0.2)
    # N_test = int(total_slices * 0.15)


    num_train = 0
    num_val = 0
    num_test = 0

    volume_name_list, volume_name_slices = np.unique(df_TC['volume_name'].values, return_counts=True)

    vol_idx = np.arange(len(volume_name_list))

    vol_idx_shuffled = np.copy(vol_idx)

    np.random.shuffle(vol_idx_shuffled)

    volume_name_list_shuffled = volume_name_list[vol_idx_shuffled]

    volume_name_slices_shuffled = volume_name_slices[vol_idx_shuffled]

    for idx, volume_name in enumerate(volume_name_list_shuffled):

        N_sample_slices = volume_name_slices_shuffled[idx]

        if (N_sample_slices + num_train) < N_train:
            train_val_test = 'train'
            num_train = num_train + N_sample_slices

        # elif (N_sample_slices + num_val) < N_val:
        #     train_val_test = 'val'
        #     num_val = num_val + N_sample_slices

        else:
            train_val_test = 'val'
            num_test = num_test + N_sample_slices

        df_TC['train_val_test'].loc[df_TC['volume_name'] == volume_name] = train_val_test


    df_TC.to_csv('seg_slice_dataframe_complete_TC.csv', index=False)
    df_TC.to_pickle('seg_slice_dataframe_complete_TC.pickle')