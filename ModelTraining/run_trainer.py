import sys

sys.path.append('..')



from ModelTraining.Trainer import Trainer
import pandas as pd


if __name__ == '__main__':



    # df = pd.read_pickle('../Dataset/seg_slice_dataframe.pickle')
    df = pd.read_pickle('../Dataset/seg_slice_dataframe_shuffled_TC.pickle')



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

    # df_shape = df.shape
    #
    # df_train = int(df_shape[0] * 0.7)
    # df = df.sample(frac=1)
    # t2_filelist_train = df['t2_filename'].iloc[:df_train].values
    # t1_filelist_train = df['t1_filename'].iloc[:df_train].values
    # flair_filelist_train = df['flair_filename'].iloc[:df_train].values
    #
    # t2_filelist_val = df['t2_filename'].iloc[df_train:].values
    # t1_filelist_val = df['t1_filename'].iloc[df_train:].values
    # flair_filelist_val = df['flair_filename'].iloc[df_train:].values
    #
    # seg_filelist_train = df['seg_filename'].iloc[:df_train].values
    # seg_slice_train = df['slice_number'].iloc[:df_train].values
    #
    # seg_filelist_val = df['seg_filename'].iloc[df_train:].values
    # seg_slice_val = df['slice_number'].iloc[df_train:].values


    # params_dictionary = dict(model_type=['UNet', 'IUNet', 'VNet', 'VGG', 'ResNet'],
    #                          Epochs=[100],
    #                          batch_size=[45],
    #                          augment_training=[True],
    #                          train_fraction=[1, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2],
    #                          with_fake=[True, False])


    params_dictionary = dict(model_type=['UNet'],
                             Epochs=[100],
                             batch_size=[15],
                             augment_training=[False],
                             real_or_fake=['real'],
                             train_fraction=[1],
                             )

    trainer = Trainer(output_directory='../TrainOutput_test',

                      t2_img_filelist_train=t2_filelist_train,
                      seg_filelist_train=seg_filelist_train,
                      seg_slice_train=seg_slice_train,

                      t2_img_filelist_val=t2_filelist_val,
                      seg_filelist_val=seg_filelist_val,
                      seg_slice_val=seg_slice_val,

                      t2_file_path=t2_file_path,
                      seg_file_path=seg_file_path,

                      t1_img_filelist_train=t1_filelist_train,
                      flair_img_filelist_train=flair_filelist_train,

                      t1_img_filelist_val=t1_filelist_val,
                      flair_img_filelist_val=flair_filelist_val,

                      t1_file_path=t1_file_path,
                      flair_file_path=flair_file_path,

                      sample_size=(240, 240),
                      trainer_grid_search=params_dictionary,
                      multi_gpu=True,
                      relative_save_weight_peroid=5)


    trainer.train()

