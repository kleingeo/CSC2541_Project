'''
2018-09-28
(c) A. Martel Lab Co.
author: G.Kuling
This is my trainer code.
WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
'''

import OtherModels.IUNet as IUNet
import OtherModels.UNet as UNet
import OtherModels.VGG as VGG
from keras.utils import multi_gpu_model
from OtherModels.Utils import My_new_loss
import OtherModels.DataGenerator as generator
import OtherModels.MyCBK as MyCBK
import OtherModels.TrainerFileNamesUtil as TrainerFileNamesUtil
import pandas as pd

class Trainer():
    def __init__(self,
                 img_filelist,
                 label_filelist,
                 file_path,
                 batch_size=5,
                 sample_size=(256, 256),
                 shuffle=True,
                 augment=False,
                 ofolder=None,
                 samples_per_card=None,
                 epochs=50,
                 gpus_used=1,
                 model_type='IUNet'):
        self.img_filelist = img_filelist
        self.label_filelist = label_filelist
        self.file_path = file_path
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.augment = augment
        self.ofolder = ofolder
        self.samples_per_card = samples_per_card
        self.epochs = epochs
        self.gpus_used = gpus_used
        self.model_type = model_type


    def train_the_model(self):
        """
        Main driver of the trainer
        :param t_opt: Optimizer option
        :param t_dilRate: dilation rate
        :param t_depth: depth
        :param t_dropOut: drop out amount
        :return:
        """
        ### Main code
        # start up the data generator
        gen = generator.data_generator(self.img_filelist,
                                       self.label_filelist,
                                       self.file_path,
                                       self.batch_size,
                                       self.sample_size,
                                       self.shuffle,
                                       self.augment)

        # load the model
        if self.model_type is 'IUNet':
            model = IUNet.get_iunet(img_x = 512,
                                    img_y = 512,
                                    optimizer = 'ADAM',
                                    dilation_rate = 1,
                                    depth = 4,
                                    base_filter = 16,
                                    batch_normalization = False,
                                    pool_1d_size = 2,
                                    deconvolution = False,
                                    dropout = 0.0,
                                    num_classes=3)
        if self.model_type is 'UNet':
            model = UNet.get_unet(mode = self.mode,
                                  img_x = 512,
                                  img_y = 512,
                                  optimizer = t_opt,
                                  dilation_rate = t_dilRate,
                                  depth = t_depth,
                                  base_filter = 16,
                                  batch_normalization = False,
                                  pool_1d_size = 2,
                                  deconvolution = False,
                                  dropout = 0.0,
                                  num_classes=3)


        if self.model_type is 'VGG':
            model = VGG.get_VGG(mode = self.mode,
                                  img_x = 256,
                                  img_y = 256,
                                  optimizer = t_opt,
                                  dilation_rate = t_dilRate,
                                  base_filter = 16,
                                  batch_normalization = False,
                                  pool_1d_size = 2,
                                  deconvolution = False,
                                  dropout = t_dropOut,
                                  num_classes=3)

        # setup a multi GPU trainer
        gmodel = multi_gpu_model(model, gpus=self.gpus_used)
        gmodel.compile(optimizer=t_opt,
                       loss=My_new_loss,
                       metrics=[My_new_loss])

        # model check points, save the best weights
        cbk = MyCBK.MyCBK(model, self.ofolder)

        # begin training
        gmodel.fit_generator(gen,
                             epochs=self.epochs,
                             verbose=1,
                             steps_per_epoch=610,
                             workers=20,
                             use_multiprocessing=True,
                             callbacks=[cbk])

        model_json = model.to_json()
        with open(TrainerFileNamesUtil.create_model_json_filename(
                output_directory=self.ofolder,
                time_stamp=TrainerFileNamesUtil.get_time(),
                keyword=self.model_type), 'w') as jason_file:
            jason_file.write(model_json)

        print('Training is finished!')


if __name__ == "__main__":
    ### controlable parameters
    # For oberon
    # batch_folder = '/jaylabs/amartel_data2/Grey/2018-06-11-FGTSeg-Data' \
    #                '/Batches/Training/data/'
    # target_folder = '/jaylabs/amartel_data2/Grey/2018-06-11-FGTSeg-Data' \
    #                '/Batches/Training/target/'
    # ofolder = '/jaylabs/amartel_data2/Grey/2018-06-11-FGTSeg-Data/Models' \
    #           '/RedoFolder/'

    # For Titania
    # batch_folder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
    #                '/Batches/2D/Training/data/'
    # target_folder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
    #                 '/Batches/2D/Training/target/'
    # ofolder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data/Models' \
    #           '/SingleClass2D/'

    file_path = '../prostate_data'

    df = pd.read_pickle('../build_dataframe/dataframe_slice.pickle')

    img_filelist = df['image_filename'].loc[df['train_val_test'] == 'train'].values

    label_filelist = df['label_filename'].loc[df['train_val_test'] == 'train'].values

    ofolder = '../OtherModels/model_output'

    a = Trainer(img_filelist,
                label_filelist,
                file_path,
                batch_size=5,
                sample_size=(256, 256),
                shuffle=True,
                augment=False,
                ofolder=ofolder,
                samples_per_card=5,
                epochs=10,
                gpus_used=1,
                model_type='UNet')

    a.train_the_model(t_opt = 'ADAM',
                      t_dilRate = 1,
                      t_depth = 5,
                      t_dropOut = 0.0)


    print('done')