'''
2018-09-28
(c) A. Martel Lab Co.
author: G.Kuling
This is my trainer code.
WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
'''

import IUNet
import UNet
import VGG
from keras.utils import multi_gpu_model
from Utils import My_new_loss
import DGenerator as generator
import MyCBK
import TrainerFileNamesUtil as TrainerFileNamesUtil

class Trainer():
    def __init__(self,
                 batch_folder,
                 target_folder,
                 ofolder,
                 samples_per_card=None,
                 epochs=50,
                 batch_size=None,
                 gpus_used=1,
                 mode = '2Ch',
                 model_type='IUNet'):
        """
        The initializer for the trainer object
        :param batch_folder: folder where training data is stored
        :param target_folder: fodler where the training targets are stored
        :param ofolder: the output folder where everything is saved in the end
        :param samples_per_card: the desired spread of training samples to
        each gpu card
        :param epochs:The amount of epochs to be used for training
        :param batch_size: the batch size desired
        :param gpus_used: the amount of gpu cards used during training
        :param mode: Option of input type. '2Ch', 'WOFS, opr 'FS'
        :param model_type: Model type to be trained. Unet or Inception Unet
        'IUNet','UNet' or 'VGG'
        """
        self.batch_folder = batch_folder
        self.target_folder = target_folder
        self.ofolder = ofolder
        if samples_per_card is None:
            self.batch_size = batch_size

        if batch_size is None:
            self.batch_size = int(gpus_used*samples_per_card)

        if batch_size is None and samples_per_card is None:
            print('You have not given sufficient information to run multi GPU '
                  'training. Please give samples_per_card or batch_size.')

        self.epochs = epochs
        self.mode = mode
        self.model_type = model_type
        self.gpus_used = gpus_used

    def train_the_model(self,
                        t_opt = 'ADAM',
                        t_dilRate = 1,
                        t_depth = 5,
                        t_dropOut = 0
                        ):
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
        gen = generator.DGenerator(data_dir=self.batch_folder,
                                   target_dir=self.target_folder,
                                   batch_size=self.batch_size,
                                   mode=self.mode)

        # load the model
        if self.model_type is 'IUNet':
            model = IUNet.get_iunet(mode = self.mode,
                                    img_x = 512,
                                    img_y = 512,
                                    optimizer = t_opt,
                                    dilation_rate = t_dilRate,
                                    depth = t_depth,
                                    base_filter = 16,
                                    batch_normalization = False,
                                    pool_1d_size = 2,
                                    deconvolution = False,
                                    dropout = t_dropOut,
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
                                  dropout = t_dropOut,
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
    batch_folder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                   '/Batches/2D/Training/data/'
    target_folder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                    '/Batches/2D/Training/target/'
    ofolder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data/Models' \
              '/SingleClass2D/'

    a = Trainer(batch_folder, target_folder, ofolder, samples_per_card=5,
                epochs=10, gpus_used=3, mode='2Ch', model_type='UNet')

    a.train_the_model(t_opt = 'ADAM',
                      t_dilRate = 1,
                      t_depth = 5,
                      t_dropOut = 0.0)


    print('done')