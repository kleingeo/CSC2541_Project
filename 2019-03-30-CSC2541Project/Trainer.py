'''
2018-09-28
(c) A. Martel Lab Co.
author: G.Kuling
This is my trainer code.
WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
'''

import UNet
from keras.utils import multi_gpu_model
from TrainingUtils import My_new_loss
import DGenerator as generator
import DGenerator_withAugmentor as gen_wAug
import MyCBK
import TrainerFileNamesUtil as TrainerFileNamesUtil

class Trainer():
    def __init__(self,
                 batch_folder,
                 target_folder,
                 ofolder,
                 samples_per_card=None,
                 epochs=50,
                 batch_size=50,
                 gpus_used=1,
                 training_direction = True,
                 num_classes =2,
                 data_aug=False,
                 train_aug = False,
                 aug_folder=None):
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
        'IUNet' or 'UNet'
        """
        self.batch_folder = batch_folder
        self.target_folder = target_folder
        self.ofolder = ofolder
        self.direction = training_direction
        self.num_classes = num_classes
        self.data_aug = data_aug
        self.train_aug = train_aug
        self.aug_folder = aug_folder
        if samples_per_card is None:
            self.batch_size = batch_size

        if batch_size is None:
            self.batch_size = int(gpus_used*samples_per_card)

        if batch_size is None and samples_per_card is None:
            print('You have not given sufficient information to run multi GPU '
                  'training. Please give samples_per_card or batch_size.')

        self.epochs = epochs
        self.gpus_used = gpus_used

    def train_the_model(self,
                        t_opt = 'ADAM',
                        loss = My_new_loss,
                        t_depth=5,
                        t_dropout=0
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
        if self.data_aug==True:
            gen = gen_wAug.DGenerator_withAugmentor(data_dir=self.batch_folder,
                                                    target_dir=self.target_folder,
                                                    batch_size=self.batch_size,
                                                    regular = self.direction,
                                                    aug_folder=self.aug_folder)
        else:
            gen = generator.DGenerator(data_dir=self.batch_folder,
                                       target_dir=self.target_folder,
                                       batch_size=self.batch_size,
                                       regular = self.direction)

        # load the model
        if self.train_aug == True:
            model = UNet.get_unet(num_classes = 1,
                                  num_seq=2,
                                  depth=t_depth,
                                  dropout=t_dropout)
        else:
            model = UNet.get_unet()

        # setup a multi GPU trainer
        if self.gpus_used>1:
            gmodel = multi_gpu_model(model, gpus=self.gpus_used)
            gmodel.compile(optimizer=t_opt,
                           loss=loss,
                           metrics=[loss])
        else:
            gmodel = model
            gmodel.compile(optimizer=t_opt,
                           loss=loss,
                           metrics=[loss])

        # model check points, save the best weights
        cbk = MyCBK.MyCBK(model, self.ofolder)

        # begin training
        gmodel.fit_generator(gen,
                             epochs=self.epochs,
                             verbose=1,
                             steps_per_epoch=gen.__len__(),
                             workers=20,
                             use_multiprocessing=True,
                             callbacks=[cbk])

        model_json = model.to_json()
        with open(TrainerFileNamesUtil.create_model_json_filename(
                output_directory=self.ofolder,
                time_stamp=TrainerFileNamesUtil.get_time(),
                keyword='UNet_RegDir_' + str(self.direction)), 'w') as \
                jason_file:
            jason_file.write(model_json)

        print('Training is finished!')


if __name__ == "__main__":
    # local Drive
    # data_dir = r'D:\prostate_data\Task05_Prostate\imagesTr'
    #
    # target_dir = r'D:\prostate_data\Task05_Prostate\labelsTr'
    #
    # ofolder = r'X:\2019-03-30-CSC2541Project\UNet_regular\\'

    # Oberon
    data_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
               '/imagesTr/'
    target_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
                 '/labelsTr/'
    ofolder = '/home/gkuling/2019-03-30-CSC2541Project/UNet_regular/'

    a = Trainer(data_dir, target_dir, ofolder, samples_per_card=25,
                epochs=1, gpus_used=2,
                batch_size=None, training_direction=True)

    a.train_the_model(t_opt = 'ADAM')


    print('done')
