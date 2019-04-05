'''
2018-09-28
(c) A. Martel Lab Co.
author: G.Kuling
Custom made callback to save results during training.
'''

import keras
import numpy as np

class MyCBK(keras.callbacks.Callback):

    def __init__(self, model, ofolder):
        """
        Initializer for my custom made call back class
        :param model: The model that I will save during training.
        :param ofolder: The location to save files
        :param logs:
        """
        
        super(MyCBK, self).__init__()
        
        self.model_to_save = model
        self.ofolder = ofolder
        self.losses = []



    def on_epoch_end(self, epoch, logs={}):
        """
        on the end of each epoch we will evaluate the current loss. If it is
        the lowest loss, save it as best weights. If this epoch is a multiple
        of 5, save the weights as well.
        :param epoch: epoch at the moment
        :param logs: the history taken during the epoch
        :return:
        """
        if epoch % 5 == 0:
            self.model_to_save.save(self.ofolder + '/model_at_epoch_%d.h5' % epoch)
        self.losses.append(logs.get('loss'))

        min_loss = np.amin(self.losses)
        if logs.get('loss') <= min_loss:
            self.model_to_save.save(self.ofolder + '/model_best_weights_.h5')
