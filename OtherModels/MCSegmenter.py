'''
2019-01-18
(c) A. Martel Lab Co.
author: G.Kuling
This is a segmentor that uses Monte Carlo Dropout during testing to return
uncertainty measurements in its segmentation.
'''

import numpy as np
import SimpleITK as sitk
import keras
import keras.backend as K
import os
from Utils import My_new_loss

def normalize(test):
    upper_lim = 99.9
    top = np.percentile(test, upper_lim)
    test[np.where(test > top)] = top
    test = (test * 255. / top).astype(np.uint8)
    return test

class MCSegmenter:
    def __init__(self,
                 scan_local,
                 model_dir,
                 ofolder):
        '''
        This is the initiator for the MCSegmenter
        :param scan_local: (list of str) location of the test subject in
        question. Give a list of locations for WOFS and FS T1w scans.
        :aram model_dir: (str) directory of the model weights and json file
        used to load  the model.
        :param ofolder: (str) outfolder.
        '''
        self.WOFS = sitk.ReadImage(scan_local[0])
        self.FS = sitk.ReadImage(scan_local[1])

        json_file_name = \
            [i for i in os.listdir(model_dir) if i.endswith('json')][0]
        weights_file_name = \
            [i for i in os.listdir(model_dir) if i.startswith(
                'model_best')][0]
        json_file = open(''.join([model_dir, '/', json_file_name]))
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = keras.models.model_from_json(loaded_model_json)

        # Load the weights to the model
        self.model.load_weights(''.join([model_dir, '/', weights_file_name]))
        self.model.compile(loss=My_new_loss, metrics=[My_new_loss],
                           optimizer='ADAM')


        for layer in self.model.layers:
            if hasattr(layer, 'rate'):
                layer.rate = 0.5
        self.model.compile(loss=My_new_loss, metrics=[My_new_loss],
                           optimizer='ADAM')

        self.f = K.function([self.model.layers[0].input, K.learning_phase()],
                            [self.model.layers[-1].output])
        print('Model is ready to predict.')
        self.pt_num = '_'.join(scan_local[0].split('\\')[-1].split('_')[0:-1])

        self.ofolder = ofolder

    def perform_segmentation(self):
        '''
        Performs the MC segmentation of the test subject.
        :return: segmentation of tissues, uncertainty of each segmentation map.
        '''
        npfs = sitk.GetArrayFromImage(self.FS)
        npwofs = sitk.GetArrayFromImage(self.WOFS)
        seg = np.empty(npfs.shape)
        uncert = np.empty((npfs.shape[0],3,512,512))
        print('Beginning Segmentation')
        for i1 in range(npfs.shape[0]):
            print('Predicting on slice ' + str(i1) + ' of ' +
                  str(npfs.shape[0]))
            X= np.empty((1, 2, 512, 512))
            X[0, 0, ...] = normalize(npfs[i1, ...])
            X[0, 1, ...] = normalize(npwofs[i1, ...])
            result = np.zeros((100,3,512,512))
            for j in range(100):
                result[j,...] = self.f((X,1))[0]
            predictions = result.mean(axis=0)
            uncertainty = result.std(axis=0)
            predictions[predictions>=0.5] = 1
            predictions[predictions<0.5] = 0

            seg[i1, : ,:] = seg[i1, :, :] + predictions[0, 1, ...]
            seg[i1, :, :] = seg[i1, :, :] + predictions[0, 2, ...]*2
            uncert[i1,...] = uncertainty
        print('Finished Segmenting')
        self.SEG = sitk.GetImageFromArray(seg)
        self.uncert = uncert
        return seg, uncert

    def save_segmentations(self):
        '''
        Save function
        :return:
        '''
        save_spt = ''.join([self.ofolder + self.pt_num +
                            '_TissueSeg.mha'])
        sitk.WriteImage(self.SEG, save_spt)
        save_spt = ''.join([self.ofolder + self.pt_num +
                            '_Uncertainties.npy'])
        np.save(save_spt, self.uncert)


if __name__ == "__main__":
    # For Oberon
    scan_local = [r'/jaylabs/amartel_data2/Hongbo/processed_data'
                  r'/15_0122_5108281_wofs.mha',
                  r'/jaylabs/amartel_data2/Hongbo/processed_data'
                  r'/15_0122_5108281_700@151912.mha']

    model_dir = os.getcwd() +'/model/'

    ofolder = r'/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data/'

    # For Local
    # scan_local = [r'Y:\Hongbo\processed_data'
    #               r'\15_0122_5108281_wofs.mha',
    #               r'Y:\Hongbo\processed_data'
    #               r'\15_0122_5108281_700@151912.mha']
    #
    # model_dir = os.getcwd() + '\model\\'
    #
    # ofolder = r'Y:\Grey\2018-10-12-FGTSeg-Data\\'

    a = MCSegmenter(scan_local, model_dir, ofolder)

    seg, uncert = a.perform_segmentation()

    np.save( ofolder+'seg_data.npy', seg)
    np.save( ofolder + 'uncert_data.npy', uncert)
    print('done')