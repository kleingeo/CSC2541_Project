"""
2018-12-10
(c) A. Martel Lab Co.
author: G.Kuling
This is my segmenter code.
WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
"""
import numpy as np
import SimpleITK as sitk
import keras as k
import os
from Utils import My_new_loss

def normalize(test):
    upper_lim = 99.9
    top = np.percentile(test, upper_lim)
    test[np.where(test > top)] = top
    test = (test * 255. / top).astype(np.uint8)
    return test

class Segmenter:
    def __init__(self,
                 scan_local,
                 model_dir,
                 ofolder):
        '''
        This is the initiator for the Segmenter
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
        self.model = k.models.model_from_json(loaded_model_json)

        # Load the weights to the model
        self.model.load_weights(''.join([model_dir, '/', weights_file_name]))
        self.model.compile(loss=My_new_loss, metrics=[My_new_loss],
                           optimizer='ADAM')
        print('Model is ready to predict.')
        self.pt_num = '_'.join(scan_local[0].split('\\')[-1].split('_')[0:-1])

        self.ofolder = ofolder

    def perform_segmentation(self):
        '''
        Performs the segmentation of the test subject.
        :return: segmentation of tissues as SITK Image
        '''
        npfs = sitk.GetArrayFromImage(self.FS)
        npwofs = sitk.GetArrayFromImage(self.WOFS)
        seg = np.empty(npfs.shape)
        print('Beginning Segmentation')
        for i1 in range(npfs.shape[0]):
            X= np.empty((1, 2, 512, 512))
            X[0,0,...] = normalize(npfs[i1, ...])
            X[0, 1, ...] = normalize(npwofs[i1, ...])

            y = self.model.predict(X, verbose=0)
            y[y>=0.5] = 1
            y[y<0.5] = 0

            seg[i1, : ,:] = seg[i1, :, :] + y[0, 1, ...]
            seg[i1, :, :] = seg[i1, :, :] + y[0, 2, ...]*2
        print('Finished Segmenting')
        self.SEG = sitk.GetImageFromArray(seg)

        return self.SEG

    def save_segmentations(self):
        '''
        Save Function
        :return:
        '''
        save_spt = ''.join([self.ofolder + self.pt_num +
                            '_TissueSeg.mha'])
        sitk.WriteImage(self.SEG, save_spt)

if __name__ == "__main__":
    scan_local = [r'Y:\Hongbo\processed_data\72_0293_7491268_wofs.mha',
                  r'Y:\Hongbo\processed_data\72_0293_7491268_600@122131.mha']

    model_dir = r'X:\2018-09-28-FGTSeg2\model\\'

    ofolder = r'Y:\Grey\2018-10-12-FGTSeg-Data\Segmented\\'

    a = Segmenter(scan_local, model_dir, ofolder)

    a.perform_segmentation()


    print('done')



