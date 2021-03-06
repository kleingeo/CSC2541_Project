from Trainer import Trainer
from Predictor import Predictor
import os
import keras as K

if __name__ == "__main__":
    '''This script is used to train a basic UNet on prostate segmentation in
    MRI. This will be used as a base line in our project. It is traained on
    75% of the volumes, resampled to istropic voxel size of 2.0, resultingn
    in 803 training examples. It is then tested on 25% of the volumes,
    resampled the same, resulting in 278 testing examples.
    '''

    # Train the Model
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,4,7"

    data_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
               '/imagesTr/'
    target_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
                 '/labelsTr/'
    ofolder = '/home/geklein/2019-03-30-CSC2541Project/UNet_regularWAug/'

    aug_folder = '/home/geklein/2019-03-30-CSC2541Project/UNetAugmentor/'

    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        CUDA_VISIBLE_DEVICES = ['None']

    a = Trainer(data_dir, target_dir, ofolder, samples_per_card=10,
                epochs=50, gpus_used=len(CUDA_VISIBLE_DEVICES),
                batch_size=None, training_direction=True,
                data_aug=True,
                aug_folder=aug_folder)

    a.train_the_model(t_opt=K.optimizers.adam(lr=1e-5))

    # Test the Model

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    data_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
               '/imagesTs/'
    target_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
                 '/labelsTs/'
    model_folder = '/home/geklein/2019-03-30-CSC2541Project/UNet_regularWAug/'

    ofolder = '/home/geklein/2019-03-30-CSC2541Project/UNet_regularWAug' \
              '/test_results'

    a = Predictor(model_folder=model_folder,
                  data_folder=data_dir,
                  target_folder=target_dir,
                  ofolder=ofolder,
                  opt='ADAM',
                  testing_direction=True)
    a.predict_and_evaluate()
    print('done')