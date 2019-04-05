from Pix2Pix import Pix2Pix
import keras as K

if __name__ == "__main__":

    # Train the Augmentation Model using a cGAN

    data_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
               '/imagesTr/'
    target_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
                 '/labelsTr/'
    pretrained_model = '/home/geklein/2019-03-30-CSC2541Project/UNetAugmentor_grey_interp/'
    ofolder = '/home/geklein/2019-03-30-CSC2541Project/cGANUnetAugmentor_grey_interp/'

    cgan = Pix2Pix(pretrained_folder=pretrained_model,
                   data_dir=data_dir,
                   target_dir=target_dir,
                   ofolder=ofolder,
                   batch_size=50)

    cgan.train(epochs=2000, sample_interval=1000)

    print('done')