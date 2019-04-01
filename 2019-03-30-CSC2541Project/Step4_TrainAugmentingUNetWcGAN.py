from CGAN import CGAN
import keras as K

if __name__ == "__main__":

    # Train the Augmentation Model using a cGAN

    data_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
               '/imagesTr/'
    target_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
                 '/labelsTr/'
    pretrained_model = '/home/gkuling/2019-03-30-CSC2541Project/UNetAugmentor/'
    ofolder = '/home/gkuling/2019-03-30-CSC2541Project/cGANUnetAugmentor/'

    cgan = CGAN(pretrained_folder=pretrained_model,
                data_dir=data_dir,
                target_dir=target_dir,
                ofolder=ofolder)
    cgan.train(epochs=2000, batch_size=50, sample_interval=200)

    print('done')