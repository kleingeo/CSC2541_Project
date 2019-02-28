import pandas as pd
import numpy as np
import csv

def MNIST_Generator(filename, batch_size, augment=False):
    csv_file = open(filename)
    csv_reader = csv.reader(csv_file, delimiter=',')

    csv_df = pd.read_csv(filename, index_col=False, header=None)

    csv_array = csv_df.values

    total_num_samples = len(csv_array)

    # if shuffle == True:
    #     sample_index = np.arange(0, total_num_samples)
    #     np.random.shuffle(sample_index)
    #
    #     csv_reader = csv_reader[sample_index]

    # Sample counter
    sample_counter = 0


    while(True):

        img_batch_hold = np.zeros((batch_size,) + (32, 32))
        label_batch_hold = np.zeros((batch_size,) + (32, 32))

        for idx in range(batch_size):

            if sample_counter >= total_num_samples:
                sample_counter = 0
                total_skipped = 0

            label = int(csv_array[sample_counter, 0])

            img = np.array(csv_array[sample_counter, 1:]).reshape(28, 28)

            img = np.pad(img, ((2, 2), (2, 2)), mode='constant', constant_values=0)

            if augment==True:
                pass

            img_batch_hold[idx] = img

            # import matplotlib.pyplot as plt
            # plt.imshow(img)
            # plt.show()

            label_batch_hold[idx] = label

            sample_counter = sample_counter + 1


        img_batch_hold = np.expand_dims(img_batch_hold, axis=-1)

        img_batch_hold = (img_batch_hold / 127.5) - 1

        yield img_batch_hold


if __name__ == '__main__':

    gen = MNIST_Generator('mnist_train.csv', batch_size=1)

    next(gen)