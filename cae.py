import argparse

from keras.models import Model
from keras import backend as K
import numpy as np

from model import build_cae_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train Convolutional AutoEncoder and inference')
    parser.add_argument('--data_path', default='./data/cifar10.npz', type=str, help='path to dataset')
    parser.add_argument('--num_epoch', default=50, type=int, help='the number of epochs')
    parser.add_argument('--batch_size', default=100, type=int, help='mini batch size')
    parser.add_argument('--output_path', default='./data/cifar10_cae.npz', type=str, help='path to directory to output')

    args = parser.parse_args()

    return args


def load_data(data_to_path):
    """load data
    data should be compressed in npz
    """
    data = np.load(data_to_path)
    all_image = data['images']
    all_label = data['lables']
    all_image = (all_image - 127.0) / 127.0

    return all_image, all_label


def flat_feature(enc_out):
    enc_out_flat = []

    for i, con in enumerate(enc_out):
        s1, s2, s3 = con.shape
        enc_out_flat.append(con.reshape((s1 * s2 * s3,)))

    return np.array(enc_out_flat)


def main():
    """main function"""
    args = parse_args()
    data_path = args.data_path
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    output_path = args.output_path

    # load CIFAR10 data from data directory
    all_image, all_label = load_data(data_path)

    # build model and train
    autoencoder = build_cae_model()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(all_image, all_image,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    shuffle=True)

    # inference from encoder
    layer_name = 'enc'
    encoded_layer = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)
    enc_out = encoded_layer.predict(all_image)

    enc_out = flat_feature(enc_out)

    # save cae output
    np.savez(output_path, ae_out=enc_out, labels=all_label)


if __name__ == '__main__':
    main()
