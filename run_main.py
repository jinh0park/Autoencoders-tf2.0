# parser_args code referred the hwalseoklee's code:
# https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/run_main.py
import tensorflow as tf
from utils import mnist, plot
from model.autoencoder import AE, VAE, CVAE
from train_utils.autoencoder import AETrain, VAETrain, CVAETrain
import time
import argparse


def parse_args():
    desc = "Tensorflow 2.0 implementation of 'AutoEncoder Families (AE, VAE, CVAE(Conditional VAE))'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--ae_type', type=str, default=False,
                        help='Type of autoencoder: [AE, VAE, CVAE]')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Degree of latent dimension(a.k.a. "z")')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='The number of training epochs')
    parser.add_argument('--learn_rate', type=float, default=1e-4,
                        help='Learning rate during training')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    return parser.parse_args()


def train_AE(latent_dim=2, epochs=100, lr=1e-4, batch_size=1000):
    model = AE(latent_dim, net_type='simple')

    train_dataset, test_dataset = mnist.load_dataset(batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(lr)

    for epoch in range(1, epochs + 1):
        t = time.time()
        last_loss = 0
        for train_x, _ in train_dataset:
            gradients, loss = AETrain.compute_gradients(model, train_x)
            AETrain.apply_gradients(optimizer, gradients, model.trainable_variables)
            last_loss = loss
        if epoch % 10 == 0:
            print('Epoch {}, Loss: {}, Remaining Time at This Epoch: {:.2f}'.format(
                epoch, last_loss, time.time() - t
            ))

    plot.plot_AE(model, test_dataset)

    return model


def train_VAE(latent_dim=2, epochs=100, lr=1e-4, batch_size=1000):
    model = VAE(latent_dim, net_type='conv')

    train_dataset, test_dataset = mnist.load_dataset(batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(lr)

    for epoch in range(1, epochs + 1):
        t = time.time()
        last_loss = 0
        for train_x, _ in train_dataset:
            gradients, loss = VAETrain.compute_gradients(model, train_x)
            VAETrain.apply_gradients(optimizer, gradients, model.trainable_variables)
            last_loss = loss
        if epoch % 10 == 0:
            print('Epoch {}, Loss: {}, Remaining Time at This Epoch: {:.2f}'.format(
                epoch, last_loss, time.time() - t
            ))

    plot.plot_VAE(model, test_dataset)

    return model


def train_CVAE(latent_dim=2, epochs=100, lr=1e-4, batch_size=1000):
    model = CVAE(latent_dim)

    train_dataset, test_dataset = mnist.load_dataset(batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(lr)

    for epoch in range(1, epochs + 1):
        t = time.time()
        last_loss = 0
        for train_x, train_y in train_dataset:
            gradients, loss = CVAETrain.compute_gradients(model, train_x, train_y)
            CVAETrain.apply_gradients(optimizer, gradients, model.trainable_variables)
            last_loss = loss
        if epoch % 10 == 0:
            print('Epoch {}, Loss: {}, Remaining Time at This Epoch: {:.2f}'.format(
                epoch, last_loss, time.time() - t
            ))

    plot.plot_CVAE(model, test_dataset)

    return model


def main(args):
    if args.ae_type == 'AE':
        train_AE(
            latent_dim=args.latent_dim,
            epochs=args.num_epochs,
            lr=args.learn_rate,
            batch_size=args.batch_size
        )
    elif args.ae_type == 'VAE':
        train_VAE(
            latent_dim=args.latent_dim,
            epochs=args.num_epochs,
            lr=args.learn_rate,
            batch_size=args.batch_size
        )
    elif args.ae_type == 'CVAE':
        train_CVAE(
            latent_dim=args.latent_dim,
            epochs=args.num_epochs,
            lr=args.learn_rate,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    args = parse_args()
    if args is None:
        exit()
    main(args)
