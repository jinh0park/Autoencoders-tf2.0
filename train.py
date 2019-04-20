import tensorflow as tf
from utils import mnist
from model.autoencoder import AE, VAE, CVAE
from train_utils.autoencoder import AETrain, VAETrain, CVAETrain
import time
import argparse


def parse_args():
    desc = "Tensorflow 2.0 implementation of 'AutoEncoder Families (AE, VAE, CVAE(Conditional VAE))'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--ae_type', type=str, default=False,
                        help='Type of autoencoder: [AE, VAE, CVAE] ')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Degree of latent dimension(a.k.a. "z")')


# only for AE
def train_AE():
    epochs = 10
    latent_dim = 10
    model = AE(latent_dim)

    train_dataset, _ = mnist.load_dataset()

    optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(1, epochs + 1):
        t = time.time()
        for train_x, _ in train_dataset:
            gradients, loss = AETrain.compute_gradients(model, train_x)
            AETrain.apply_gradients(optimizer, gradients, model.trainable_variables)
        if (epoch % 1 == 0):
            print(f'Epoch {epoch}, Loss: {loss}, Remaining Time at This Epoch: {time.time() - t:.2f}')
    return model


def train_VAE():
    epochs = 10
    latent_dim = 10
    model = VAE(latent_dim)

    train_dataset, _ = mnist.load_dataset()

    optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(1, epochs + 1):
        t = time.time()
        for train_x, _ in train_dataset:
            gradients, loss = VAETrain.compute_gradients(model, train_x)
            VAETrain.apply_gradients(optimizer, gradients, model.trainable_variables)
        if (epoch % 1 == 0):
            print(f'Epoch {epoch}, Loss: {loss}, Remaining Time at This Epoch: {time.time() - t:.2f}')
    return model


def train_CVAE():
    epochs = 10
    latent_dim = 10
    model = CVAE(latent_dim)

    train_dataset, _ = mnist.load_dataset()

    optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(1, epochs + 1):
        t = time.time()
        for train_x, train_y in train_dataset:
            gradients, loss = CVAETrain.compute_gradients(model, train_x, train_y)
            CVAETrain.apply_gradients(optimizer, gradients, model.trainable_variables)
        if (epoch % 1 == 0):
            print(f'Epoch {epoch}, Loss: {loss}, Remaining Time at This Epoch: {time.time() - t:.2f}')
    return model

if __name__ == "__main__":
    pass
