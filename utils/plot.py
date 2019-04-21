import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import numpy as np
import os


def plot_AE(model, test_dataset):
    if not os.path.exists("results"):
        os.makedirs("results")
    # --- Reconstruction plot ---
    n = 5
    sample_dataset = test_dataset
    x_input, y_input = next(sample_dataset.__iter__())
    x_input_sample, y_input_sample = map(lambda x: x[:n], (x_input, y_input))
    z = model.encode(x_input_sample).numpy()

    fig, axarr = plt.subplots(2, 5, figsize=(5, 2))
    x_input_sample = x_input_sample.numpy().reshape([n, 28, 28])
    x_output = model.decode(z, apply_sigmoid=True).numpy().reshape([n, 28, 28])

    for i in range(n):
        axarr[0, i].axis('off')
        axarr[1, i].axis('off')
        axarr[0, i].imshow(x_input_sample[i], cmap='binary')
        axarr[1, i].imshow(x_output[i], cmap='binary')

    fig.savefig("results/AE_reconstruction.png")

    # --- Distribution plot ---
    '''
    This part is significant only if the latent dimension is 2,
    but it works in other cases anyway.
    '''
    z = model.encode(x_input)
    labels = y_input.numpy()
    z1, z2 = z.numpy().T[0], z.numpy().T[1]

    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
    cs = [colors[y] for y in labels]
    classes = list(range(10))

    recs = []
    for i in range(0, len(cs)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=cs[i]))

    fig_dist = plt.figure(figsize=(8, 8))
    ax_dist = fig_dist.add_subplot(111)
    ax_dist.legend(recs, classes, loc=0)
    ax_dist.scatter(z1, z2, color=cs)

    fig_dist.savefig("results/AE_distribution.png")


def plot_VAE(model, test_dataset):
    if not os.path.exists("results"):
        os.makedirs("results")
    # --- Reconstruction plot ---
    n = 5
    sample_dataset = test_dataset
    x_input, y_input = next(sample_dataset.__iter__())
    x_input_sample, y_input_sample = map(lambda x: x[:n], (x_input, y_input))
    z = model.encode(x_input_sample)[0].numpy()

    fig, axarr = plt.subplots(2, 5, figsize=(5, 2))
    x_input_sample = x_input_sample.numpy().reshape([n, 28, 28])
    x_output = model.decode(z, apply_sigmoid=True).numpy().reshape([n, 28, 28])

    for i in range(n):
        axarr[0, i].axis('off')
        axarr[1, i].axis('off')
        axarr[0, i].imshow(x_input_sample[i], cmap='binary')
        axarr[1, i].imshow(x_output[i], cmap='binary')

    fig.savefig("results/VAE_reconstruction.png")

    # --- Distribution plot ---
    '''
    This part is significant only if the latent dimension is 2,
    but it works in other cases anyway.
    '''
    z, _ = model.encode(x_input)
    labels = y_input.numpy()
    z1, z2 = z.numpy().T[0], z.numpy().T[1]

    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
    cs = [colors[y] for y in labels]
    classes = list(range(10))

    recs = []
    for i in range(0, len(cs)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=cs[i]))

    fig_dist = plt.figure(figsize=(8, 8))
    ax_dist = fig_dist.add_subplot(111)
    ax_dist.legend(recs, classes, loc=0)
    ax_dist.scatter(z1, z2, color=cs)

    fig_dist.savefig("results/VAE_distribution.png")


def plot_CVAE(model, test_dataset):
    if not os.path.exists("results"):
        os.makedirs("results")
    # --- Reconstruction plot ---
    n = 5
    sample_dataset = test_dataset
    x_input, y_input = next(sample_dataset.__iter__())
    x_input_sample, y_input_sample = map(lambda x: x[:n], (x_input, y_input))
    z = model.encode(x_input_sample, y_input_sample)[0].numpy()

    fig, axarr = plt.subplots(2, 5, figsize=(5, 2))
    x_input_sample = x_input_sample.numpy().reshape([n, 28, 28])
    x_output = model.decode(z, y_input_sample, apply_sigmoid=True).numpy().reshape([n, 28, 28])

    for i in range(n):
        axarr[0, i].axis('off')
        axarr[1, i].axis('off')
        axarr[0, i].imshow(x_input_sample[i], cmap='binary')
        axarr[1, i].imshow(x_output[i], cmap='binary')

    fig.savefig("results/CVAE_reconstruction.png")

    # --- Distribution plot ---
    '''
    This part is significant only if the latent dimension is 2,
    but it works in other cases anyway.
    '''
    z, _ = model.encode(x_input)
    labels = y_input.numpy()
    z1, z2 = z.numpy().T[0], z.numpy().T[1]

    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
    cs = [colors[y] for y in labels]
    classes = list(range(10))

    recs = []
    for i in range(0, len(cs)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=cs[i]))

    fig_dist = plt.figure(figsize=(8, 8))
    ax_dist = fig_dist.add_subplot(111)
    ax_dist.legend(recs, classes, loc=0)
    ax_dist.scatter(z1, z2, color=cs)

    fig_dist.savefig("results/CVAE_distribution.png")

