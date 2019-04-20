import matplotlib.pyplot as plt


def plot_AE(model, test_dataset):
    # --- Reconstruction plot ---
    n = 5
    sample_dataset = test_dataset
    x_input, _ = map(lambda x: x[:n], next(sample_dataset.__iter__()))
    z = model.encode(x_input).numpy()

    f, axarr = plt.subplots(2, 5, figsize=(5, 2))
    x_input = x_input.numpy().reshape([n, 28, 28])
    x_output = model.decode(z, apply_sigmoid=True).numpy().reshape([n, 28, 28])

    for i in range(n):
        axarr[0, i].axis('off')
        axarr[1, i].axis('off')
        axarr[0, i].imshow(x_input[i], cmap='binary')
        axarr[1, i].imshow(x_output[i], cmap='binary')

    f.savefig("AE_reconstruction.png")

    # --- Distribution plot ---

