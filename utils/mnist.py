import tensorflow as tf


def download_images():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    return (train_images, train_labels), (test_images, test_labels)


def normalize(train_images, test_images):
    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.
    return train_images, test_images


def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = download_images()
    train_images, test_images = normalize(train_images, test_images)

    TRAIN_BUF = 60000
    TEST_BUF = 10000

    BATCH_SIZE = 1000

    train_dataset_image = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)
    train_dataset_label = tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE)
    train_dataset = tf.data.Dataset.zip((train_dataset_image, train_dataset_label)).shuffle(TRAIN_BUF)

    test_dataset_image = tf.data.Dataset.from_tensor_slices(test_images).batch(BATCH_SIZE)
    test_dataset_label = tf.data.Dataset.from_tensor_slices(test_labels).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.zip((test_dataset_image, test_dataset_label)).shuffle(TEST_BUF)

    return train_dataset, test_dataset
