import os
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

'''
概念：
VAE 由兩部分組成：編碼器(Encoder)和解碼器(Decoder)。
編碼器負責將輸入資料壓縮成一個潛在空間的表示，而解碼器則負責從這個潛在表示中重建原始資料。
VAE 透過最小化重建誤差和潛在表示的分佈與預定義分佈（如高斯分佈）之間的差異來進行訓練。
'''

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(models.Model):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""
    def __init__(self, latent_dim=32, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(64, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(models.Model):
    """Converts z, the encoded digit vector, back into a readable digit."""
    def __init__(self, original_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        return self.dense_output(inputs)

class VAE(models.Model):
    def __init__(self, original_dim, latent_dim=32, name="vae", **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(original_dim=original_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed

def plot_and_save_images(images, filename, title):
    num_images = images.shape[0]
    if num_images == 1:
        # If there's only one image, we handle it slightly differently
        fig, ax = plt.subplots()
        ax.imshow(images[0].reshape(28, 28), cmap='gray')
        ax.axis('off')
        plt.title(title)
    else:
        fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
        for img, ax in zip(images, axes):
            ax.imshow(img.reshape(28, 28), cmap='gray')
            ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)

def train_and_evaluate():
    # Load MNIST dataset
    (train_images, _), (test_images, _) = mnist.load_data()
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    train_images = train_images.reshape((train_images.shape[0], 784))
    test_images = test_images.reshape((test_images.shape[0], 784))

    # Create dataset objects
    batch_size = 128
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(1024).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)

    original_dim = 784
    vae = VAE(original_dim)
    vae.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # Check if training or testing
    if os.path.exists('models/vae-weights.weights.h5'):
        vae.load_weights('models/vae-weights.weights.h5')
        latent_samples = tf.random.normal(shape=(1, 32))
        generated_images = vae.decoder(latent_samples)
        plot_and_save_images(generated_images.numpy(), "visual_result/vae-generated_images.png", "Generated Images")
    else:
        # Training mode
        vae.fit(train_dataset.map(lambda x: (x, x)), epochs=30, validation_data=test_dataset.map(lambda x: (x, x)))
        vae.save_weights('models/vae-weights.weights.h5')

if __name__ == "__main__":
    train_and_evaluate()
