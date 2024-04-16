import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import time

'''
概念：
GAN 包括兩個主要部分：生成器(Generator)和鑑別器(Discriminator)。
生成器產生新的、盡量接近真實的資料；而鑑別器則嘗試區分真實資料和產生器產生的假資料。
'''
tf.random.set_seed(42)

# 定義產生器
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(784, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# 定義鑑別器
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def compile_models():
    binary_cross_entropy = keras.losses.BinaryCrossentropy()
    return (keras.optimizers.Adam(), keras.optimizers.Adam(), binary_cross_entropy)

@tf.function
def train_step(images, generator, discriminator, gen_optimizer, disc_optimizer, loss_fn, latent_dim):
    noise = tf.random.normal([tf.shape(images)[0], latent_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = loss_fn(tf.ones_like(fake_output), fake_output)
        disc_loss = loss_fn(tf.ones_like(real_output), real_output) + loss_fn(tf.zeros_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def setup_dataset(batch_size):
    (mnist_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    mnist_images = mnist_images.astype('float32') / 255.0
    mnist_images = mnist_images.reshape(-1, 784)  # Flatten images to 784-dimensional vectors
    return tf.data.Dataset.from_tensor_slices(mnist_images).shuffle(10000).batch(batch_size)

def train_gan(generator, discriminator, epochs, batch_size, latent_dim):
    gen_optimizer, disc_optimizer, loss_fn = compile_models()
    dataset = setup_dataset(batch_size)
    gen_losses = []
    disc_losses = []
    
    for epoch in range(epochs):
        start_time = time.time()
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, gen_optimizer, disc_optimizer, loss_fn, latent_dim)
            gen_losses.append(gen_loss.numpy())
            disc_losses.append(disc_loss.numpy())
        print(f'Time for epoch {epoch + 1} is {time.time() - start_time} sec')
    return gen_losses, disc_losses

def test_generator(generator, latent_dim, test_noise):
    test_noise = tf.random.normal([16, latent_dim])
    generated_images = generator(test_noise, training=False).numpy().reshape(16, 28, 28)
    plt.figure(figsize=(15, 15))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
        noise_stats = f"Max: {test_noise[i].numpy().max():.2f}, Min: {test_noise[i].numpy().min():.2f}"
        plt.title(noise_stats)
    plt.savefig("visual_result/gan-generated.png")

if __name__ == "__main__":
    latent_dim = 100
    batch_size = 256
    epochs = 50

    generator = Generator()
    discriminator = Discriminator()

    gen_losses, disc_losses = train_gan(generator, discriminator, epochs, batch_size, latent_dim)
    generator.save_weights('models/gan-generator_model.weights.h5')
    discriminator.save_weights('models/gan-discriminator_model.weights.h5')
    tf.keras.utils.plot_model(generator, to_file='visual_result/gan-generator_model.png', show_shapes=True, show_layer_names=True)
    tf.keras.utils.plot_model(discriminator, to_file='visual_result/gan-generator_model.png', show_shapes=True, show_layer_names=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(gen_losses, label='Generator Loss')
    plt.title('Generator Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.title('Discriminator Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visual_result/gan-training_losses.png')


    test_noise = tf.random.normal([16, latent_dim])
    test_generator(generator, latent_dim, test_noise)