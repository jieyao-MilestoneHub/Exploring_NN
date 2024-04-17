import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt
import math

'''
概念：
GAN 包括兩個主要部分：生成器(Generator)和鑑別器(Discriminator)。
生成器產生新的、盡量接近真實的資料；而鑑別器則嘗試區分真實資料和產生器產生的假資料。
'''

tf.random.set_seed(42)
class GAN(keras.Model):
    def __init__(self, latent_dim):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(784, activation='sigmoid')  # Output layer to match the flattened MNIST image size
        ])
        return model

    def build_discriminator(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification: real or fake
        ])
        return model

    def compile(self, gen_optimizer, disc_optimizer, loss_fn):
        super(GAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([tf.shape(images)[0], self.latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            disc_loss = (self.loss_fn(tf.ones_like(real_output), real_output) +
                         self.loss_fn(tf.zeros_like(fake_output), fake_output))
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def save_models(self, generator_path, discriminator_path):
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)

    @staticmethod
    def load_generator(generator_path):
        '''
        Note: optimizer and loss setting need to be same as the training setting
        '''
        model = keras.models.load_model(generator_path)
        # Recompile the model if you plan to continue training or evaluation
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy())
        return model

    @staticmethod
    def load_discriminator(discriminator_path):
        '''
        Note: optimizer and loss setting need to be same as the training setting
        '''
        model = keras.models.load_model(discriminator_path)
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy())
        return model

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start_time = time.time()
            for image_batch in dataset:
                gen_loss, disc_loss = self.train_step(image_batch)
            print(f'Time for epoch {epoch + 1} is {time.time() - start_time} sec - Gen loss: {gen_loss.numpy()} Disc loss: {disc_loss.numpy()}')

    # @classmethod
    # def test_generator(cls, generator_path, latent_dim, num_examples=16):        
    #     generator = cls.load_generator(generator_path)
    #     test_noise = tf.random.normal([num_examples, latent_dim])
    #     generated_images = generator(test_noise, training=False).numpy().reshape(num_examples, 28, 28)
    #     plt.figure(figsize=(15, 15))
    #     for i in range(num_examples):
    #         plt.subplot(4, 4, i + 1)
    #         plt.imshow(generated_images[i], cmap='gray')
    #         plt.axis('off')
    #         noise_stats = f"Max: {test_noise[i].numpy().max():.2f}, Min: {test_noise[i].numpy().min():.2f}"
    #         plt.title(noise_stats)
    #     plt.savefig("visual_result/gan-generated.png")

    @classmethod
    def test_generator(cls, generator_path, latent_dim, num_examples=16):
        try:
            generator = cls.load_generator(generator_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        test_noise = tf.random.normal([num_examples, latent_dim])
        generated_images = generator(test_noise, training=False).numpy().reshape(num_examples, 28, 28)
        
        cols = int(math.sqrt(num_examples))
        rows = math.ceil(num_examples / cols)
        
        plt.figure(figsize=(cols * 2, rows * 2))
        for i in range(num_examples):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(generated_images[i], cmap='gray')
            plt.axis('off')
            noise_stats = f"Max: {test_noise[i].numpy().max():.2f}, Min: {test_noise[i].numpy().min():.2f}"
            plt.title(noise_stats)
        
        plt.tight_layout()
        plt.savefig("visual_result/gan-generated.png")
        plt.close()

def prepare_dataset(batch_size):
    (mnist_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    mnist_images = mnist_images.astype('float32') / 255.0
    mnist_images = mnist_images.reshape(-1, 784)  # Flatten images to 784-dimensional vectors
    
    # Check if the dataset size is divisible by the batch size or at least has enough data
    dataset_size = len(mnist_images)
    if dataset_size < batch_size:
        raise ValueError(f"Dataset size ({dataset_size}) is smaller than the batch size ({batch_size}).")
    
    dataset = tf.data.Dataset.from_tensor_slices(mnist_images).shuffle(10000).batch(batch_size)
    return dataset

def main(mode):
    latent_dim = 100
    batch_size = 64
    epochs = 10
    generator_path = "models/generator_model.keras"
    discriminator_path = "models/discriminator_model.keras"

    gan = GAN(latent_dim)

    if mode == "Train":
        gan.compile(gen_optimizer=keras.optimizers.Adam(), disc_optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.BinaryCrossentropy())
        dataset = prepare_dataset(batch_size)
        gan.train(dataset, epochs)
        gan.save_models(generator_path, discriminator_path)
    elif mode == "Test":
        # Test the generator
        GAN.test_generator(generator_path=generator_path, latent_dim=latent_dim, num_examples=15)
    else:
        raise ValueError("Invalid mode. Use 'Train' or 'Test'.")

if __name__ == "__main__":
    main(mode="Test")

