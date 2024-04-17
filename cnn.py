import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt
from datetime import datetime
import os

class CNNModels:
    def __init__(self, model_name="LeNet"):
        self.model_name = model_name
        if model_name == "LeNet":
            self.model = self.build_lenet_model()
        elif model_name == "AlexNet":
            self.model = self.build_alexnet_model()
        else:
            raise ValueError("Unsupported model type. Choose 'LeNet' or 'AlexNet'.")
        
        self.model_path = f'models/{model_name.lower()}_model.keras'

    def load_data(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        if self.model_name == "LeNet":
            train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
            test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
        elif self.model_name == "AlexNet":
            train_images = tf.image.resize(train_images, [227, 227]).numpy().reshape((60000, 227, 227, 1)).astype('float32') / 255
            test_images = tf.image.resize(test_images, [227, 227]).numpy().reshape((10000, 227, 227, 1)).astype('float32') / 255
        return (train_images, train_labels), (test_images, test_labels)

    def build_lenet_model(self):
        model = models.Sequential([
            layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(120, activation='relu'),
            layers.Dense(84, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        return model

    def build_alexnet_model(self):
        model = models.Sequential([
            layers.Resizing(227, 227),
            layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 1)),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Conv2D(256, kernel_size=(5, 5), activation='relu', padding="same"),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding="same"),
            layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding="same"),
            layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same"),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        return model

    def train_model(self, train_images, train_labels):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        start = datetime.now()
        history = self.model.fit(train_images, train_labels, epochs=10, validation_split=0.1)
        print(f"Training time: {(datetime.now() - start).total_seconds()} seconds")
        self.model.save(self.model_path)
        return history

    def plot_loss(self, history):
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{self.model_name} Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'visual_result/{self.model_name.lower()}_loss.png')
        plt.close()

    def predict_single_image(self, test_images, image_index):
        model = tf.keras.models.load_model(self.model_path)
        prediction = model.predict(test_images[image_index:image_index + 1])
        predicted_digit = tf.argmax(prediction, axis=1).numpy()[0]
        plt.imshow(test_images[image_index].reshape(28, 28), cmap='gray')
        plt.title(f"Predict: {predicted_digit}")
        plt.savefig(f"visual_result/{self.model_name.lower()}-test.png")

if __name__ == "__main__":
    
    # LeNet
    cnn_model = CNNModels("LeNet")
    (train_images, train_labels), (test_images, test_labels) = cnn_model.load_data()
    history = cnn_model.train_model(train_images, train_labels)
    # cnn_model.plot_loss(history)
    cnn_model.predict_single_image(test_images, 0)
    cnn_model.model.summary()

    # # AlexNet
    # alexnet_model = CNNModels("AlexNet")
    # (train_images, train_labels), (test_images, test_labels) = alexnet_model.load_data()
    # alexnet_history = alexnet_model.train_model(train_images, train_labels)
    # alexnet_model.plot_loss(alexnet_history)
    # alexnet_model.predict_single_image(test_images, 0)
    # alexnet_model.model.summary()