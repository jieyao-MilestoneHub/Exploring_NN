import os
import matplotlib.pyplot as plt
import tensorflow as tf
來自 datetime import datetime
from tensorflow.keras import datasets, layers, models

# 模型儲存路徑
LENET_MODEL_PATH = 'models/lenet_model.h5'
ALEXNET_MODEL_PATH = 'models/alexnet_model.h5'

# 準備數據
def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    return (train_images, train_labels), (test_images, test_labels)

# 建立 LeNet 模型
def build_lenet_model():
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

# 建立 AlexNet 模型
def build_alexnet_model():
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

# 訓練模型
def train_model(model, train_images, train_labels, model_path):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    start = datetime.now()
    history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)
    print(f"Training time: {(datetime.now()-start).total_seconds()} seconds")
    model.save(model_path)
    return history

# 畫出損失曲線
def plot_loss(history, model_name):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'visual_result/{model_name}_loss.png')
    plt.close()

# 測試單張圖片
def predict_single_image(model_path, test_images, image_index):
    # 載入模型
    model = tf.keras.models.load_model(model_path)
    # 預測數字
    prediction = model.predict(test_images[image_index:image_index+1])
    predicted_digit = tf.argmax(prediction, axis=1).numpy()[0]
    # 顯示圖片
    plt.imshow(test_images[image_index].reshape(28, 28), cmap='gray')
    plt.title(f"Prdict: {predicted_digit}")
    plt.savefig("visual_result/" + os.path.basename("models/lenet_model.h5").split('_')[0] + "-test.png")


def train_lenet(train_images, train_labels):
    lenet_model = build_lenet_model()
    lenet_history = train_model(lenet_model, train_images, train_labels, LENET_MODEL_PATH)
    tf.keras.utils.plot_model(lenet_model, to_file='visual_result/lenet_model.png', show_shapes=True, show_layer_names=True)
    plot_loss(lenet_history, 'LeNet')


def train_alexnet(train_images, train_labels):
    alexnet_model = build_alexnet_model()
    alexnet_history = train_model(alexnet_model, train_images, train_labels, ALEXNET_MODEL_PATH)
    tf.keras.utils.plot_model(alexnet_model, to_file='visual_result/alexnet_model.png', show_shapes=True, show_layer_names=True)
    plot_loss(alexnet_history, 'AlexNet')


if __name__ == '__main__':

    (train_images, train_labels), _ = load_data()
    model_train_and_test = {"lenet": LENET_MODEL_PATH, "alexnet": ALEXNET_MODEL_PATH}

    for key, value in model_train_and_test.items():
        if os.path.exists(value):
            test_images, _ = datasets.mnist.load_data()[1]
            predict_single_image(value, test_images, 0)
        else:
            if key == "lenet":
                train_lenet()
            elif key == "alexnet":
                train_alexnet()