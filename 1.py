#采用训练好的模型进行——VGG16
#大约达到92%的训练精度
#启用显存
#import tensorflow as tf
#config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
#sess = tf.compat.v1.Session(config=config)

#爆显存了，找了这段代码，但是实际上还是没啥作用
#不知道为什么爆显存了还是能跑的动
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = InteractiveSession(config=config)

#加载VGG16
from tensorflow import keras
from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

#数据集的加载
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'D:/电力学院/出国/在线项目/项目素材/dataset/train'
validation_dir = 'D:/电力学院/出国/在线项目/项目素材/dataset/validation'
test_dir = 'D:/电力学院/出国/在线项目/项目素材/dataset/test'

#数据集的处理（没有很理解在干什么）
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 10

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 120)
validation_features, validation_labels = extract_features(validation_dir, 40)
test_features, test_labels = extract_features(test_dir, 40)

train_features = np.reshape(train_features, (120, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (40, 4 * 4 * 512))
test_features = np.reshape(test_features, (40, 4 * 4 * 512))

#VGG16的微调
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=10,
                    validation_data=(validation_features, validation_labels))

#作图，懂得懂得
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
