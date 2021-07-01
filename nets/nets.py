from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Activation


def getDenerator(fer2013plus_path="datasets/fer2013plus", batch_size=32):
    """

    :param fer2013plus_path:fer2013plus数据集目录
    :param batch_size:
    :return:
    """
    train_datagen = ImageDataGenerator(
        brightness_range=(0.7, 1.3),  # 随机亮度
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,  # 随机旋转
        zoom_range=(0.7, 1.3),  # 聚焦
        horizontal_flip=True)  # 水平翻转
    train_generator = train_datagen.flow_from_directory(fer2013plus_path+"/train",
                                                        target_size=(48, 48),
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    test_datagen = ImageDataGenerator()  # 水平翻转
    test_generator = test_datagen.flow_from_directory(fer2013plus_path+"/test",
                                                      target_size=(48, 48),
                                                      color_mode="grayscale",
                                                      batch_size=batch_size,
                                                      class_mode='categorical')

    return train_generator,test_generator


def getVXSlim():
    model = Sequential()

    # block1
    model.add(Conv2D(64, (5, 5), input_shape=(48, 48, 1), activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # block2
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # block3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # single conv
    model.add(Conv2D(64, (3, 3), padding='same'))

    # classifier
    model.add(Flatten())
    model.add(Dense(7))
    model.add(BatchNormalization())
    model.add(Dense(7))

    # config
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model
