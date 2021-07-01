from keras.models import Model,load_model
import tensorflow as tf

# keras模型的文件按路径
model=load_model(r"D:\FacialExpressionInteraction\VXSlim\outputs\models\mixVX_48x48_fer2013plus_32bs_43-0.83.hdf5",compile=False)

converter =tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("output.tflite", 'wb').write(tflite_model)