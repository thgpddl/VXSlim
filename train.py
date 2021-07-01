import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from nets.nets import getVXSlim, getDenerator

patience = 50
batch_size = 32

# 获取数据生成器和模型
train_generator, test_generator = getDenerator(batch_size=batch_size)
model = getVXSlim()

# 定义回调函数
model_checkpoint = ModelCheckpoint('outputs/models/' + '{epoch:02d}-{val_accuracy:.2f}.hdf5', 'val_loss', verbose=1,save_best_only=True)
csv_logger = CSVLogger("outputs/logs/train.log", append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
Tensorboard = TensorBoard(log_dir="outputs/TensorBoardLog")

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr, Tensorboard]

# 训练
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.samples / batch_size,
                              epochs=100,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=test_generator)

# log可视化
dic = history.history
train_acc = dic['accuracy']
val_acc = dic['val_accuracy']
train_loss = dic['loss']
val_loss = dic['val_loss']

epoch = range(1, len(train_acc) + 1)

plt.plot(epoch, train_acc, 'r', label='Train acc')
plt.plot(epoch, val_acc, 'g', label='validation acc')
plt.title('training and validation accuracy')
plt.legend()  # 给图像加图例
plt.figure()
plt.plot(epoch, train_loss, 'r', label='Training loss')
plt.plot(epoch, val_loss, 'g', label='Validation loss')
plt.title('Taining and calidation loss')
plt.legend()
plt.show()
