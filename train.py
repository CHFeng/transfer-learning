# IMPORT MODULES
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet201, Xception

# DenseNet201 default image size is 224
# Xception default image size is 299
IMG_SIZE = 224
# 辨識物件的種類
CLASSES_NUM = 7
# 設定預訓練使用的CNN辨識模組
MODE_NAME = 'DenseNet201'
# -----------------------------1.客製化模型--------------------------------
# 載入keras模型(更換輸出圖片尺寸為指定的尺寸)
if MODE_NAME == 'DenseNet201':
    model = DenseNet201(include_top=False, weights='imagenet', input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    checkpoint_file_name = 'DenseNet201_checkpoint_v2.h5'
    retrained_file_name = 'DenseNet201_retrained_v2.h5'
elif MODE_NAME == 'Xception':
    model = Xception(include_top=False, weights='imagenet', input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    checkpoint_file_name = 'Xception_checkpoint_v2.h5'
    retrained_file_name = 'Xception_retrained_v2.h5'
# 設定凍結原始捲積層為不可訓練
model.trainable = False
# 定義輸出層
x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(CLASSES_NUM, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)
model.summary()
# 編譯模型
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------2.設置callbacks----------------------------
# 設定earlystop條件
estop = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

# 設定模型儲存條件
checkpoint = ModelCheckpoint(checkpoint_file_name, verbose=1, monitor='val_loss', save_best_only=True, mode='min')

# 設定lr降低條件(0.001 → 0.0002 → 0.00004 → 0.00001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='min', verbose=1, min_lr=1e-4)

# -----------------------------3.設置資料集--------------------------------
# 設定ImageDataGenerator參數(路徑、批量、圖片尺寸)
train_dir = './data/train/'
valid_dir = './data/val/'
test_dir = './data/test/'
batch_size = 64
target_size = (IMG_SIZE, IMG_SIZE)

# 設定批量生成器
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.5,
                                   fill_mode="nearest")

val_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# 讀取資料集+批量生成器，產生每epoch訓練樣本
train_generator = train_datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size)

valid_generator = val_datagen.flow_from_directory(valid_dir, target_size=target_size, batch_size=batch_size)

test_generator = test_datagen.flow_from_directory(test_dir, target_size=target_size, batch_size=batch_size, shuffle=False)
print('物件識別序號:', train_generator.class_indices)
# -----------------------------4.開始訓練模型------------------------------
# 重新訓練權重
history = model.fit_generator(train_generator,
                              epochs=100,
                              verbose=1,
                              steps_per_epoch=train_generator.samples // batch_size,
                              validation_data=valid_generator,
                              validation_steps=valid_generator.samples // batch_size,
                              callbacks=[checkpoint, estop, reduce_lr])
# -----------------------5.儲存模型、紀錄學習曲線------------------------
# 儲存模型
model.save(retrained_file_name)
print('已儲存' + retrained_file_name)

# 畫出acc學習曲線
acc = history.history['accuracy']
epochs = range(1, len(acc) + 1)
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.grid()
# 儲存acc學習曲線
plt.savefig('./{}_acc.png'.format(MODE_NAME))
plt.show()

# 畫出loss學習曲線
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='upper right')
plt.grid()
# 儲存loss學習曲線
plt.savefig('./{}_loss.png'.format(MODE_NAME))
plt.show()