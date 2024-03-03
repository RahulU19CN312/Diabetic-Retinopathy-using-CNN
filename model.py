
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
import warnings
warnings.filterwarnings("ignore")



import os

root = 'dataset/'
train_img_path = os.path.join(root, 'train_images')
train_path = os.path.join(root, 'train.csv')
test_img_path = os.path.join(root, 'test_images')
test_path = os.path.join(root, 'test.csv')
import pandas as pd
train_data = pd.read_csv(train_path)
train_data.head()
#train_data.hist()


n_classes = train_data.diagnosis.nunique()

classes = train_data.diagnosis.unique()
train_data.diagnosis.value_counts()

train_data['id_code'] = train_data['id_code'].apply(lambda x:x+'.png')
train_data['diagnosis'] = train_data['diagnosis'].apply(lambda x:str(x))
train_data.head()

train_data.dtypes


sz=224
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications.densenet import DenseNet121


inp = layers.Input(shape=(sz,sz,3))
conv_base = DenseNet121(include_top=False,
               weights='imagenet',
               input_tensor=inp)

x = layers.GlobalAveragePooling2D()(conv_base.output)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(n_classes, activation='softmax')(x)
model = models.Model(inp, out)

conv_base.trainable = False

from sklearn.utils import shuffle
train_data = shuffle(train_data, random_state=8)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_data, test_size=0.15, random_state=8, stratify=train_data[['diagnosis']])

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.applications.densenet import preprocess_input

# our batch size
bs=32



train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_batches = train_datagen.flow_from_dataframe(train_df, # The df
                                                  train_img_path, # Place on desk
                                                  x_col='id_code', # The column to get x
                                                  y_col='diagnosis', # The column to get y
                                                  #has_ext=True,
                                                  target_size=(sz, sz),
                                                  color_mode='rgb',
                                                  classes=None,

                                                  class_mode='categorical',
                                                  batch_size=bs,
                                                  shuffle=True)



valid_batches = valid_datagen.flow_from_dataframe(val_df,
                                                  train_img_path,
                                                  x_col='id_code',
                                                  y_col='diagnosis',
                                                  #has_ext=True,
                                                  target_size=(sz, sz),
                                                  color_mode='rgb',
                                                  classes=None,#list(train_batches.class_indices),#classes,
                                                  class_mode='categorical',
                                                  batch_size=bs,
                                                  shuffle=False)

NbClasses = len(train_batches.class_indices)


def flatten(y):
    return np.argmax(y, axis=1).reshape(-1)



val_df['diagnosis'].apply(lambda x: int(x))

from tensorflow.keras.callbacks import  EarlyStopping, ReduceLROnPlateau

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                                   verbose=1, mode='min', min_delta=0.0001)

early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=9)


model.summary()


model.compile(loss='categorical_crossentropy',
              #optimizer=Adam(learning_rate=1e-3),
              metrics=['acc'])
epochs = 20




history = model.fit(train_batches,
                              steps_per_epoch = train_batches.n // train_batches.batch_size,
                              epochs=epochs)

history = model.history
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')

plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('mo.h5')

