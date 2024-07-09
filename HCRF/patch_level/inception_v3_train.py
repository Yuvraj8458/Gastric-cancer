import time
import math, json, os, sys
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import pandas as pd
import tensorflow as tf

# num_classes=2
# def mycrossentropy(y_true, y_pred, e=0.14):
#        return (1-e)*K.categorical_crossentropy(y_pred,y_true) + e*K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/num_classes)

"""
#1. Data Path
#2. Model Save Path and Name, ensure to update when loading the model
#3. Modify the storage path for history
#4. Store figure titles and paths
"""
DATA_DIR = 'E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')

VALID_DIR = os.path.join(DATA_DIR, 'val')

SIZE = (75, 75)
BATCH_SIZE = 35


def save_history(History):
    acc = pd.Series(History.history['accuracy'], name='accuracy')
    loss = pd.Series(History.history['loss'], name='loss')
    val_acc = pd.Series(History.history['val_accuracy'], name='val_accuracy')
    val_loss = pd.Series(History.history['val_loss'], name='val_loss')
    com = pd.concat([acc, loss, val_acc, val_loss], axis=1)
    com.to_csv('E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\inception_v3\\model_03\\history.csv')


# "Plot accuracy and loss curves"
def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # "Change the title of the graph here!!"
    plt.title("inception_v3 model")
    plt.ylabel("acc-loss")
    plt.xlabel("epoch")
    plt.legend([" acc", "val acc", " loss", "val loss"], loc="upper right")
    # plt.show()
    # "Be sure to update the storage name!!"
    plt.savefig("E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\inception_v3\\model_03\\inception_v3_model.png")


if __name__ == "__main__":
    start = time.perf_counter()
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples / BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator()
    val_gen = keras.preprocessing.image.ImageDataGenerator()

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True,
                                      batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True,
                                              batch_size=BATCH_SIZE)
    classes = list(iter(batches.class_indices))
    Inp=Input((75,75,3))
    #model = keras.applications.resnet50.ResNet50()
    base_model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False,input_shape=(75, 75, 3))
    #model = keras.applications.vgg16.VGG16()
    #    model = keras.applications.xception.Xception()

    # base_model.layers.pop()
    x = base_model(Inp)
    # x = base_model(Inp).layers[-1].output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(classes), activation="softmax")(x)
    finetuned_model = Model(inputs=Inp, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=15)

    checkpointer = ModelCheckpoint('E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\inception_v3\\model_03\\inception_v3_best.h5',
                                   verbose=1, save_best_only=True)

    History = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=15,
                                            callbacks=[early_stopping, checkpointer], validation_data=val_batches,
                                            validation_steps=num_valid_steps)
    # end1 = time.clock()
    # print("save model before", end1)
    finetuned_model.save('E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\inception_v3\\model_03\\inception_v3_best_save.h5')
    save_history(History)
    plot_history(History)

    end2 = time.perf_counter()
    print("final is in ", end2 - start)