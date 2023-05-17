import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, GlobalMaxPool2D
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping


def CNN_model(arr_img_tns_train_aug_res_copy, arr_y_label_train_aug, arr_img_tns_val_aug_res_copy,
              arr_y_label_val_aug):
    tf.keras.backend.clear_session()
    CNN = Sequential(name="Sequential_CNN")

    CNN.add(Conv2D(6, kernel_size=(3, 3),
                   strides=(2, 2), padding="same",
                   activation="relu", input_shape=(250, 300, 2)))

    CNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding="valid"))
    # CNN.add(GlobalMaxPool2D())

    # Add another pair of Conv2D and MaxPooling2D for more model depth,
    # followed by the flatten and multiple dense layers
    #
    CNN.add(Conv2D(12, kernel_size=(3, 3),
                   strides=(2, 2), padding="same",
                   activation="relu"))

    CNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding="valid"))
    # CNN.add(GlobalMaxPool2D())

    CNN.add(Conv2D(24, kernel_size=(3, 3),
                   strides=(2, 2), padding="same",
                   activation="relu"))

    # CNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
    #                      padding="valid"))
    #
    # CNN.add(Flatten())
    CNN.add(GlobalMaxPool2D())
    CNN.add(Dropout(0.7))
    CNN.add(Dense(4, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4),
                  bias_regularizer=regularizers.L2(1e-4),
                  activity_regularizer=regularizers.L2(5e-5), activation='linear'))
    CNN.add(Activation(activation='relu'))
    # CNN.add(Dense(4, activation='relu'))
    # CNN.add(Dense(32, activation='relu'))
    CNN.add(Dense(1, activation='sigmoid'))

    CNN.summary()

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=5e-4,
                decay_steps=1000,
                decay_rate=0.9)
    CNN.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  # Low learning rate
                loss=keras.losses.BinaryCrossentropy(),
                metrics=[keras.metrics.BinaryAccuracy()])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=0.005,
                               restore_best_weights=True)
    # Training del modello su un numero di epoche dato in input ed un numero di immagini definito nello shuffling
    # CNN.fit(arr_img_tns_train_aug_res_copy[:, :, :, :2]/255, arr_y_label_train_aug,
    #         batch_size=10, epochs=250,
    #         validation_data=(arr_img_tns_val_aug_res_copy[:, :, :, :2]/255, arr_y_label_val_aug), shuffle=True,
    #         callbacks=[es])

    return CNN
    # CNN(arr_img_tns_test_res_copy[:, :, :, :2], training=False)

    # CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])