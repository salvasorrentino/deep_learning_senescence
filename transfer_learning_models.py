from tensorflow import keras
from tensorflow.keras import regularizers
import keras.applications.inception_v3 as inception
import keras.applications.efficientnet as efficientnet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet import EfficientNetB4
from keras.applications.resnet import ResNet50
import keras.applications.resnet as resnet
from keras.applications.mobilenet import MobileNet
import keras.applications.mobilenet as mobnet
from keras.applications.densenet import DenseNet121
import keras.applications.densenet as denseNet
from keras.applications.xception import Xception
import keras.applications.xception as xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import keras.applications.inception_resnet_v2 as incres
from keras.applications.nasnet import NASNetMobile
import keras.applications.nasnet as nasnet
from keras.applications.efficientnet import EfficientNetB2
from keras.applications.vgg16 import VGG16
import keras.applications.vgg16 as vgg
from keras.applications.mobilenet_v2 import MobileNetV2
import keras.applications.mobilenet_v2 as mobnet2
from keras.applications.resnet_v2 import ResNet101V2
import keras.applications.resnet_v2 as resnet2

from keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Qt5Agg')


class TransferLearningModel:
    def __init__(self, str_pretrained_model, flt_dropout=0.8,
                 flt_initial_learning_rate=5e-4, bln_include_top=False, input_shape=(250, 300, 3), bln_trainable=False,
                 bln_training=False):
        self.dct_pretrained_model = {'Inception': InceptionV3,
                                     'EfficientNetB4': EfficientNetB4,
                                     'ResNet': ResNet50,
                                     'Xception': Xception,
                                     'MobileNet': MobileNet,
                                     'DenseNet': DenseNet121,
                                     'EfficientNetB2': EfficientNetB2,
                                     'IncRes': InceptionResNetV2,
                                     'NasNet': NASNetMobile,
                                     'VGG16': VGG16,
                                     'MobNet2': MobileNetV2,
                                     'ResNet2': ResNet101V2}
        self.input_shape = input_shape
        self.bln_include_top = bln_include_top
        self.str_pretrained_model = str_pretrained_model
        self.bln_trainable = bln_trainable
        self.flt_dropout = flt_dropout
        self.flt_initial_learning_rate = flt_initial_learning_rate
        self.bln_training = bln_training
        self.pretrained_model = self.upload_model()
        self.model = self.build_model()

    def upload_model(self):
        # Model Initialization

        pretrained_model = self.dct_pretrained_model[self.str_pretrained_model]\
            (include_top=self.bln_include_top, input_shape=self.input_shape)
        pretrained_model.trainable = self.bln_trainable
        return pretrained_model

    def preprocess_input(self):

        if self.str_pretrained_model == 'Inception':
            return inception.preprocess_input
        elif (self.str_pretrained_model == 'EfficientNetB4') or (self.str_pretrained_model == 'EfficientNetB2'):
            return efficientnet.preprocess_input
        elif self.str_pretrained_model == 'ResNet':
            return resnet.preprocess_input
        elif self.str_pretrained_model == 'Xception':
            return xception.preprocess_input
        elif self.str_pretrained_model == 'MobileNet':
            return mobnet.preprocess_input
        elif self.str_pretrained_model == 'IncRes':
            return incres.preprocess_input
        elif self.str_pretrained_model == 'NasNet':
            return nasnet.preprocess_input
        elif self.str_pretrained_model == 'VGG16':
            return vgg.preprocess_input
        elif self.str_pretrained_model == 'MobNet2':
            return mobnet2.preprocess_input
        elif self.str_pretrained_model == 'ResNet2':
            return resnet2.preprocess_input
        else:
            return denseNet.preprocess_input

    def decode_predictions(self):

        if self.str_pretrained_model == 'Inception':
            return inception.decode_predictions
        elif (self.str_pretrained_model == 'EfficientNetB4') or (self.str_pretrained_model == 'EfficientNetB2'):
            return efficientnet.decode_predictions
        elif self.str_pretrained_model == 'ResNet':
            return resnet.decode_predictions
        elif self.str_pretrained_model == 'Xception':
            return xception.decode_predictions
        elif self.str_pretrained_model == 'MobileNet':
            return mobnet.decode_predictions
        elif self.str_pretrained_model == 'IncRes':
            return incres.decode_predictions
        elif self.str_pretrained_model == 'NasNet':
            return nasnet.decode_predictions
        elif self.str_pretrained_model == 'VGG16':
            return vgg.decode_predictions
        elif self.str_pretrained_model == 'MobNet2':
            return mobnet2.decode_predictions
        elif self.str_pretrained_model == 'ResNet2':
            return resnet2.decode_predictions
        else:
            return denseNet.decode_predictions

    def build_model(self):
        # Model building
        inputs = keras.Input(shape=self.input_shape)
        x = self.pretrained_model(inputs, training=self.bln_training)
        x = keras.layers.GlobalMaxPooling2D()(x)
        x = keras.layers.Dropout(self.flt_dropout)(x)  # Regularize with dropout
        x = keras.layers.Dense(4, kernel_regularizer=regularizers.L1L2(l1=1e-2, l2=1e-2),
                               bias_regularizer=regularizers.L2(1e-2),
                               activity_regularizer=regularizers.L2(5e-5), activation='linear')(x)
        x = keras.layers.Activation(activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)
        model.summary()

        return model

    def compile_model(self):
        # Model compiling
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.flt_initial_learning_rate,
            decay_steps=1000,
            decay_rate=1)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  # Low learning rate
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=[keras.metrics.BinaryAccuracy()])

    def fit_model(self, arr_img_tns_train, arr_y_label_train, arr_img_tns_val, arr_y_label_val,
                  patience=10, batch_size=15, epochs=100):
        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, min_delta=0.005,
                           restore_best_weights=True)
        self.compile_model()
        # Model Training
        fitted_model = self.model.fit(arr_img_tns_train, arr_y_label_train,
                                      batch_size=batch_size, epochs=epochs,
                                      validation_data=(arr_img_tns_val, arr_y_label_val), shuffle=True,
                                      callbacks=[es])

        return fitted_model

    def fit_model_fine_tuning(self, arr_img_tns_train, arr_y_label_train, arr_img_tns_val, arr_y_label_val, patience=3,
                              batch_size=15, epochs=15, flt_initial_learning_rate=2e-5):
        self.pretrained_model.trainable = True
        self.flt_initial_learning_rate = flt_initial_learning_rate
        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience,
                           restore_best_weights=True)
        self.compile_model()
        # Model Training
        fitted_model_ft = self.model.fit(arr_img_tns_train, arr_y_label_train,
                                         batch_size=batch_size, epochs=epochs,
                                         validation_data=(arr_img_tns_val, arr_y_label_val), shuffle=True,
                                         callbacks=[es])
        return fitted_model_ft
