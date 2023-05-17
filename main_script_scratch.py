import pandas as pd
import numpy as np
import sklearn.metrics
import tensorflow as tf
tf.random.set_seed(54)
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.models import load_model
from keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from Scripts import grad_cam
from Scripts import utils
from Scripts.transfer_learning_models import TransferLearningModel
from Scripts.model_from_scratch import CNN_model
import matplotlib
import time
matplotlib.use('Qt5Agg')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# --------- Import dataset -----------------
dct_train = pd.read_pickle(r'C:\Users\AI\OneDrive - Politecnico di Milano\Desktop\Deep Learning'
                           r' Senescence\Codice_GitHub\dct_train.pickle')
dct_test = pd.read_pickle(r'C:\Users\AI\OneDrive - Politecnico di Milano\Desktop\Deep Learning'
                           r' Senescence\Codice_GitHub\dct_test.pickle')
arr_img_tns_test = dct_test['test_data']
arr_y_label_test = dct_test['test_label']
arr_img_tns_test_res = np.copy(utils.rescaling_array(arr_img_tns_test))

lst_accuracy = []
lst_precision = []
lst_recall = []
dct_y_probas = {}
dct_pred_probas = {}

for i in range(16):
    tf.keras.backend.clear_session()
    dct_y_probas[f'{i}'] = {}
    obj_data_aug = utils.DataAugmentation()
    arr_img_tns_train_val = dct_train['train_data']
    arr_y_label_train_val = dct_train['train_label']

    arr_img_tns_train, arr_img_tns_val, arr_y_label_train, arr_y_label_val = \
        train_test_split(arr_img_tns_train_val, arr_y_label_train_val, test_size=0.25,
                         stratify=arr_y_label_train_val, shuffle=True)

    arr_img_tns_train_aug, arr_y_label_train_aug = obj_data_aug.data_augmented(arr_img_tns_train, arr_y_label_train[:, 0],
                                                                           11, [0, 1])

    arr_img_tns_val_aug, arr_y_label_val_aug = obj_data_aug.data_augmented(arr_img_tns_val, arr_y_label_val[:, 0],
                                                                       11, [0, 1])
    arr_y_label_train_aug = arr_y_label_train_aug.astype('float64')
    arr_y_label_val_aug = arr_y_label_val_aug.astype('float64')

    arr_img_tns_test_res_copy = np.copy(arr_img_tns_test_res)
    arr_img_tns_train_aug_res_copy = np.copy(utils.rescaling_array(arr_img_tns_train_aug))
    arr_img_tns_val_aug_res_copy = np.copy(utils.rescaling_array(arr_img_tns_val_aug))
    
    # Shuffling dataset 
    model_fit = CNN_model(arr_img_tns_train_aug_res_copy, arr_y_label_train_aug, arr_img_tns_val_aug_res_copy,
                          arr_y_label_val_aug)
    pred_probas = model_fit.predict(arr_img_tns_test_res_copy[:, :, :, :2]/255)

    tf.keras.backend.clear_session()

    dct_pred_probas[f'{i}'] = pred_probas
    y_classes = np.where(pred_probas > 0.5, 1, 0)
    arr_y_label_test_flt = arr_y_label_test[:, 0].astype('float64')
    
    #accuracy
    obj_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    obj_acc.update_state(arr_y_label_test_flt, pred_probas)
    flt_accuracy = obj_acc.result().numpy()
    lst_accuracy.append(flt_accuracy)
    print(arr_y_label_test)
    print(flt_accuracy, lst_accuracy)
    print(pred_probas)

    # precision
    obj_prec = tf.keras.metrics.Precision(thresholds=0.5)
    obj_prec.update_state(arr_y_label_test_flt, pred_probas)
    flt_precision = obj_prec.result().numpy()
    lst_precision.append(flt_precision)

    # rcall
    obj_rec = tf.keras.metrics.Recall(thresholds=0.5)
    obj_rec.update_state(arr_y_label_test_flt, pred_probas)
    flt_rec = obj_rec.result().numpy()
    lst_recall.append(flt_rec)
    dct_train_test[f'{i}'] = {'train': arr_img_tns_train, 'val': arr_img_tns_val, 'test': arr_img_tns_test,
                              'train_lab': arr_y_label_train, 'test_lab': arr_y_label_test, 'val_lab': arr_y_label_val}
