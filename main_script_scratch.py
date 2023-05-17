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

# --------- Import del dataset di immagini dalla cartella ------------------
dct_train_test = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier '
                                r'Senescence\Cell classifier\magic set\dct_train_test_1114.pickle')
arr_img_tns_test = dct_train_test['3']['test']
arr_y_label_test = dct_train_test['3']['test_lab']
arr_img_tns_test_res = np.copy(utils.rescaling_array(arr_img_tns_test))

lst_accuracy = []
lst_precision = []
lst_recall = []
dct_y_probas = {}
dct_pred_probas = {}
# dct_train_test = {}

for i in range(16):
    tf.keras.backend.clear_session()
    dct_y_probas[f'{i}'] = {}
    obj_data_aug = utils.DataAugmentation()
    # arr_img_tns_train_val, arr_img_tns_test, arr_y_label_train_val, arr_y_label_test = \
    #     train_test_split(arr_img_tns, arr_y_label, test_size=0.26, stratify=arr_y_label, shuffle=True)
    # arr_img_tns_test_res = np.copy(utils.rescaling_array(arr_img_tns_test))
    arr_img_tns_train_val = np.concatenate((dct_train_test['3']['train'], dct_train_test['3']['val']))
    arr_y_label_train_val = np.concatenate((dct_train_test['3']['train_lab'],
                                            dct_train_test['3']['val_lab']))

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
    # Shuffling del dataset in modo da suddividere il dataset in uno di train e di validation
    # preds = CNN(arr_img_tns_test_res_copy[:, :, :, :2], training=False)
    model_fit = CNN_model(arr_img_tns_train_aug_res_copy, arr_y_label_train_aug, arr_img_tns_val_aug_res_copy,
                          arr_y_label_val_aug)
    pred_probas = model_fit.predict(arr_img_tns_test_res_copy[:, :, :, :2]/255)

    tf.keras.backend.clear_session()
    # pred_probas = sum([1./len(lst_nn)*probs for probs in dct_y_probas[f'{i}'].values()])
    dct_pred_probas[f'{i}'] = pred_probas
    y_classes = np.where(pred_probas > 0.5, 1, 0)
    arr_y_label_test_flt = arr_y_label_test[:, 0].astype('float64')
    # Calcolo dell'accuracy
    obj_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    obj_acc.update_state(arr_y_label_test_flt, pred_probas)
    flt_accuracy = obj_acc.result().numpy()
    lst_accuracy.append(flt_accuracy)
    print(arr_y_label_test)
    print(flt_accuracy, lst_accuracy)
    print(pred_probas)
    # print(dct_y_probas[f'{i}'].values())

    # Calcolo della precision
    obj_prec = tf.keras.metrics.Precision(thresholds=0.5)
    obj_prec.update_state(arr_y_label_test_flt, pred_probas)
    flt_precision = obj_prec.result().numpy()
    lst_precision.append(flt_precision)

    # Calcolo della recall
    obj_rec = tf.keras.metrics.Recall(thresholds=0.5)
    obj_rec.update_state(arr_y_label_test_flt, pred_probas)
    flt_rec = obj_rec.result().numpy()
    lst_recall.append(flt_rec)
    dct_train_test[f'{i}'] = {'train': arr_img_tns_train, 'val': arr_img_tns_val, 'test': arr_img_tns_test,
                              'train_lab': arr_y_label_train, 'test_lab': arr_y_label_test, 'val_lab': arr_y_label_val}

pd.to_pickle(dct_train_test, r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier'
                                r' Senescence\performance rete from scratch\dct_train_test_1223.pickle')
pd.to_pickle(dct_pred_probas, r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier '
                                 r'Senescence\performance rete from scratch\dct_pred_probas_1223.pickle')
pd.to_pickle(lst_accuracy, r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier'
                              r' Senescence\performance rete from scratch\lst_accuracy_1223.pickle')
pd.to_pickle(lst_precision, r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier '
                               r'Senescence\performance rete from scratch\lst_precision_1223.pickle')
pd.to_pickle(lst_recall, r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier'
                            r' Senescence\performance rete from scratch\lst_recall_1223.pickle')
pd.to_pickle(dct_y_probas, r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier '
                              r'Senescence\performance rete from scratch\dct_y_probas_1223.pickle')


dct_train_test = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier'
                                r' Senescence\performance rete from scratch\dct_train_test_1223.pickle')
dct_pred_probas = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier '
                                 r'Senescence\performance rete from scratch\dct_pred_probas_1223.pickle')
lst_accuracy = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier'
                              r' Senescence\performance rete from scratch\lst_accuracy_1223.pickle')
lst_precision = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier '
                               r'Senescence\performance rete from scratch\lst_precision_1223.pickle')
lst_recall = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier'
                            r' Senescence\performance rete from scratch\lst_recall_1223.pickle')
dct_y_probas = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\NN Classifier '
                              r'Senescence\performance rete from scratch\dct_y_probas_1223.pickle')

flt_mean = np.mean(np.array(lst_f1))
flt_std = np.std(np.array(lst_f1))

lst_f1 = [2*a*b/(a+b) for a, b in zip(lst_precision, lst_recall)]
