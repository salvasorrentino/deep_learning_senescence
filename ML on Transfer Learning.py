# Code to reproduce the Hybrid ML/TL Approach

import pandas as pd
import numpy as np
import tensorflow as tf
tf.random.set_seed(48)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from tensorflow import keras
from Scripts import utils
from Scripts.transfer_learning_models import TransferLearningModel
import matplotlib
from sklearn.metrics import roc_auc_score

matplotlib.use('Qt5Agg')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# --------- Dataset Import  ------------------

lst_nn = ['Inception', 'EfficientNetB4', 'ResNet', 'DenseNet', 'MobileNet', 'IncRes', 'Xception']

dct_train = pd.read_pickle(r'C:\Users\AI\OneDrive - Politecnico di Milano\Desktop\Deep Learning'
                           r' Senescence\Codice_GitHub\dct_train.pickle')
dct_test = pd.read_pickle(r'C:\Users\AI\OneDrive - Politecnico di Milano\Desktop\Deep Learning'
                           r' Senescence\Codice_GitHub\dct_test.pickle')
lst_accuracy = []
lst_precision = []
lst_recall = []

tf.keras.backend.clear_session()
obj_data_aug = utils.DataAugmentation()
arr_img_tns_train_val = dct_train['train_data']
arr_y_label_train_val = dct_train['train_label']
arr_img_tns_test = dct_test['test_data']
arr_y_label_test = dct_test['test_label']
arr_img_tns_test_res = np.copy(utils.rescaling_array(arr_img_tns_test))
dct_pred_probas = {}
dct_class_report = {}
dct_arr_pred = {}
dct_auc = {}
for i in range(20):
    for str_nn in lst_nn:
        tf.keras.backend.clear_session()
        arr_img_tns_train_val_aug, arr_y_label_train_val_aug = \
            obj_data_aug.data_augmented(arr_img_tns_train_val, arr_y_label_train_val[:, 0], 1, [0, 1])
        arr_img_tns_train_val_aug = arr_img_tns_train_val_aug.astype('float64')
        arr_y_label_train_val_aug = arr_y_label_train_val_aug.astype('float64')

        print(str_nn)

        obj_transf = TransferLearningModel(str_nn)

        arr_img_tns_test_res_copy = np.copy(arr_img_tns_test_res)
        arr_img_tns_train_val_aug = obj_transf.preprocess_input()(utils.rescaling_array(arr_img_tns_train_val_aug))
        arr_img_tns_test_prep = obj_transf.preprocess_input()(arr_img_tns_test_res_copy)

        inputs = keras.Input(shape=obj_transf.input_shape)
        x = obj_transf.pretrained_model(inputs, training=False)
        x = keras.layers.GlobalMaxPooling2D()(x)
        model = keras.Model(inputs, x)
        a = model(arr_img_tns_train_val_aug)
        a = np.array(a)
        pca = PCA()
        b = pca.fit_transform(a)
        print(sum(pca.explained_variance_ratio_))
        param_grid = {'C': [1, 0.01, 0.1, 10, 100],
                      'gamma': [10, 2000, 100, 1],
                      'kernel': ['poly'],
                      'degree': [2, 3, 4, 5]}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
        # fitting the model for grid search
        grid.fit(b, arr_y_label_train_val_aug)

        from sklearn.metrics import classification_report

        grid_predictions = grid.predict(np.array(pca.transform(model(arr_img_tns_test_prep)))).reshape(-1, 1)

        if str_nn == 'Inception':
            grid_predictions_fin = grid_predictions.copy().reshape(-1, 1)

        else:
            grid_predictions_fin = np.concatenate((grid_predictions_fin, grid_predictions), axis=1)

        tf.keras.backend.clear_session()

    arr_pred = np.where(np.mean(grid_predictions_fin, axis=1) > 0.5, 1, 0).astype('float')
    arr_pred_prob = np.mean(grid_predictions_fin, axis=1)
    flt_acc = np.mean(arr_pred == arr_y_label_test[:, 0])
    class_report = classification_report(arr_y_label_test[:, 0].astype(int), arr_pred, digits=4)
    dct_arr_pred[f'{i}'] = arr_pred
    dct_pred_probas[f'{i}'] = arr_pred_prob
    dct_auc[f'{i}'] = roc_auc_score(arr_y_label_test[:, 0].astype(float), arr_pred_prob)
    dct_class_report[f'{i}'] = class_report
