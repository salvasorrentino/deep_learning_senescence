import pandas as pd
import numpy as np
import tensorflow as tf
tf.random.set_seed(48)
from sklearn.model_selection import train_test_split
from Scripts import grad_cam
from Scripts import utils
from Scripts.transfer_learning_models import TransferLearningModel
import matplotlib
matplotlib.use('Qt5Agg')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

lst_nn = ['Inception', 'EfficientNetB4', 'ResNet', 'DenseNet', 'MobileNet', 'IncRes', 'Xception']

dct_train = pd.read_pickle(r'C:\Users\AI\OneDrive - Politecnico di Milano\Desktop\Deep Learning'
                           r' Senescence\Codice_GitHub\dct_train.pickle')
dct_test = pd.read_pickle(r'C:\Users\AI\OneDrive - Politecnico di Milano\Desktop\Deep Learning'
                           r' Senescence\Codice_GitHub\dct_test.pickle')

lst_accuracy = []
lst_precision = []
lst_recall = []
dct_y_probas = {}
dct_pred_probas = {}
dct_heatmap = {}

for i in range(9):

    tf.keras.backend.clear_session()
    dct_y_probas[f'{i}'] = {}
    dct_heatmap[f'{i}'] = {}
    obj_data_aug = utils.DataAugmentation()
    arr_img_tns_train_val = dct_train['train_data']
    arr_y_label_train_val = dct_train['train_label']
    arr_img_tns_test = dct_test['test_data']
    arr_y_label_test = dct_test['test_label']
    arr_img_tns_test_res = np.copy(utils.rescaling_array(arr_img_tns_test))

    for str_nn in lst_nn:

        arr_img_tns_train, arr_img_tns_val, arr_y_label_train, arr_y_label_val = \
            train_test_split(arr_img_tns_train_val, arr_y_label_train_val, test_size=0.25,
                             stratify=arr_y_label_train_val, shuffle=True)
        arr_img_tns_train_aug, arr_y_label_train_aug = obj_data_aug.data_augmented(arr_img_tns_train,
                                                                                   arr_y_label_train[:, 0],
                                                                                   11, [0, 1])

        arr_img_tns_val_aug, arr_y_label_val_aug = obj_data_aug.data_augmented(arr_img_tns_val, arr_y_label_val[:, 0],
                                                                               11, [0, 1])
        arr_y_label_train_aug = arr_y_label_train_aug.astype('float64')
        arr_y_label_val_aug = arr_y_label_val_aug.astype('float64')

        print(i, str_nn)

        obj_transf = TransferLearningModel(str_nn)

        arr_img_tns_test_res_copy = np.copy(arr_img_tns_test_res)
        arr_img_tns_train_aug_res = obj_transf.preprocess_input()(utils.rescaling_array(arr_img_tns_train_aug))
        arr_img_tns_val_aug_res = obj_transf.preprocess_input()(utils.rescaling_array(arr_img_tns_val_aug))
        arr_img_tns_test_prep = obj_transf.preprocess_input()(arr_img_tns_test_res_copy)

        # Dataset shuffling in order to divide the dataset in a traind and validation dataset

        fitted_model = obj_transf.fit_model(arr_img_tns_train_aug_res, arr_y_label_train_aug, arr_img_tns_val_aug_res,
                                            arr_y_label_val_aug, patience=20, batch_size=20, epochs=200)
        fitted_model_ft = obj_transf.fit_model_fine_tuning(arr_img_tns_train_aug_res, arr_y_label_train_aug,
                                                           arr_img_tns_val_aug_res, arr_y_label_val_aug, patience=15,
                                                           batch_size=20, epochs=200)
        dct_y_probas[f'{i}'][f'{str_nn}'] = obj_transf.model.predict(arr_img_tns_test_prep)

        obj_transf.model.layers[-1].activation = None

        # Print what the top predicted class is
        preds = obj_transf.model.predict(arr_img_tns_test_prep)

        # Generate class activation heatmap
        dct_heatmap[f'{i}'][f'{str_nn}'] = {}
        for k in range(len(arr_img_tns_test_prep)):
            heatmap = grad_cam.make_gradcam_heatmap(np.expand_dims(arr_img_tns_test_prep[k], axis=0), obj_transf.model, -6)
            jet_heatmap = grad_cam.save_and_display_gradcam(arr_img_tns_test_res[k], heatmap)
            dct_heatmap[f'{i}'][f'{str_nn}'][f'{k}'] = {'heatmap': jet_heatmap, 'arr_img_test': arr_img_tns_test_res[k],
                                                        'arr_y_label': arr_y_label_test}

        tf.keras.backend.clear_session()
    pred_probas = sum([1./len(lst_nn)*probs for probs in dct_y_probas[f'{i}'].values()])
    dct_pred_probas[f'{i}'] = pred_probas
    y_classes = np.where(pred_probas > 0.5, 1, 0)
    arr_y_label_test_flt = arr_y_label_test[:, 0].astype('float64')

    # Accuracy calculation
    obj_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    obj_acc.update_state(arr_y_label_test_flt, pred_probas)
    flt_accuracy = obj_acc.result().numpy()
    lst_accuracy.append(flt_accuracy)

    # Precision calculation
    obj_prec = tf.keras.metrics.Precision(thresholds=0.5)
    obj_prec.update_state(arr_y_label_test_flt, pred_probas)
    flt_precision = obj_prec.result().numpy()
    lst_precision.append(flt_precision)

    # Recall calculation
    obj_rec = tf.keras.metrics.Recall(thresholds=0.5)
    obj_rec.update_state(arr_y_label_test_flt, pred_probas)
    flt_rec = obj_rec.result().numpy()
    lst_recall.append(flt_rec)
