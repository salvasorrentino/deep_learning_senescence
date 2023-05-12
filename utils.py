import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings('error')


class Normalization:
    def __init__(self):
        pass

    @staticmethod
    def iqr(arr_input):
        q3, q1 = np.percentile(arr_input, [75, 25])
        iqr = q3 - q1
        return iqr

    def min_no_out(self, arr_img_tns):
        flt_median = np.median(arr_img_tns)
        iqr = self.iqr(arr_img_tns)
        arr_img_tns = arr_img_tns.ravel()
        flt_min = np.min(arr_img_tns[(arr_img_tns > flt_median - 3.5*iqr) & (arr_img_tns < flt_median + 3.5*iqr)])

        return flt_min

    def max_no_out(self, arr_img_tns):
        flt_median = np.median(arr_img_tns)
        iqr = self.iqr(arr_img_tns)
        arr_img_tns = arr_img_tns.ravel()
        flt_max = np.max(arr_img_tns[(arr_img_tns > flt_median - 3.5 * iqr) & (arr_img_tns < flt_median + 3.5 * iqr)])

        return flt_max


class DataAugmentation:
    def __init__(self, rotation_range=45, width_shift_range=0.15,
                 height_shift_range=0.15, zoom_range=0):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range

    def datagen(self):
        datagen = ImageDataGenerator(
            rotation_range=self.rotation_range,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            rescale=1,
            shear_range=0,
            zoom_range=self.zoom_range,
            horizontal_flip=True,
            fill_mode='constant',
            cval=0)

        return datagen

    def data_augmented(self, arr_img, arr_y_label, int_num_augmented, lst_class):
        datagen = self.datagen()
        for int_class in lst_class:
            arr_idx_class = np.where(arr_y_label == int_class)[0]
            for int_it in range(int_num_augmented * len(arr_img[arr_idx_class])):
                it = datagen.flow(arr_img[arr_idx_class], batch_size=1)
                arr_img = np.concatenate((arr_img, it.next()))
            arr_y_label = np.concatenate((arr_y_label, np.array([int_class] * (int_it + 1))))

        return arr_img, arr_y_label


# Input data normalization between [0, 255]
def rescaling_array(arr_img_tns):
    obj_outliers = Normalization()
    arr_img_tns_2 = np.empty_like(arr_img_tns)
    for i in range(arr_img_tns.shape[3]):
        arr_img_tns_2[:, :, :, i] = (255*((arr_img_tns[:, :, :, i] - obj_outliers.min_no_out(arr_img_tns[:, :, :, i]))/ \
                                  (obj_outliers.max_no_out(arr_img_tns[:, :, :, i]) -
                                   obj_outliers.min_no_out(arr_img_tns[:, :, :, i])))).astype('int')
        arr_img_tns_2[:, :, :, i] = np.where(arr_img_tns_2[:, :, :, i] > 255, 255, arr_img_tns_2[:, :, :, i])
        arr_img_tns_2[:, :, :, i] = np.where(arr_img_tns_2[:, :, :, i] < 0, 0, arr_img_tns_2[:, :, :, i])

    return arr_img_tns_2


