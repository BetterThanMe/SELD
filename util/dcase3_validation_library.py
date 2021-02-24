"""
Provide metric validation functions for Sound event detection and Direction of Audio 2019 and 2020
"""
from parameter import get_params
from metrics.SELD_evaluation_metrics import SELDMetrics, early_stopping_metric
from metrics.evaluation_metrics import compute_doa_scores_regr_xyz, compute_sed_scores
from cls_feature_class import FeatureClass

import tensorflow as tf
import tensorflow.keras.backend as K


# ---Custom losses function
def custom_mse(y_true, y_pred):
    # calculating squared difference between target and predicted values
    loss = K.square(y_pred - y_true)  # (batch_size, 2)
    # summing all dimensions
    loss = K.mean(loss)
    return loss


def mse_loss_mask(y_true, y_pred):
    gt = y_true[:, :, 14:]
    mask = y_true[:, :, :14]
    mask = K.concatenate((mask, mask, mask), axis=-1)
    doa_loss = K.square(y_pred - gt)
    doa_loss = tf.multiply(doa_loss, mask)  # mask here
    doa_loss = K.sum(doa_loss) / K.sum(mask)  # mean in all dimensions
    return doa_loss


class Dcase3Validation:
    def __init__(self, threshold_doa=20, threshold_sed=0.5):
        """
        :return metrics for doa, sed or seld, 2019 or 2020
        :type threshold_sed: float in (0,1), classify threshold
        :type threshold_doa: float in (0, 180), maximum threshold error for doa
        """
        self._params = get_params("4")
        self.feat_cls = FeatureClass(self._params)
        self._threshold_sed = threshold_sed
        self._threshold_doa = threshold_doa

    def sed_2019(self, sed_predict, sed_gt):
        """
        calculate validation for Sound event detection
        :param sed_predict: shape(Batch_size, frames_per_files, number_classes)
        :param sed_gt: shape = shape_predict
        :return: metrics[sed_error, sed_f1_score]
        """
        # reshape in from 3D to 2D
        sed_predict = tf.reshape(sed_predict, shape=[-1, sed_predict.shape[-1]])
        sed_gt = tf.reshape(sed_gt, shape=[-1, sed_gt.shape[-1]])
        metric_sed = compute_sed_scores(sed_predict.numpy() > self._threshold_sed, sed_gt.numpy() > self._threshold_sed,
                                        self.feat_cls.nb_frames_1s())
        return metric_sed

    def validation_sed(self, model, dataset_valid):
        """
        :param model: model sed
        :param dataset_valid: (mel, sed_gt, doa_gt) with prefixed batch size
        :return: metrics[sed_error, sed_f1_score], loss
        """
        @tf.function
        def testStep(inputs):
            return model(inputs, training=False)

        metric = [0., 0.]
        loss = 0.
        step = 0
        for step, (mel, sed_gt, doa_gt) in enumerate(dataset_valid):
            sed_predict = testStep(mel)

            # sed 2019
            sed_metric = self.sed_2019(sed_predict, sed_gt)
            metric[0] += sed_metric[0]
            metric[1] += sed_metric[1]
            loss = loss + custom_mse(sed_gt, sed_predict)

        metric = [x / (step + 1) for x in metric]
        loss = loss / (step + 1)

        return metric, loss

    def doa_2019(self, doa_predict, doa_gt, sed_predict, sed_gt):
        """
        calculate validation for Direction of audio
        :param sed_predict: shape(Batch_size, frames_per_files, number_classes)
        :param sed_gt: shape = shape_predict
        :param doa_predict: shape(Batch_size, frames_per_files, number_classes * number_dimension) dimension: x, y, z
        :param doa_gt: shape = shape_predict
        :return: metrics[doa_error, f_recall]
        """
        # reshape in from 3D to 2D
        sed_predict = tf.reshape(sed_predict, shape=[-1, sed_predict.shape[-1]])
        sed_gt = tf.reshape(sed_gt, shape=[-1, sed_gt.shape[-1]])
        doa_predict = tf.reshape(doa_predict, shape=[-1, doa_predict.shape[-1]])
        doa_gt = tf.reshape(doa_gt, shape=[-1, doa_gt.shape[-1]])
        metric_doa = compute_doa_scores_regr_xyz(doa_predict.numpy(), doa_gt.numpy(),
                                                 sed_predict.numpy() > self._threshold_sed,
                                                 sed_gt.numpy() > self._threshold_sed)
        return metric_doa[:2]

    def validation_doa(self, model, dataset_valid):
        """
        :param model: model doa
        :param dataset_valid: (mel, sed_gt, doa_gt) with prefixed batch size
        :return: metrics[doa_error, sed_f1_score], loss
        """

        @tf.function
        def testStep(inputs):
            return model(inputs, training=False)

        metric = [0., 0.]
        loss = 0.
        step = 0
        for step, (mel, sed_gt, doa_gt) in enumerate(dataset_valid):
            doa_predict = testStep(mel)

            # doa 2019
            doa_metric = self.doa_2019(doa_predict, doa_gt, sed_gt, sed_gt)
            metric[0] += doa_metric[0]
            metric[1] += doa_metric[1]
            loss = loss + mse_loss_mask(tf.concat((sed_gt, doa_gt), axis=-1), doa_predict)

        metric = [x / (step + 1) for x in metric]
        loss = loss / (step + 1)

        return metric, loss

    def seld_2020(self, sed_predict, sed_gt, doa_predict, doa_gt):
        """
        calculate validation for Detection and Classification of Acoustic Scenes and Events
        :param sed_predict: shape(Batch_size, frames_per_files, number_classes)
        :param sed_gt: shape = shape_predict
        :param doa_predict: shape(Batch_size, frames_per_files, number_classes * number_dimension) dimension: x, y, z
        :param doa_gt: shape = shape_predict
        :return: metrics[sed_error, sed_f1_score, doa_error, sed_f1_score, seld_error]
        """
        # reshape from 3D into 2D
        sed_predict = tf.reshape(sed_predict, shape=[-1, sed_predict.shape[-1]])
        sed_gt = tf.reshape(sed_gt, shape=[-1, sed_gt.shape[-1]])
        doa_predict = tf.reshape(doa_predict, shape=[-1, doa_predict.shape[-1]])
        doa_gt = tf.reshape(doa_gt, shape=[-1, doa_gt.shape[-1]])

        metric_2020 = [0., 0., 0., 0., 0.]
        cls_new_metric = SELDMetrics(nb_classes=14, doa_threshold=self._threshold_doa)

        pred_dict = self.feat_cls.regression_label_format_to_output_format(sed_predict > self._threshold_sed, doa_predict)
        gt_dict = self.feat_cls.regression_label_format_to_output_format(sed_gt > self._threshold_sed, doa_gt)

        pred_blocks_dict = self.feat_cls.segment_labels(pred_dict, sed_predict.shape[0])
        gt_blocks_dict = self.feat_cls.segment_labels(gt_dict, sed_gt.shape[0])

        cls_new_metric.update_seld_scores_xyz(pred_blocks_dict, gt_blocks_dict)
        new_metric = cls_new_metric.compute_seld_scores()
        new_seld_metric = early_stopping_metric(new_metric[:2], new_metric[2:])

        metric_2020[0] = new_metric[0]
        metric_2020[1] = new_metric[1]
        metric_2020[2] = new_metric[2]
        metric_2020[3] = new_metric[3]
        metric_2020[4] = new_seld_metric
        return metric_2020

    def validation_seld(self, model, dataset_valid, using_2020=False):
        """
        :param using_2020: using metric of 2020 or not
        :param model: model seld
        :param dataset_valid: (mel, sed_gt, doa_gt) with prefixed batch size
        :return: metric_2019, metric_2020, loss[sed, doa]
        """
        @tf.function
        def testStep(inputs):
            return model(inputs, training=False)

        metric_2019 = [0., 0., 0., 0.]
        metric_2020 = [0., 0., 0., 0., 0.]
        loss = [0., 0.]
        step = 0
        for step, (mel, sed_gt, doa_gt) in enumerate(dataset_valid):

            sed_predict, doa_predict = testStep(mel)

            # doa 2019
            doa_metric = self.doa_2019(doa_predict, doa_gt, sed_predict, sed_gt)
            metric_2019[2] += doa_metric[0]
            metric_2019[3] += doa_metric[1]
            loss[1] = loss[1] + mse_loss_mask(tf.concat((sed_gt, doa_gt), axis=-1), doa_predict)

            # sed 2019
            sed_metric = self.sed_2019(sed_predict, sed_gt)
            metric_2019[0] += sed_metric[0]
            metric_2019[1] += sed_metric[1]
            loss[0] = loss[0] + custom_mse(sed_gt, sed_predict)

            # seld 2020
            if using_2020:
                seld_metric = self.seld_2020(sed_predict, sed_gt, doa_predict, doa_gt)
                metric_2020 = [x + y for x, y in zip(metric_2020, seld_metric)]

        metric_2019 = [x / (step + 1) for x in metric_2019]
        metric_2020 = [x / (step + 1) for x in metric_2020]
        loss = [x / (step + 1) for x in loss]

        return metric_2019, metric_2020, loss

    def validation(self, model, dataset_valid):
        if 'sed' in model.name.lower():
            return self.validation_sed(model, dataset_valid)
        elif 'doa' in model.name.lower():
            return self.validation_doa(model, dataset_valid)
        else:
            return self.validation_seld(model, dataset_valid)


