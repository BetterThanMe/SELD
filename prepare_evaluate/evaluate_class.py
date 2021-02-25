import tensorflow as tf
import sys
import datetime
import time
sys.path.append('./prepare_evaluate')
from dcase3_validation_library import Dcase3Validation


class MultiOutputTest:
    def __init__(self, model: tf.keras.Model, dataset, model_weight_path, batch_size, file_save_path, fold_set):
        self.model = model
        self.dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size)
        self.weight_model_path = model_weight_path
        self.file_path = file_save_path
        self.fold_set = fold_set

    def __call__(self):
        start_time = time.perf_counter()
        now = datetime.datetime.now()
        # load model
        if self.weight_model_path is not None:
            self.model.load_weights(self.weight_model_path)

        # run metric
        metric_2019, metric_2020, loss = Dcase3Validation(threshold_sed=0.5).validation_seld(self.model, self.dataset,
                                                                                             using_2020=True)

        # print out the valid result
        metrics_string = now.strftime("%Y-%m-%d %H:%M:%S: Model {} with fold: {} \n".format(self.model.name,
                                                                                            self.fold_set))
        metrics_string += '\nMetric 2019: Sed Er = {:.4f} F1 = {:.2f} ~ Doa Er = {:.4f} Fr = {:.4f}' \
                          ''.format(metric_2019[0], metric_2019[1] * 100, metric_2019[2],
                                    metric_2019[3] * 100)
        metrics_string += '\nMetric 2020: Sed Er = {:.4f} F1 = {:.2f} ~ Doa Er = {:.4f} Fr = {:.4f}' \
                          ''.format(metric_2020[0], metric_2020[1] * 100, metric_2020[2],
                                    metric_2020[3] * 100)
        metrics_string += '\nLoss Sed: {:.4f} ~ Doa: {:.4f} ~ Seld: {:.4f}\n' \
                          ''.format(loss[0], loss[1], metric_2020[4])
        metrics_string += '---------------------------------\n\n'
        print(metrics_string + 'Total time: {:.2f}\n'.format(time.perf_counter() - start_time))
        # Save metric result to file
        with open(self.file_path, 'a') as f:
            f.write(metrics_string)
        print(f'Saved in {self.file_path}, have a good day!')

