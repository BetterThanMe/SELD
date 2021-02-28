import socket
import sys
from tensorflow.keras.utils import plot_model
import os
import time
import tensorflow as tf
import tensorflow_addons as tfa
import pathlib
from math import ceil
sys.path.insert(0, './prepare_train')
from train_library import FindLearningRate, SetUpFileName, TrainFlow, OneCycle
from dcase3_validation_library import Dcase3Validation, custom_mse, mse_loss_mask
from augmentation import augment_spec


# TRAIN CLASS FOR FINDING LEARNING RATE
class TrainTest:
    """
    Used to find learning rate
    """
    def __init__(self, name, dataset, model, num_steps, batch_size):
        """
        :param name: name of TrainTest object, not affect anything
        :param dataset: Data_regulator object
        :param model: keras model object
        :param num_steps: number of steps to find learning rate
        :param batch_size:
        """
        self._name = name
        # dataset
        self._dataset = dataset
        self._batch_size = batch_size
        self._model = model
        # directory of Track
        base_dir = None
        if socket.gethostname() == 'minh':
            base_dir = "/home/ad/PycharmProjects/Sound_processing/venv/m/Track"
        elif socket.gethostname() == 'aiotlab':
            base_dir = "/mnt/disk1/minh/2021/Track"
        elif socket.gethostname() == 'ubuntu':
            base_dir = "/mnt/disk2/minh/Track"
        base_dir = pathlib.Path(base_dir)
        if not base_dir.is_dir():
            base_dir.mkdir()
        # Set up file
        self._set_up = SetUpFileName(name=model.name, base_dir=base_dir)
        _, _, _, _, self._hyper_file, self._findLR_path = self._set_up()
        # Plot model to image
        plot_model(model, os.path.join(os.path.dirname(self._hyper_file), model.name + '.png'), show_shapes=True)
        self._findLr = FindLearningRate(1e-5, 1e-1, num_steps, self._name, path=self._findLR_path)
        self._global_step = 0
        self._num_step = num_steps
        self._num_step_per_epoch = ceil(self._dataset.data_size / batch_size)
        # Training control
        self._optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=1e-4)
        # Loss weight of sed + loss weight of doa = 1
        self._loss_weight = 0.5

    @tf.function
    def update_weight(self, mel, sed, doa):
        with tf.GradientTape() as tape:
            sed_output, doa_output = self._model(mel, training=True)
            loss_sed = custom_mse(sed, sed_output)
            loss_doa = mse_loss_mask(tf.concat((sed, doa), axis=-1), doa_output)
            loss = self._loss_weight*loss_sed + (1 - self._loss_weight)*loss_doa
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        return loss, loss_sed, loss_doa

    def train(self):
        try:
            start_time = time.perf_counter()
            while self._global_step < self._num_step:
                self._global_step += 1
                if self._global_step % self._num_step_per_epoch == 0:
                    self._dataset.reset_pointer()
                    self._dataset.shuffle_data()
                mel, sed, doa = self._dataset.next_batch(self._batch_size)
                # setting learning rate
                self._optimizer.lr.assign(self._findLr.cal_lr())
                # random augmenting data
                # if args.augment_data and tf.random.uniform(shape=[], maxval=2, dtype=tf.int32):
                #     mel = augment_spec(mel)

                # cat type from float64 to float32
                mel = tf.cast(mel, dtype=tf.float32)
                sed = tf.cast(sed, dtype=tf.float32)
                doa = tf.cast(doa, dtype=tf.float32)
                # update weight and get loss
                loss, loss_sed, loss_doa = self.update_weight(mel, sed, doa)
                self._findLr.record_loss(loss)
                print('At step {:d} Lr {:.6f}: loss = {:.4f}'.format(self._global_step, self._optimizer.lr.numpy(),
                                                                     loss))
            end_time = time.perf_counter()
            print('Total time: {:.2f} s'.format(end_time - start_time))
        finally:
            self._findLr()


# TRAIN CLASS TO TRAIN BOTH TASK: SED AND DOA
class TrainMultiOutput(TrainFlow):
    """
    This only support training by OneCycle method
    """
    def __init__(self, name, dataset, dataset_valid, model, learning_rate, weight_decay, number_steps, batch_size_train,
                 batch_size_valid, interval_valid):
        """
        :param name: name of the TrainObject - not so important
        :param dataset: dataredulator object
        :param dataset_valid: numpy array
        :param model: model keras object
        :param learning_rate: maximum learning rate for OneCycle train
        :param weight_decay:
        :param number_steps: total steps for train
        :param batch_size_train:
        :param batch_size_valid:
        :param interval_valid: number of step for starting validation
        """
        super().__init__(name, dataset, dataset_valid, model)
        # hyper parameter
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

        # train control
        self._number_steps = number_steps
        self._number_step_per_epoch = ceil(self._dataset.data_size / batch_size_train)
        self.oneCycle = OneCycle(nb=number_steps, max_lr=learning_rate)
        self._batch_size_train = batch_size_train
        self._batch_size_valid = batch_size_valid
        self._current_step = 0

        # data control
        self._dataset_valid = tf.data.Dataset.from_tensor_slices(self._dataset_valid).batch(batch_size_valid)

        # metric
        self._interval_valid = interval_valid
        self.validation_machine = Dcase3Validation(threshold_sed=0.5)
        self._weight_loss = tf.Variable(0.5)  # weight between sed_loss(weight_loss) and doa_loss(1 - weight_loss)
        # Setup metric file
        with open(self._metric_file, 'a') as f:
            f.write(f'\n------------{self._today_times}-------------\n')

        # tracking
        self.loss_writer = {
            'train': {
                'sed': tf.summary.create_file_writer(os.path.join(self._tensorboard_path, 'sed_train_loss')),
                'doa': tf.summary.create_file_writer(os.path.join(self._tensorboard_path, 'doa_train_loss')),
                'total': tf.summary.create_file_writer(os.path.join(self._tensorboard_path, 'total_train_loss'))
            },
            'valid': {
                'sed': tf.summary.create_file_writer(os.path.join(self._tensorboard_path, 'sed_valid_loss')),
                'doa': tf.summary.create_file_writer(os.path.join(self._tensorboard_path, 'doa_valid_loss')),
                'total': tf.summary.create_file_writer(os.path.join(self._tensorboard_path, 'total_valid_loss'))
            },
            'lr': tf.summary.create_file_writer(os.path.join(self._tensorboard_path, 'learning rate'))
        }

        # log info
        self.log_hyper_parameter()

    def log_hyper_parameter(self):
        with open(self.hyper_file, 'a') as f:
            info = 'Name model: {}\n'.format(self.model.name)
            info += 'Max learning rate: {}\n'.format(self._learning_rate)
            info += 'Weight decay: {}\n'.format(self._weight_decay)
            info += 'Number steps: {}\n'.format(self._number_steps)
            info += 'Batch size: {}\n'.format(self._batch_size_train)
            info += 'Interval valid: {}\n'.format(self._interval_valid)
            f.write(info)

    @tf.function
    def updateWeight(self, input_batch, sed_label, doa_label):
        with tf.GradientTape() as tape:
            sed_output, doa_output = self.model(input_batch, training=True)
            sed_loss = custom_mse(sed_label, sed_output)
            doa_loss = mse_loss_mask(tf.concat((sed_label, doa_label), axis=-1), doa_output)
            total_loss = self._weight_loss.value()*sed_loss + (1 - self._weight_loss.value())*doa_loss
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return sed_loss, doa_loss, total_loss

    def validCalculate(self):
        step = self._current_step
        start_time = time.perf_counter()
        metric_2019, metric_2020, loss = self.validation_machine.validation_seld(self.model, self._dataset_valid)
        # calculate total loss
        total_loss = self._weight_loss * loss[0] + (1 - self._weight_loss.value()) * loss[1]
        self.tensorboard('valid', loss[0], loss[1], total_loss, step)
        # print out the valid result
        metrics_string = '\nStep {:d} lr={:.8f}:'.format(step, self.optimizer.lr.numpy())
        metrics_string += '\nMetric 2019: Sed Er = {:.4f} F1 = {:.2f} ~ Doa Er = {:.4f} Fr = {:.4f}' \
                          ''.format(metric_2019[0], metric_2019[1] * 100, metric_2019[2],
                                    metric_2019[3] * 100)
        metrics_string += '\nMetric 2020: Sed Er = {:.4f} F1 = {:.2f} ~ Doa Er = {:.4f} Fr = {:.4f}' \
                          ''.format(metric_2020[0], metric_2020[1] * 100, metric_2020[2],
                                    metric_2020[3] * 100)
        metrics_string += '\nLoss Sed: {:.4f} ~ Doa: {:.4f} ~ Seld: {:.4f}\n' \
                          ''.format(loss[0], loss[1], metric_2020[4])
        print(metrics_string + 'Total time: {:.2f}\n'.format(time.perf_counter() - start_time))
        # Save metric result to file
        with open(self._metric_file, 'a') as f:
            f.write(metrics_string)
        return total_loss

    def tensorboard(self, kind, sed_loss, doa_loss, total_loss, step):
        """
        tracking value of sed, doa, total loss and learning rate every step
        """
        with self.loss_writer[kind]['sed'].as_default():
            tf.summary.scalar('loss', sed_loss, step)
        with self.loss_writer[kind]['doa'].as_default():
            tf.summary.scalar('loss', doa_loss, step)
        with self.loss_writer[kind]['total'].as_default():
            tf.summary.scalar('loss', total_loss, step)
        with self.loss_writer['lr'].as_default():
            tf.summary.scalar('loss', self.optimizer.lr.numpy(), step)

    def train(self):
        try:
            while self._current_step < self._number_steps:
                self._current_step += 1
                start_time = time.perf_counter()
                if self._current_step % self._number_step_per_epoch == 0:
                    self._dataset.reset_pointer()
                    self._dataset.shuffle_data()

                # get data
                input_batch, sed_label, doa_label = self._dataset.next_batch(self._batch_size_train)
                # random apply spec_augment on input data
                if tf.random.uniform(shape=[], maxval=2, dtype=tf.int32):
                    input_batch = augment_spec(input_batch)
                # cast type to float32
                input_batch = tf.cast(input_batch, dtype=tf.float32)
                sed_label = tf.cast(sed_label, dtype=tf.float32)
                doa_label = tf.cast(doa_label, dtype=tf.float32)

                # set learning rate
                self.optimizer.lr.assign(self.oneCycle.calc()[0])
                # update weight and get loss
                sed_loss, doa_loss, total_loss = self.updateWeight(input_batch, sed_label, doa_label)
                # tensorboard
                self.tensorboard('train', sed_loss, doa_loss, total_loss, self._current_step)
                # print the result
                time_train_batch = time.perf_counter() - start_time
                print('Step {:d} - Lr {:.6f}: Loss Sed {:.6f} Doa {:.6f} - {:.2f}'
                      .format(self._current_step, self.optimizer.lr.numpy(), sed_loss, doa_loss, time_train_batch))

                # validation
                if self._current_step % self._interval_valid == 0:
                    total_loss_valid = self.validCalculate()
                    # checkpoint when loss is smaller
                    self.checkPoint.on_epoch_end(self.model, self._current_step, total_loss_valid)

        finally:
            self.finish(next_times=True)
