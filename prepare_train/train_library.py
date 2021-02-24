from abc import ABC, abstractmethod
import numpy as np
import os
from datetime import date, datetime
import matplotlib.pyplot as plt
import pathlib
import csv
import functools
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import shutil


def measureTimePerformance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        value = func(*args, **kwargs)
        total = datetime.now() - start_time
        print(f"Finished {func.__name__} in {total.microseconds:d} ms")
        return value
    return wrapper


# ---learning rate scheduler update by epoch
class Scheduler:
    def __init__(self, learning_rate=2e-4, decay_rate=0.8, warm_up_epoch=10, schedule=None, warm_rate=0.1, **kwargs):
        if schedule is None:
            schedule = [200, 600, 1000]
        elif isinstance(schedule, str):
            seps = ['-', '_', ',', ' ', '~', '!']
            for sep in seps:
                if sep in schedule:
                    break
            if sep == '!':
                schedule = [200, 600, 1000]
                print('Wrong syntax for schedule, eg: 100_200_300, so initialize as default: 200, 600, 1000')
            else:
                schedule = schedule.split(sep)
                schedule = list(map(int, schedule))

        self.schedule = schedule
        self.warm_up_epoch = warm_up_epoch
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.divide_epoch = np.array(schedule)
        self.warm_rate = warm_rate

    def __call__(self, epoch, *args, **kwargs):
        decay = sum(epoch >= self.divide_epoch)
        if epoch <= self.warm_up_epoch:
            return self.learning_rate * self.warm_rate
        return self.learning_rate * np.power(self.decay_rate, decay)


# ---class stopping when Seld score is not improved
class EarlyStoppingAtMinLoss:
    def __init__(self, patience=50):
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        self.stop = False

    def on_epoch_end(self, model, epoch, loss):
        current = loss

        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                print("Restoring model weights from the end of the best epoch.")
                self.stop = True
                model.set_weights(self.best_weights)
                print("Early stopping at epoch {}".format(epoch + 1))
        return self.stop


# ---Save model weight thats has best evaluate at the moment
class Checkpoint:
    def __init__(self, checkpoint_path):
        self.best = np.Inf
        self.current_epoch = 0
        self.best_weight = None
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, model, epoch, loss_valid):
        if np.less(loss_valid, self.best):
            self.best = loss_valid
            self.best_weight = model.get_weights()
            self.current_epoch = epoch
            checkpoint_file = os.path.join(self.checkpoint_path, '{:d}_{:.4f}.tf'.format(epoch, loss_valid))
            model.save_weights(checkpoint_file, save_format='tf')
            print('Best_weight is saved!')

    def save(self, model, step, loss_valid):
        checkpoint_file = os.path.join(self.checkpoint_path, '{:d}_{:.4f}.tf'.format(step, loss_valid))
        model.save_weights(checkpoint_file, save_format='tf')
        print('Current weight is saved!')

    def extra_on_epoch_end(self, model, module_name, epoch, loss_valid):
        if np.less(loss_valid, self.best):
            self.best = loss_valid
            self.best_weight = model.get_weights()
            self.current_epoch = epoch
            checkpoint_file = os.path.join(self.checkpoint_path, 'Total', '{:d}_{:.4f}.tf'.format(epoch, loss_valid))
            model.save_weights(checkpoint_file, save_format='tf')

            checkpoint_file = os.path.join(self.checkpoint_path, module_name, '{:d}_{:.4f}.tf'.format(epoch, loss_valid))
            model.get_layer(module_name).save_weights(checkpoint_file, save_format='tf')
            print('Best_weight is saved!')


class SetUpFileName:
    """
    This class set of the name for name_time.txt, tensorboard_path, checkpoint_path, metric_file...
    based on data_month and name of the model training
    """
    @staticmethod
    def getDir(path: pathlib.Path) -> pathlib.Path:
        if not path.is_dir():
            path.mkdir()
        return path

    def __init__(self, name=None, base_dir=None, is_continue=False):
        """
        :type name: basestring, name of the model
        Track___ModelName___info.txt
                        |___time.txt
                        |___[time]___[tensorboard]
                                 |___[checkpoint]
                                 |___[find_learning_rate]
                                 |___[metric.txt]
                                 |___[info.txt]
        """
        self.name = name
        if base_dir is None:
            self.base_dir = self.getDir(pathlib.Path.cwd().joinpath('Track'))
        else:
            self.base_dir = self.getDir(pathlib.Path(base_dir))

        self.base_dir = self.getDir(self.base_dir.joinpath(self.name))

        self.today = date.today().strftime('%d_%m')
        self.time_file = self.base_dir.joinpath('time.txt')
        self.info_file = self.base_dir.joinpath('info.txt')
        # initial info_file
        if not self.info_file.is_file():
            with open(self.info_file.__str__(), 'w') as f:
                f.write(self.name+' created at '+self.today+'\n')

        self.times = None
        self.hyper_file = None
        self.time_start = datetime.now()
        self._is_continue = is_continue

    def __call__(self, *args, **kwargs):
        # current training times
        if not self.time_file.is_file():
            with open(self.time_file.__str__(), 'w') as f:
                f.write(str(0))
                times = str(0)
        else:
            with open(self.time_file.__str__(), 'r') as f:
                times = f.read()
            if self._is_continue:
                times = str(int(times) - 1)
        self.times = times

        # get today_times
        today_times = self.today + '_' + times
        # get in time folder
        if self.base_dir.joinpath(times).is_dir():
            if not self._is_continue:
                for folder in self.base_dir.joinpath(times).iterdir():
                    # only keep find_learning_rate folder
                    if folder.is_file():
                        folder.unlink()
                    elif folder.name != 'find_learning_rate':
                        shutil.rmtree(str(folder))

        self.base_dir = self.getDir(self.base_dir.joinpath(times))

        # tensorboard path
        tensorboard_dir = self.getDir(self.base_dir.joinpath('tensorboard'))

        # checkpoint_path
        checkpoint_dir = self.getDir(self.base_dir.joinpath('checkpoint'))

        # find_learning
        learning_rate_dir = self.getDir(self.base_dir.joinpath('find_learning_rate'))

        # metric_file
        metrics_file = self.base_dir.joinpath('metrics.txt')

        # hyper-parameter log out
        hyper_dir = self.getDir(self.base_dir.joinpath('info'))
        hyper_file = hyper_dir.joinpath('info.txt')
        self.hyper_file = hyper_file

        with open(self.hyper_file, 'a') as f:
            f.write('\n-------' + today_times + '--------\n')

        return today_times, tensorboard_dir.__str__(), checkpoint_dir.__str__(), metrics_file.__str__(), \
               str(hyper_file), learning_rate_dir

    def setNextTimes(self):
        if self.times is not None:
            with open(self.time_file, 'w') as f:
                f.write(str(int(self.times) + 1))

    def finish(self, next_times=True):
        total_time = datetime.now() - self.time_start
        # only save if it's greater than 5 minutes. ps: in my computer 10s
        if total_time.seconds >= 10 and next_times:
            self.setNextTimes()
            with open(self.hyper_file, 'a') as f:
                f.write('\nTotal time train = {}'.format(total_time.__str__()))
                f.write('\n---------------\n')


class FindLearningRate:
    """
    :return suitable learning rate for model
    """
    def __init__(self, low_bound, up_bound, nb, model_name=None, path=None):
        """
        Find the suitable learning rate - when the loss start to explode
        :type low_bound: float, start at lowest learning rate
        :type up_bound: float, reach the highest learning rate at the end of total iterations
        :type nb: int, total number of iterations in all epochs
        :returns: suitable learning rate; plot the map diagram between lr and loss
        """
        self.low = low_bound
        self.up = up_bound
        self.nb = nb
        self.iteration = 0
        self.losses = []  # record loss
        self.lrs = []  # record lr
        self.lr = 0
        if path is None:
            self.dir = pathlib.Path.cwd().joinpath("LearningRate")
        else:
            self.dir = pathlib.Path(path)

        self.file_lr = self.dir.joinpath("learningRate.txt")
        self.file_plot = self.dir.joinpath("plot.png")
        self.file_csv = self.dir.joinpath("lr_loss.csv")

    def record_loss(self, loss):
        self.losses.append(loss)

    def cal_lr(self, *args, **kwargs):
        self.lr = self.low * (self.up / self.low) ** (float(self.iteration) / self.nb)
        self.iteration += 1
        self.lrs.append(self.lr)
        return self.lr

    def cal_epochs(self, batch_per_epoch):
        return self.nb//batch_per_epoch

    def __call__(self, is_show=False, *args, **kwargs):
        if not self.dir.is_dir():
            self.dir.mkdir()

        if len(self.losses) > 0:
            pre_lr = self.lrs[self.losses.index(min(self.losses))]
            with open(str(self.file_lr), 'w') as f:
                f.write(str(pre_lr))

            fig = plt.figure()
            ax = fig.gca()
            if len(self.lrs) < len(self.losses):
                self.lrs.append(self.lrs[-1])
            if len(self.lrs) > len(self.losses):
                self.losses.append(1.)
            ax.plot(self.lrs, self.losses)
            ax.set_xscale('log')
            fig.savefig(self.file_plot)
            if is_show:
                plt.show()

            with open(self.file_csv, 'w') as f:
                write = csv.writer(f)
                self.losses = np.reshape(self.losses, (-1, 1))
                self.lrs = np.reshape(self.lrs, (-1, 1))
                write.writerows(np.concatenate((self.lrs, self.losses), axis=-1))

            print(pre_lr)


class OneCycle:
    """
    :return learning rate for each iterations apply one cycle policy
    """
    def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt=10, div=10, iteration=0):
        """
        :type nb: int, Total number of iterations including all epochs
        :type max_lr: float, The optimum learning rate. This learning rate will be used as highest
                             learning rate. The learning rate will fluctuate between max_lr to
                             max_lr/div and then (max_lr/div)/div.
        :type momentum_vals: tuple, The maximum and minimum momentum values between which momentum will
                                    fluctuate during cycle.
                                    Default values are (0.95, 0.85)
        :type prcnt: int,   The percentage of cycle length for which we annihilate learning rate
                            way below the lower learnig rate.
                            The default value is 10
        :type div: int, The division factor used to get lower boundary of learning rate. This
                        will be used with max_lr value to decide lower learning rate boundary.
                        This value is also used to decide how much we annihilate the learning
                        rate below lower learning rate.
                        The default value is 10.
        """
        self.nb = nb
        self.div = div
        self.high_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt
        self.iteration = iteration
        self.lrs = []
        self.moms = []
        self.step_len = int(self.nb*(1 - self.prcnt/100)/2)

    def calc(self):
        lr = self.calc_lr()
        mom = self.calc_mom()
        self.iteration += 1
        return (lr, mom)

    def calc_lr(self):
        if self.iteration == 0:
            self.lrs.append(self.high_lr / self.div)
            return self.high_lr / self.div
        elif self.iteration == self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr / self.div)
            return self.high_lr / self.div
        elif self.iteration > 2 * self.step_len:
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            # lr = self.high_lr * ( 1 - 0.99 * ratio)/self.div
            lr = (self.high_lr / self.div) * (1 - ratio * (1 - 1 / self.div))
        elif self.iteration > self.step_len:
            ratio = 1 - (self.iteration - self.step_len) / self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else:
            ratio = self.iteration / self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr

    def calc_mom(self):
        if self.iteration == 0:
            self.moms.append(self.high_mom)
            return self.high_mom
        elif self.iteration == self.nb:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        elif self.iteration > 2 * self.step_len:
            mom = self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration - self.step_len) / self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else:
            ratio = self.iteration / self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom


class TrainFlow(ABC):
    def __init__(self, name, dataset, dataset_valid, model, is_continue=False):
        """
        :type is_continue: bool, continue training or not
        :type model: model keras object, or at least have call function
        :type dataset_valid: dataset object, using for validation
        :type dataset: dataset object, using for training
        :type name: str, name for this training, must be pass for setting name for other files
        """
        self._name = name
        self._dataset = dataset
        self._dataset_valid = dataset_valid
        self.model = model
        self.setUp = SetUpFileName(name=model.name, is_continue=is_continue)

        self._today_times, self._tensorboard_path, self._checkpoint_path, self._metric_file, self._hyper_file, \
        self._findLR_path = self.setUp()

        self.checkPoint = Checkpoint(checkpoint_path=self._checkpoint_path)
        self.earlyStop = EarlyStoppingAtMinLoss(patience=50)
        self._global_step = 0
        plot_model(model, os.path.join(os.path.dirname(self._hyper_file), model.name+'.png'), show_shapes=True)
        # restore model weight and current step if is_continue is true
        if is_continue:
            self.restore()

    def restore(self, checkpoint_dir=None):
        if checkpoint_dir is not None:
            latest = tf.train.latest_checkpoint(checkpoint_dir)
        else:
            latest = tf.train.latest_checkpoint(self._checkpoint_path)

        if latest is not None:
            current_step = int(latest.split('/')[-1].split('_')[0])
            self.model.load_weights(latest)
            self._global_step = current_step - 2
            print('Restore successful')

    def saveState(self):
        # save current weight, fake loss_valid, also save current step
        self.checkPoint.save(self.model, self._global_step, 0.)

    @abstractmethod
    def updateWeight(self):
        pass

    @abstractmethod
    def validCalculate(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def finish(self, next_times=True):
        self.setUp.finish(next_times)

    @property
    def hyper_file(self):
        return self._hyper_file


def limitGpuMemory(gpu_index, memory_limit):
    """
    :param gpu_index: index of gpu use
    :param memory_limit: in gigabyte, eg 4
    :return:
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu_index],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit * 1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


