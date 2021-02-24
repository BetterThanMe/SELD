import pathlib
import numpy as np
import sys
import tensorflow as tf
sys.path.append('./prepare_data')
from augmentation import augment_spec
from abc import ABC, abstractmethod


class GetData(ABC):
    def __init__(self, data_path, label_path, is_shuffle=False, spec_extension=None, is_sort=True):
        """
        :type data_path: str, full direction for only data folder
        :type label_path: str, full direction for only label folder
        :type is_shuffle: bool, shuffle data and label or not
        :type spec_extension: str, extension for data and label file. eg: npy, txt,...
        """
        self.dataPath = pathlib.Path(data_path)
        self.labelPath = pathlib.Path(label_path)
        if not self.dataPath.is_dir():
            raise Exception('No direction exist: ' + data_path)
        if not self.labelPath.is_dir():
            raise Exception('No direction exist: ' + label_path)

        self.is_shuffle = is_shuffle
        self.extension = spec_extension
        self.is_sort = is_sort
        super().__init__()

    @abstractmethod
    def readFileName(self):
        """
        Read all files the data and label folder(even nest folder)
        *Note: data and its label must have the same name or one is part of the other name.
            eg: data file: sound_file_fold1.npy, with label file: file_fold1.npy
        :return: a list of tuple(data file name, corresponding label file name)
        """
        if self.extension is not None:
            data_files = list(self.dataPath.rglob('*.' + self.extension))
            label_files = list(self.labelPath.rglob('*.' + self.extension))
            if len(data_files) == 0:
                raise Exception('This Folder does not contain this extension file: ' + self.extension)
        else:
            data_files = list(self.dataPath.rglob('*.*'))
            label_files = list(self.labelPath.rglob('*.*'))
            if len(data_files) == 0:
                raise Exception('This Folder is empty')

        shuffle_files = []
        if self.is_shuffle:
            permutation = np.random.permutation(len(data_files))
        else:
            permutation = np.arange(len(data_files))

        for i in permutation:
            for label in label_files:
                if data_files[i].name in label.name or label.name in data_files[i].name:
                    shuffle_files.append((data_files[i], label))
                    break

        return shuffle_files

    def preprocessing(self):
        """
        Preprocessing data, not necessary
        :return: list of tuple(data, label)
        """
        pass

    @abstractmethod
    def load(self):
        """
        :return: object tf.data.Dataset
        """
        pass


class GetDataMel(GetData):
    def __init__(self, data_path, label_path, num_cat=14, seq_len=600, is_shuffle=False, spec_extension=None,
                 folds=None, batch_size=None, is_sort=True, is_augment=False, is_combine=False, is_custom=False):
        """
        :type num_cat: int, number of sound categories
        :type is_custom: bool, True: using custom dataset object ~ using processing_modify func, False: normal
        """
        super(GetDataMel, self).__init__(data_path, label_path, is_shuffle=is_shuffle, spec_extension=spec_extension,
                                         is_sort=is_sort)
        self._num_cat = num_cat
        self._seq_len = seq_len
        self._num_dim = 64
        self._data_label = None
        self._resolution = 5
        if folds is None:
            folds = [3, 4, 5, 6]
        self.folds = folds
        self._batch_size = batch_size
        self._is_augment = is_augment
        self._is_combine = is_combine
        self._is_custom = is_custom

    def readFileName(self):
        # get right fold file
        double_files = super(GetDataMel, self).readFileName()
        files = []
        for double_file in double_files:
            for fold_index in self.folds:
                if 'fold'+str(fold_index) in double_file[0].name:
                    files.append((str(double_file[0]), str(double_file[1])))
        if self.is_sort:
            files = np.array(files)
            files = np.sort(files, axis=0, kind='mergesort')
            print('files are sorted')
        return files

    def readFile(self):
        files = self.readFileName()
        # read the file in list file name in files
        data_label = []
        count = 0
        for data_file, label_file in files:
            if not self._is_combine:
                data = np.load(data_file)
            else:
                data_foa = np.load(data_file.replace('mic', 'foa'))
                data_mic = np.load(data_file.replace('foa', 'mic'))
                data = np.concatenate((data_foa, data_mic), axis=-1)
            label = np.load(label_file)
            # there are 2 task in the model -> divide the label in to 2: sed and doa
            sed_label = label[:, :self._num_cat]
            doa_label = label[:, self._num_cat:]
            data_label.append((data, sed_label, doa_label))
            count += 1
            print(f'\rReading file {int(float(count)/len(files)*100):d}%', end='')
        self._data_label = data_label
        return data_label

    def preprocessing(self):
        data_preprocessing = []
        sed_labels = []
        doa_labels = []
        length = len(self._data_label)
        for count, (data, sed_label, doa_label) in enumerate(self._data_label):
            data = np.reshape(data, [-1, self._num_dim, data.shape[1] // self._num_dim])
            # reposition into foa 4 + mic 4 + foa_intensity 3 + mic_gcc 6
            if self._is_combine:
                data_separate = np.split(data, [4, 7, 11], axis=-1)
                data = np.concatenate((data_separate[0], data_separate[2], data_separate[1], data_separate[3]), axis=-1)
            data_list = np.split(data, data.shape[0] // self._seq_len)
            sed_list = np.split(sed_label, data.shape[0] // self._seq_len)
            doa_list = np.split(doa_label, data.shape[0] // self._seq_len)
            for index, part in enumerate(data_list):
                if self._is_augment:
                    part = augment_spec(np.expand_dims(part, axis=0))
                data_preprocessing.append(part)
                sed_labels.append(sed_list[index])
                doa_labels.append(doa_list[index])

            print(f'\rPreprocessing data {int(float(count + 1) / length * 100):d}%', end='')
        return (np.array(data_preprocessing, dtype=np.float32), np.array(sed_labels, dtype=np.float32),
                np.array(doa_labels, dtype=np.float32))

    def preprocessing_modify(self):
        data_preprocessing = []
        sed_labels = []
        doa_labels = []
        length = len(self._data_label)
        for count, (data, sed_label, doa_label) in enumerate(self._data_label):
            data = np.reshape(data, [-1, self._seq_len, self._num_dim, data.shape[1] // self._num_dim])
            sed_label = np.split(sed_label, self._resolution)
            doa_label = np.split(doa_label, self._resolution)
            data_preprocessing.append(data)
            sed_labels.append(sed_label)
            doa_labels.append(doa_label)
            print(f'\rPreprocessing data {int(float(count + 1) / length * 100):d}%', end='')

        return (np.array(data_preprocessing, dtype=np.float32), np.array(sed_labels, dtype=np.float32),
                np.array(doa_labels, dtype=np.float32))

    def load(self):
        _ = self.readFile()
        if not self._is_custom:
            data_label = self.preprocessing()
        else:
            data_label = self.preprocessing_modify()

        if isinstance(self._batch_size, int):
            dataset = tf.data.Dataset.from_tensor_slices(data_label)
            dataset = dataset.batch(self._batch_size)
            return dataset
        else:
            return data_label
