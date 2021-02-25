import socket
import sys
import pathlib
sys.path.append('./prepare_data')
from get_data_class import GetDataMel
from dataregulator import DataRegulator


def get_data(kind_data='train', params=None, fold_set=None):
    data_dir = None
    if socket.gethostname() == 'minh':
        data_dir = "/home/ad/PycharmProjects/Sound_processing/venv/m/data"
    elif socket.gethostname() == 'aiotlab':
        data_dir = "/mnt/disk1/minh/2021/data"
    elif socket.gethostname() == 'ubuntu':
        data_dir = "/mnt/disk2/minh/data"
    data_dir = pathlib.Path(data_dir)

    def get_train_data(fold_set=None):
        if fold_set is not None:
            train_split = fold_set
        else:
            train_split = [3, 4, 5, 6]

        feat_dir = [data_dir.joinpath('feat_label').joinpath('foa_dev_norm'),
                    data_dir.joinpath('feat_label_acs').joinpath('foa_dev_norm'),
                    data_dir.joinpath('feat_label_overlap_ver2').joinpath('foa_dev_norm')]
        label_dir = [data_dir.joinpath('feat_label').joinpath('foa_dev_label'),
                     data_dir.joinpath('feat_label_acs').joinpath('foa_dev_label'),
                     data_dir.joinpath('feat_label_overlap_ver2').joinpath('foa_dev_label')]

        data_gen = DataRegulator(train_split, label_dir, feat_dir, seq_len=600, seq_hop=5)

        data_gen.load_data()
        print('Loading data train successfully!')
        return data_gen

    def get_valid_data(fold_set=None):
        if fold_set is not None:
            split = fold_set
        else:
            split = [2]
        # get data for validation
        valid_data = GetDataMel(data_path=data_dir.__str__() + '/feat_label/' + params['dataset'] + '_dev_norm',
                                label_path=data_dir.__str__() + '/feat_label/' + params['dataset'] + '_dev_label',
                                batch_size=None, folds=split, is_augment=False)

        dataset_valid = valid_data.load()
        print('Load valid data successfully!')
        return dataset_valid

    if 'train' in kind_data.lower():
        return get_train_data(fold_set)
    else:
        return get_valid_data(fold_set)


# from parameter import get_params
# train_data = get_data('train', get_params('4'))
# print(train_data.data_size)
