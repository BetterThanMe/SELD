from prepare_data.get_data import get_data
from prepare_model.get_model import get_model
from parameter import get_params
from prepare_train.train_class import TrainTest, TrainMultiOutput

import yaml


# GET DATA
params = get_params('4')
data_train = get_data(kind_data='train', params=params)
data_valid = get_data(kind_data='valid', params=params)

# GET MODEL
model = get_model(model_name='seldv0', input_shape=[600, 64, 7], params=params)

# TRAINING
with open('config/hyper_parameter.yaml', 'r') as f:
    hyper_parameter = yaml.load(f, Loader=yaml.FullLoader)
for key, value in hyper_parameter.items():
    print(f"{key:20} {value}")

# train_machine = TrainMultiOutput('Two_tasks', data_train, data_valid, model,
#                                  learning_rate=float(hyper_parameter['learning_rate']),
#                                  weight_decay=float(hyper_parameter['weight_decay']),
#                                  number_steps=hyper_parameter['number_steps'],
#                                  batch_size_train=hyper_parameter['batch_size_train'],
#                                  batch_size_valid=hyper_parameter['batch_size_valid'],
#                                  interval_valid=hyper_parameter['interval_valid'])
train_machine = TrainTest('Two_tasks', data_train, model, num_steps=20, batch_size=16)
train_machine.train()
