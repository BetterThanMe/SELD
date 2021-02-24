from prepare_data.get_data import get_data
from prepare_model.get_model import get_model
from prepare_evaluate.evaluate_class import MultiOutputTest
from parameter import get_params
import yaml


# parameter
params = get_params('4')
with open('config/evaluate_config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
for key, value in config.items():
    print(f"{key:20} {value}")

# Get data from fold 1
data_eval = get_data(kind_data='valid', params=params, fold_set=config['fold_set'])

# Get model
model = get_model(model_name=config['model_name'], input_shape=config['input_shape'], params=params)

# Get weight model
weight_path = config['weight_path']

# Running evaluation
evaluate_machine = MultiOutputTest(model=model, dataset=data_eval, model_weight_path=weight_path,
                                   batch_size=config['batch_size'], file_save_path=config['file_path'],
                                   fold_set=config['fold_set'])
evaluate_machine()

