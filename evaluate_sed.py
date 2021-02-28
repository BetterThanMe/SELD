from prepare_data.get_data import get_data
from prepare_model.get_model import get_model
from parameter import get_params
import yaml
import tensorflow as tf


@tf.function
def batchStep(model_, inputs_):
    return model_(inputs_, training=False)


def compare(label, output, batch_number):
    count_overlap_err = 0
    count_frame_err = 0
    error_string = f'Batch: {batch_number:5d}: '
    for batch in range(label.shape[0]):
        for frame in range(label.shape[1]):
            count = 0
            is_err = False
            for class_active in range(label.shape[-1]):
                if label[batch, frame, class_active] != output[batch, frame, class_active]:
                    error_string += '{}: {} - {} | '.format(class_active, label[batch, frame, class_active],
                                                            output[batch, frame, class_active])
                    is_err = True
                if label[batch, frame, class_active]:
                    count += 1
            if is_err:
                count_frame_err += 1
                if count >= 2:
                    count_overlap_err += 1
    return error_string, count_overlap_err, count_frame_err


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

dataset = tf.data.Dataset.from_tensor_slices(data_eval).batch(16)

error_content = ''
count_ov, count_all = 0, 0
sed_eval_path = './sed_eval.txt'
for index, (inputs, sed_label, doa_label) in enumerate(dataset):
    sed_output, doa_output = batchStep(model, inputs)
    sed_output_ = sed_output > 0.5
    sed_label_ = sed_label > 0.5
    error_string, count_overlap_frame_error, count_total_frame_err = compare(sed_label_, sed_output_, index)
    error_content += error_string
    count_ov += count_overlap_frame_error
    count_all += count_total_frame_err
    print(f'\rProcessing {int(float(index+1)/len(dataset)*100)}%', end='')


with open(sed_eval_path, 'w') as f:
    f.write(error_content + '\nError_overlap/Error_total = {}/{}'.format(count_ov, count_all))

