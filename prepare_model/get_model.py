import sys
sys.path.append('./prepare_model')
from dcase3_model_class import DcaseModelSet


def get_model(model_name, input_shape, params, is_summary=True):
    model = DcaseModelSet(model_name, input_shape, params)(name_model=model_name)
    if is_summary:
        model.summary()
    return model


