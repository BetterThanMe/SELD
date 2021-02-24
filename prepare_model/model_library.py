from abc import ABC, abstractmethod


class ModelSet(ABC):
    """
    Object hold all the models, and return required model by pass its name
    """
    def __init__(self, name, input_shape, *args, **kwargs):
        self.name = name
        self.input_shape = input_shape
        super(ModelSet, self).__init__()

    @abstractmethod
    def baseModel(self, *args, **kwargs):
        """
        :return: model object class
        """
        pass

    def __call__(self, input_shape=None, name_model=None, *args, **kwargs):
        if input_shape is not None:
            self.input_shape = input_shape
        model_func = None
        if name_model is not None:
            for attr in dir(self):
                if name_model.lower() in attr.lower():
                    model_func = self.__getattribute__(attr)

        if model_func is None:
            return self.baseModel(*args, **kwargs)
        else:
            return model_func(*args, **kwargs)
