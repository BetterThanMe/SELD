import tensorflow as tf
import sys
sys.path.append('./prepare_model')
from tensorflow.keras.layers import Bidirectional, GRU, Conv2D, \
     BatchNormalization, MaxPool2D, Dropout, Dense
from model_components import ConvSubsampling, ConformerBlock, SelfAttention, GRUs
from model_library import ModelSet


class DcaseModelSet(ModelSet):
    def __init__(self, name, input_shape, params):
        super(DcaseModelSet, self).__init__(name, input_shape)
        self.params = params
        self.seq_length = input_shape[0]
        self.out_shape_sed = [self.seq_length//5, 14]
        self.out_shape_doa = [self.seq_length//5, 42]
        self.input_shape = input_shape
        self.name = name

    def baseModel(self):
        input_combine = tf.keras.Input(shape=self.input_shape)
        x = Bidirectional(GRU(28, return_sequences=True), name='Bi_1')(input_combine)
        # x = Dropout(rate=0.2)(x)
        x = Bidirectional(GRU(28, return_sequences=True), name='Bi_2')(x)
        model = tf.keras.Model(
            inputs=input_combine,
            outputs=x,
            name='combined_model')
        return model

    # base model with sed task
    def sedV0(self, *args, **kwargs):
        out_shape_sed = self.out_shape_sed
        params = self.params
        inputs = tf.keras.Input(self.input_shape)
        drop_rate = 1. - params['dropout_keep_prob_cnn']

        x = Conv2D(name='conv1', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization(name='bn1', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)

        x = Conv2D(name='conv2', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn2', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool2', pool_size=(5, 2), strides=(5, 2), padding='same')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv3', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn3', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool3', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv4', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn4', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool4', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv5', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn5', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool5', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv6', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn6', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool6', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = tf.reshape(x, [-1, out_shape_sed[0], 2 * 256])

        x = Bidirectional(GRU(units=params['rnn_hidden_size'], return_sequences=True),
                                 name='bidirecrtionalGRU')(x)

        x = SelfAttention(attention_size=params['attention_size'])(x)

        x = tf.reshape(x, [-1, 2 * params['rnn_hidden_size']])

        drop_rate_dnn = 1. - params['dropout_keep_prob_dnn']
        # -------------SED----------------
        x_sed = Dense(params['dnn_size'], activation='relu', name='dense_relu_sed1')(x)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(params['dnn_size'], activation='relu', name='dense_relu_sed2')(x_sed)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(out_shape_sed[-1], name='dense_sed3')(x_sed)
        x_sed = tf.keras.activations.sigmoid(x_sed)
        x_sed = tf.reshape(x_sed, shape=[-1, out_shape_sed[0], out_shape_sed[1]], name='output_sed')

        model = tf.keras.Model(
            inputs=inputs,
            outputs=x_sed,
            name="Sed_net_v0")
        return model

    # replace bidirectional layer with conformer encoder
    def doaV1(self, dmodel=512, num_blocks=16, *args, **kwargs):
        params = self.params
        out_shape_doa = self.out_shape_doa
        inputs_shape = self.input_shape
        inputs = tf.keras.Input(inputs_shape)
        drop_rate = 1. - params['dropout_keep_prob_cnn']

        x = Conv2D(name='conv1', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization(name='bn1', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)

        x = Conv2D(name='conv2', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn2', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool2', pool_size=(5, 2), strides=(5, 2), padding='same')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv3', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn3', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool3', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv4', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn4', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool4', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv5', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn5', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool5', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv6', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn6', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool6', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        # [None, 120, 2, 256]
        x = ConvSubsampling(odim=dmodel)(x)

        for i in range(num_blocks):
            x = ConformerBlock(input_dim=dmodel, name=f"conformer_block_{i}")(x)

        drop_rate_dnn = 1. - params['dropout_keep_prob_dnn']
        # -------------DOA----------------
        x = SelfAttention(attention_size=params['attention_size'])(x)
        x = tf.reshape(x, [-1, 2 * params['rnn_hidden_size']])
        x = Dense(params['dnn_size'], activation='relu', name='dense_relu_doa1')(x)
        x = Dropout(rate=drop_rate_dnn)(x)
        x = Dense(params['dnn_size'], activation='relu', name='dense_relu_doa2')(x)
        x = Dropout(rate=drop_rate_dnn)(x)
        x = Dense(out_shape_doa[-1], name='dense_doa3')(x)
        x = tf.keras.activations.tanh(x)
        x = tf.reshape(x, shape=[-1, out_shape_doa[0], out_shape_doa[1]], name='output_doa')

        model = tf.keras.Model(
            inputs=inputs,
            outputs=x,
            name="Doa_net_v1")
        return model

    # base doa model
    def doaV0(self):
        inputs = tf.keras.Input(self.input_shape)
        drop_rate = 1. - self.params['dropout_keep_prob_cnn']

        x = Conv2D(name='conv1', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization(name='bn1', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)

        x = Conv2D(name='conv2', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn2', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool2', pool_size=(5, 2), strides=(5, 2), padding='same')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv3', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn3', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool3', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv4', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn4', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool4', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv5', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn5', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool5', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv6', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn6', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool6', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = tf.reshape(x, [-1, self.out_shape_doa[0], 2 * 256])

        x = Bidirectional(GRU(units=self.params['rnn_hidden_size'], return_sequences=True),
                                 name='bidirecrtionalGRU')(x)

        x = SelfAttention(attention_size=self.params['attention_size'])(x)

        x = tf.reshape(x, [-1, 2 * self.params['rnn_hidden_size']])

        drop_rate_dnn = 1. - self.params['dropout_keep_prob_dnn']
        # -------------DOA----------------
        x = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_doa1')(x)
        x = Dropout(rate=drop_rate_dnn)(x)
        x = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_doa2')(x)
        x = Dropout(rate=drop_rate_dnn)(x)
        x = Dense(self.out_shape_doa[-1], name='dense_doa3')(x)
        x = tf.keras.activations.tanh(x)
        x = tf.reshape(x, shape=[-1, self.out_shape_doa[0], self.out_shape_doa[1]], name='output_doa')

        model = tf.keras.Model(
            inputs=inputs,
            outputs=x,
            name="Doa_net_v0")
        return model

    def sedv1(self, dmodel=512, num_blocks=16, *args, **kwargs):
        inputs = tf.keras.Input(self.input_shape)
        params = self.params
        out_shape_sed = self.out_shape_sed
        drop_rate = 1. - params['dropout_keep_prob_cnn']

        x = Conv2D(name='conv1', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization(name='bn1', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)

        x = Conv2D(name='conv2', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn2', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool2', pool_size=(5, 2), strides=(5, 2), padding='same')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv3', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn3', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool3', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv4', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn4', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool4', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv5', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn5', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool5', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv6', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn6', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool6', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        # [None, 120, 2, 256]
        x = ConvSubsampling(odim=dmodel)(x)

        for i in range(num_blocks):
            x = ConformerBlock(input_dim=dmodel, name=f"conformer_block_{i}")(x)

        drop_rate_dnn = 1. - params['dropout_keep_prob_dnn']
        # -------------SED----------------
        x_sed = SelfAttention(attention_size=params['attention_size'])(x)
        x_sed = tf.reshape(x_sed, [-1, 2 * params['rnn_hidden_size']])
        x_sed = Dense(params['dnn_size'], activation='relu', name='dense_relu_sed1')(x_sed)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(params['dnn_size'], activation='relu', name='dense_relu_sed2')(x_sed)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(out_shape_sed[-1], name='dense_sed3')(x_sed)
        x_sed = tf.keras.activations.sigmoid(x_sed)
        x_sed = tf.reshape(x_sed, shape=[-1, out_shape_sed[0], out_shape_sed[1]], name='output_sed')

        model = tf.keras.Model(
            inputs=inputs,
            outputs=x_sed,
            name="Sed_net_v1")
        return model

    def sedvn(self):
        inputs = tf.keras.Input(self.input_shape)
        params = self.params
        out_shape_sed = self.out_shape_sed

        # (None, 600, 64, 4)
        x = Conv2D(name='conv1', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization(name='bn1', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.AvgPool2D((5, 2), strides=(5, 2))(x)

        x = Conv2D(name='conv2', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn2', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.AvgPool2D((2, 2))(x)

        x = Conv2D(name='conv3', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn3', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.AvgPool2D((2, 2))(x)

        x = tf.reshape(x, [-1, out_shape_sed[0], 2 * 256])

        x = Bidirectional(GRU(units=256, return_sequences=True), name='bidirecrtionalGRU')(x)
        x = Dropout(rate=0.35)(x)
        x = Dense(units=14, activation='sigmoid')(x)

        model = tf.keras.Model(
            inputs=inputs,
            outputs=x,
            name='Sed_net_vn'
        )
        return model

    def combV01(self):
        input_combine = tf.keras.Input(shape=self.input_shape)
        x = Bidirectional(GRU(28, return_sequences=True), name='Bi_1')(input_combine)
        x = Bidirectional(GRU(28, return_sequences=True), name='Bi_2')(x)
        # x = Dropout(rate=0.1)(x)
        x = SelfAttention(attention_size=64)(x)
        model = tf.keras.Model(
            inputs=input_combine,
            outputs=x,
            name='combined_model_v01')
        return model

    def combSed(self):
        input_combine = tf.keras.Input(shape=self.input_shape)
        x = Bidirectional(GRU(units=56, return_sequences=True), name='Bi_1')(input_combine)
        x = Bidirectional(GRU(units=56, return_sequences=True), name='Bi_2')(x)
        x = SelfAttention(attention_size=64)(x)
        x = Dense(14, activation='sigmoid')(x)
        model = tf.keras.Model(
            inputs=input_combine,
            outputs=x,
            name='Combined_model_sed'
        )
        return model

    @staticmethod
    def convBase(name_module, inputShape, drop_rate_cnn, seq_length):
        inputs = tf.keras.Input(inputShape)
        drop_rate = drop_rate_cnn

        x = Conv2D(name='conv1', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization(name='bn1', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)

        x = Conv2D(name='conv2', filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn2', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool2', pool_size=(5, 2), strides=(5, 2), padding='same')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv3', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn3', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool3', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv4', filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn4', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool4', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv5', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn5', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool5', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = Conv2D(name='conv6', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='bn6', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)
        x = MaxPool2D(name='maxpool6', pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        # label has resolution: 5
        x = tf.reshape(x, [-1, seq_length//5, 2 * 256])

        model = tf.keras.Model(
            inputs=inputs,
            outputs=x,
            name=name_module)
        return model

    def sedModule(self):
        out_shape_sed = self.out_shape_sed
        params = self.params
        inputs = tf.keras.Input([out_shape_sed[0], 2 * 256])
        x = Bidirectional(GRU(units=params['rnn_hidden_size'], return_sequences=True),
                          name='bidirecrtionalGRU')(inputs)

        x = SelfAttention(attention_size=params['attention_size'])(x)

        x = tf.reshape(x, [-1, 2 * params['rnn_hidden_size']])

        drop_rate_dnn = 1. - params['dropout_keep_prob_dnn']
        # -------------SED----------------
        x_sed = Dense(params['dnn_size'], activation='relu', name='dense_relu_sed1')(x)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(params['dnn_size'], activation='relu', name='dense_relu_sed2')(x_sed)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(out_shape_sed[-1], name='dense_sed3')(x_sed)
        x_sed = tf.keras.activations.sigmoid(x_sed)
        x_sed = tf.reshape(x_sed, shape=[-1, out_shape_sed[0], out_shape_sed[1]], name='output_sed')
        model = tf.keras.Model(
            inputs=inputs,
            outputs=x_sed,
            name="Sed_module")
        return model

    def doaModule(self):
        inputs = tf.keras.Input([self.out_shape_doa[0], 2 * 256])

        x = Bidirectional(GRU(units=self.params['rnn_hidden_size'], return_sequences=True),
                          name='bidirecrtionalGRU')(inputs)

        x = SelfAttention(attention_size=self.params['attention_size'])(x)

        x = tf.reshape(x, [-1, 2 * self.params['rnn_hidden_size']])

        drop_rate_dnn = 1. - self.params['dropout_keep_prob_dnn']
        # -------------DOA----------------
        x = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_doa1')(x)
        x = Dropout(rate=drop_rate_dnn)(x)
        x = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_doa2')(x)
        x = Dropout(rate=drop_rate_dnn)(x)
        x = Dense(self.out_shape_doa[-1], name='dense_doa3')(x)
        x = tf.keras.activations.tanh(x)
        x = tf.reshape(x, shape=[-1, self.out_shape_doa[0], self.out_shape_doa[1]], name='output_doa')

        model = tf.keras.Model(
            inputs=inputs,
            outputs=x,
            name="Doa_module")
        return model

    def transferV0(self):
        drop_rate = 1. - self.params['dropout_keep_prob_cnn']
        return self.convBase(name_module="Conv_module", inputShape=(600, 64, 7), drop_rate_cnn=drop_rate,
                             seq_length=self.seq_length), self.sedModule(), self.doaModule()

    def combSedV1(self):
        input_combine = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv1D(name='conv1D', filters=256, kernel_size=7, strides=1, padding='same')(input_combine)
        x = BatchNormalization(name='bn', center=True, scale=True, trainable=True)(x)
        x = tf.keras.activations.relu(x)

        x = Bidirectional(GRU(units=64, return_sequences=True), name='BiDirection')(x)
        x = SelfAttention(attention_size=64)(x)
        x = Dense(14, activation='sigmoid')(x)
        model = tf.keras.Model(
            inputs=input_combine,
            outputs=x,
            name='Combined_model_sed_v1'
        )
        return model

    def seldV2(self, sed_channel=4, doa_channel=7):
        drop_rate_cnn = 1. - self.params['dropout_keep_prob_cnn']
        drop_rate_dnn = 1. - self.params['dropout_keep_prob_dnn']
        inputs = tf.keras.Input(shape=self.input_shape)
        total_channel = self.input_shape[-1]
        sed_shape, doa_shape = list(self.input_shape), list(self.input_shape)
        sed_shape[2], doa_shape[2] = abs(sed_channel), abs(doa_channel)
        sed_conv_module = self.convBase("Sed_conv_module", sed_shape, drop_rate_cnn, self.seq_length)
        doa_conv_module = self.convBase("Doa_conv_module", doa_shape, drop_rate_cnn, self.seq_length)
        sed_output = sed_conv_module(tf.split(inputs, [sed_channel, -1], axis=-1)[0])
        if doa_channel > 0:
            doa_output = doa_conv_module(tf.split(inputs, [doa_channel, -1], axis=-1)[0])
        else:
            # when take the last abs(doa_channel) number of channel for doa task
            doa_output = doa_conv_module(tf.split(inputs, [total_channel + doa_channel, -doa_channel], axis=-1)[1])

        total_output = tf.concat((sed_output, doa_output), axis=-1)
        total_output = Dense(512, activation='relu', name='dense_relu_1')(total_output)
        total_output = Dropout(rate=0.2)(total_output)

        total_output = Bidirectional(GRU(256, return_sequences=True), name='BiGru')(total_output)
        total_output = SelfAttention(attention_size=64)(total_output)

        # -------------SED----------------
        x_sed = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_sed1')(total_output)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(self.params['dnn_size']//2, activation='relu', name='dense_relu_sed2')(x_sed)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(self.out_shape_sed[-1], name='dense_sed3')(x_sed)
        x_sed = tf.keras.activations.sigmoid(x_sed)

        # -------------DOA----------------
        x_doa = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_doa1')(total_output)
        x_doa = Dropout(rate=drop_rate_dnn)(x_doa)
        x_doa = Dense(self.params['dnn_size']//2, activation='relu', name='dense_relu_doa2')(x_doa)
        x_doa = Dropout(rate=drop_rate_dnn)(x_doa)
        x_doa = Dense(self.out_shape_doa[-1], name='dense_doa3')(x_doa)
        x_doa = tf.keras.activations.tanh(x_doa)
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[x_sed, x_doa],
            name='Seld_V2'
        )
        return model

    def doaV2(self):
        drop_rate_cnn = 1. - self.params['dropout_keep_prob_cnn']
        drop_rate_dnn = 1. - self.params['dropout_keep_prob_dnn']
        inputs = tf.keras.Input(shape=self.input_shape)
        sed_conv_module = self.convBase("Sed_conv_module", (600, 64, 4), drop_rate_cnn, self.seq_length)
        doa_conv_module = self.convBase("Doa_conv_module", (600, 64, 7), drop_rate_cnn, self.seq_length)
        sed_output = sed_conv_module(tf.split(inputs, [4, -1], axis=-1)[0])
        doa_output = doa_conv_module(inputs)

        total_output = tf.concat((sed_output, doa_output), axis=-1)
        total_output = Dense(512, activation='relu', name='dense_relu_1')(total_output)
        total_output = Dropout(rate=0.2)(total_output)

        total_output = Bidirectional(GRU(256, return_sequences=True), name='BiGru')(total_output)
        total_output = SelfAttention(attention_size=64)(total_output)

        # -------------SED----------------
        x_sed = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_sed1')(total_output)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(self.params['dnn_size']//2, activation='relu', name='dense_relu_sed2')(x_sed)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(self.out_shape_sed[-1], name='dense_sed3')(x_sed)
        x_sed = tf.keras.activations.sigmoid(x_sed)

        # -------------DOA----------------
        x_doa = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_doa1')(total_output)
        x_doa = Dropout(rate=drop_rate_dnn)(x_doa)
        x_doa = Dense(self.params['dnn_size']//2, activation='relu', name='dense_relu_doa2')(x_doa)
        x_doa = Dropout(rate=drop_rate_dnn)(x_doa)
        x_doa = Dense(self.out_shape_doa[-1], name='dense_doa3')(x_doa)
        x_doa = tf.keras.activations.tanh(x_doa)

        # masking Doa
        mask = tf.concat((x_sed, x_sed, x_sed), axis=-1)
        x_doa = tf.multiply(x_doa, mask)

        model = tf.keras.Model(
            inputs=inputs,
            outputs=x_doa,
            name='Doa_V2'
        )
        return model

    def doaV3(self):
        drop_rate_cnn = 1. - self.params['dropout_keep_prob_cnn']
        drop_rate_dnn = 1. - self.params['dropout_keep_prob_dnn']
        inputs = tf.keras.Input(shape=self.input_shape)
        sed_conv_module = self.convBase("Sed_conv_module", (600, 64, 4), drop_rate_cnn, self.seq_length)
        doa_conv_module = self.convBase("Doa_conv_module", (600, 64, 7), drop_rate_cnn, self.seq_length)
        sed_output = sed_conv_module(tf.split(inputs, [4, -1], axis=-1)[0])
        doa_output = doa_conv_module(inputs)

        total_output = tf.concat((sed_output, doa_output), axis=-1)
        total_output = Dense(512, activation='relu', name='dense_relu_1')(total_output)
        total_output = Dropout(rate=0.2)(total_output)

        total_output = Bidirectional(GRU(256, return_sequences=True), name='BiGru')(total_output)
        total_output = SelfAttention(attention_size=64)(total_output)

        # -------------SED----------------
        x_sed = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_sed1')(total_output)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(self.params['dnn_size']//2, activation='relu', name='dense_relu_sed2')(x_sed)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(self.out_shape_sed[-1], name='dense_sed3')(x_sed)
        x_sed = tf.keras.activations.sigmoid(x_sed)

        # -------------DOA----------------
        x_doa = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_doa1')(total_output)
        x_doa = Dropout(rate=drop_rate_dnn)(x_doa)
        x_doa = Dense(self.params['dnn_size']//2, activation='relu', name='dense_relu_doa2')(x_doa)
        x_doa = Dropout(rate=drop_rate_dnn)(x_doa)
        x_doa = Dense(self.out_shape_doa[-1], name='dense_doa3')(x_doa)
        x_doa = tf.keras.activations.tanh(x_doa)

        # masking Doa
        mask = tf.concat((x_sed, x_sed, x_sed), axis=-1)
        x_doa = tf.multiply(x_doa, mask)

        # concatenate sed and doa
        outputs = tf.concat((x_sed, x_doa), axis=-1)
        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name='Doa_V2_1'
        )
        return model

    def doaV4(self):
        drop_rate_cnn = 1. - self.params['dropout_keep_prob_cnn']
        drop_rate_dnn = 1. - self.params['dropout_keep_prob_dnn']
        inputs = tf.keras.Input(shape=self.input_shape)
        sed_conv_module = self.convBase("Sed_conv_module", (600, 64, 4), drop_rate_cnn, self.seq_length)
        doa_conv_module = self.convBase("Doa_conv_module", (600, 64, 7), drop_rate_cnn, self.seq_length)
        sed_output = sed_conv_module(tf.split(inputs, [4, -1], axis=-1)[0])
        doa_output = doa_conv_module(inputs)

        total_output = tf.concat((sed_output, doa_output), axis=-1)
        total_output = Dense(512, activation='relu', name='dense_relu_1')(total_output)
        total_output = Dropout(rate=0.2)(total_output)

        total_output = Bidirectional(GRU(256, return_sequences=True), name='BiGru')(total_output)
        total_output = SelfAttention(attention_size=64)(total_output)

        # -------------SED----------------
        x_sed = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_sed1')(total_output)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(self.params['dnn_size']//2, activation='relu', name='dense_relu_sed2')(x_sed)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(self.out_shape_sed[-1], name='dense_sed3')(x_sed)
        x_sed = tf.keras.activations.sigmoid(x_sed)

        # -------------DOA----------------
        x_doa = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_doa1')(total_output)
        x_doa = Dropout(rate=drop_rate_dnn)(x_doa)
        x_doa = Dense(self.params['dnn_size']//2, activation='relu', name='dense_relu_doa2')(x_doa)
        x_doa = Dropout(rate=drop_rate_dnn)(x_doa)
        x_doa = Dense(self.out_shape_doa[-1], name='dense_doa3')(x_doa)
        x_doa = tf.keras.activations.tanh(x_doa)

        # concatenate sed and doa
        outputs = tf.concat((x_sed, x_doa), axis=-1)
        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name='Doa_V2_3'
        )
        return model

    def seldV0(self):
        drop_rate_cnn = 1. - self.params['dropout_keep_prob_cnn']
        drop_rate_dnn = 1. - self.params['dropout_keep_prob_dnn']
        inputs = tf.keras.Input(self.input_shape)

        # Conv phase
        conv_module = self.convBase('Conv_module', self.input_shape, drop_rate_cnn, self.seq_length)
        x = conv_module(inputs)  # out shape: [None, 120, 512]

        # RNN phase
        x = Bidirectional(GRU(units=self.params['rnn_hidden_size'], return_sequences=True),
                          name='bidirecrtionalGRU')(x)
        x = SelfAttention(attention_size=self.params['attention_size'])(x)
        x = tf.reshape(x, [-1, 2 * self.params['rnn_hidden_size']])  # out shape: [None, 512]

        # -------------SED----------------
        x_sed = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_sed1')(x)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_sed2')(x_sed)
        x_sed = Dropout(rate=drop_rate_dnn)(x_sed)
        x_sed = Dense(self.out_shape_sed[-1], name='dense_sed3')(x_sed)
        x_sed = tf.keras.activations.sigmoid(x_sed)
        x_sed = tf.reshape(x_sed, shape=[-1, self.out_shape_sed[0], self.out_shape_sed[1]], name='output_sed')

        # -------------DOA----------------
        x_doa = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_doa1')(x)
        x_doa = Dropout(rate=drop_rate_dnn)(x_doa)
        x_doa = Dense(self.params['dnn_size'], activation='relu', name='dense_relu_doa2')(x_doa)
        x_doa = Dropout(rate=drop_rate_dnn)(x_doa)
        x_doa = Dense(self.out_shape_doa[-1], name='dense_doa3')(x_doa)
        x_doa = tf.keras.activations.tanh(x_doa)
        x_doa = tf.reshape(x_doa, shape=[-1, self.out_shape_doa[0], self.out_shape_doa[1]], name='output_doa')

        model = tf.keras.Model(
            inputs=inputs,
            outputs=(x_sed, x_doa),
            name="Seld_v0")
        return model


# from parameter import get_params
# params = get_params("4")
# modelSet = DcaseModelSet(name="Dcase model", input_shape=(600, 64, 7), params=params)
# model = modelSet(name_model="seldv0")
# print(type(model.input_shape))
# print(model.output_shape)
# model.summary()
