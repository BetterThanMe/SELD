import tensorflow as tf
import tensorflow_addons as tfa
import sys
sys.path.append('./prepare_model')
from activations import GLU
from utils.utils import merge_two_last_dims
from layers.positional_encoding import PositionalEncoding

L2 = tf.keras.regularizers.l2(0.)


class ConvSubsampling(tf.keras.layers.Layer):
    def __init__(self,
                 odim: int,
                 reduction_factor: int = 4 or list or tuple,
                 dropout: float = 0.0,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conv_subsampling",
                 **kwargs):
        super(ConvSubsampling, self).__init__(name=name, **kwargs)
        if isinstance(reduction_factor, (list, tuple)):
            stride = reduction_factor
        else:
            assert reduction_factor % 2 == 0, "reduction_factor must be divisible by 2"
            stride = [1, reduction_factor // 2]
        # (-1, 600, 64, 7)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=odim, kernel_size=3, strides=stride,
            padding="same", name=f"{name}_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        # (-1, 120, 32, 512)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=odim, kernel_size=3, strides=[1, 1],
            padding="same", name=f"{name}_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        # (-1, 120, 32, 512)
        self.linear = tf.keras.layers.Dense(
            odim, name=f"{name}_linear",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

    def call(self, inputs, **kwargs):
        outputs = self.conv1(inputs)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = merge_two_last_dims(outputs)
        outputs = self.linear(outputs)
        return self.do(outputs)

    def get_config(self):
        conf = super(ConvSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        conf.update(self.linear.get_config())
        conf.update(self.do.get_config())
        return conf


class FFModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="ff_module",
                 **kwargs):
        super(FFModule, self).__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.ffn1 = tf.keras.layers.Dense(
            4 * input_dim, name=f"{name}_dense_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name=f"{name}_swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            input_dim, name=f"{name}_dense_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_2")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, **kwargs):
        outputs = self.ln(inputs)
        outputs = self.ffn1(outputs)
        outputs = self.swish(outputs)
        outputs = self.do1(outputs)
        outputs = self.ffn2(outputs)
        outputs = self.do2(outputs)
        outputs = self.res_add([inputs, self.fc_factor * outputs])
        return outputs

    def get_config(self):
        conf = super(FFModule, self).get_config()
        conf.update({"fc_factor": self.fc_factor})
        conf.update(self.ln.get_config())
        conf.update(self.ffn1.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        conf.update(self.res_add.get_config())
        return conf


class MHSAModule(tf.keras.layers.Layer):
    def __init__(self,
                 head_size,
                 num_heads,
                 dropout=0.0,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="mhsa_module",
                 **kwargs):
        super(MHSAModule, self).__init__(name=name, **kwargs)
        self.pc = PositionalEncoding(name=f"{name}_pe")
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.mha = tfa.layers.MultiHeadAttention(
            head_size=head_size, num_heads=num_heads, name=f"{name}_mhsa",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, **kwargs):
        outputs = self.pc(inputs)
        outputs = self.ln(outputs)
        outputs = self.mha([outputs, outputs, outputs])
        outputs = self.do(outputs)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(MHSAModule, self).get_config()
        conf.update(self.pc.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ConvModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 kernel_size=32,
                 dropout=0.0,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conv_module",
                 **kwargs):
        super(ConvModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization()
        self.pw_conv_1 = tf.keras.layers.Conv1D(
            filters=2 * input_dim, kernel_size=1, strides=1,
            padding="same", name=f"{name}_pw_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.glu = GLU(name=f"{name}_glu")
        self.dw_conv = tf.keras.layers.SeparableConv1D(
            filters=2 * input_dim, kernel_size=kernel_size, strides=1,
            padding="same", depth_multiplier=1, name=f"{name}_dw_conv",
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{name}_bn",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name=f"{name}_swish_activation")
        self.pw_conv_2 = tf.keras.layers.Conv1D(
            filters=input_dim, kernel_size=1, strides=1,
            padding="same", name=f"{name}_pw_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, **kwargs):
        outputs = self.ln(inputs)
        outputs = self.pw_conv_1(outputs)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs)
        outputs = self.bn(outputs)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs)
        outputs = self.do(outputs)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(ConvModule, self).get_config()
        conf.update(self.ln.get_config())
        conf.update(self.pw_conv_1.get_config())
        conf.update(self.glu.get_config())
        conf.update(self.dw_conv.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.pw_conv_2.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 head_size=36,
                 num_heads=4,
                 kernel_size=32,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conformer_block",
                 **kwargs):
        super(ConformerBlock, self).__init__(name=name, **kwargs)
        self.ffm1 = FFModule(
            input_dim=input_dim, dropout=dropout,
            fc_factor=fc_factor, name=f"{name}_ff_module_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.mhsam = MHSAModule(
            head_size=head_size, num_heads=num_heads,
            dropout=dropout, name=f"{name}_mhsa_module",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.convm = ConvModule(
            input_dim=input_dim, kernel_size=kernel_size,
            dropout=dropout, name=f"{name}_conv_module",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.ffm2 = FFModule(
            input_dim=input_dim, dropout=dropout,
            fc_factor=fc_factor, name=f"{name}_ff_module_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=kernel_regularizer
        )

    def call(self, inputs, **kwargs):
        outputs = self.ffm1(inputs)
        outputs = self.mhsam(outputs)
        outputs = self.convm(outputs)
        outputs = self.ffm2(outputs)
        outputs = self.ln(outputs)
        return outputs

    def get_config(self):
        conf = super(ConformerBlock, self).get_config()
        conf.update(self.ffm1.get_config())
        conf.update(self.mhsam.get_config())
        conf.update(self.convm.get_config())
        conf.update(self.ffm2.get_config())
        conf.update(self.ln.get_config())
        return conf


class ConformerEncoder(tf.keras.Model):
    def __init__(self,
                 dmodel=512,
                 reduction_factor=4,
                 num_blocks=16,
                 head_size=36,
                 num_heads=4,
                 kernel_size=32,
                 fc_factor=0.5,
                 dropout=0.0,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conformer_encoder",
                 **kwargs):
        super(ConformerEncoder, self).__init__(name=name, **kwargs)
        self.conv_subsampling = ConvSubsampling(
            odim=dmodel, reduction_factor=reduction_factor,
            dropout=dropout, name=f"{name}_subsampling",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.conformer_blocks = []
        for i in range(num_blocks):
            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"conformer_block_{i}"
            )
            self.conformer_blocks.append(conformer_block)

    def call(self, inputs, **kwargs):
        # input with shape [B, T, V1, V2]
        outputs = self.conv_subsampling(inputs)
        for cblock in self.conformer_blocks:
            outputs = cblock(outputs)
        return outputs

    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf


# -----------------------Create Model structure---
# class layer selfAttention for last share layer
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, attention_size, scaled_=True, masked_=False, name="self-attention", *args, **kwargs):
        super(SelfAttention, self).__init__(name=name, *args, **kwargs)
        self.attention_size = attention_size
        self.scaled = scaled_
        self.masked = masked_
        self.attention = None

    def build(self, input_shape):  # (batch_num, seq_len, ndim)
        ndim = input_shape[2]
        self.Q = tf.keras.layers.Dense(self.attention_size)
        self.K = tf.keras.layers.Dense(self.attention_size)
        self.V = tf.keras.layers.Dense(ndim)

    def call(self, inputs, **kwargs):
        q = self.Q(inputs)
        k = self.K(inputs)
        v = self.V(inputs)
        self.attention = tf.matmul(q, k, transpose_b=True)
        if self.scaled:
            d_k = tf.cast(k.shape[-1], dtype=tf.float32)
            self.attention = tf.divide(self.attention, tf.sqrt(d_k))
        if self.masked:
            raise NotImplementedError
        self.attention = tf.nn.softmax(self.attention, axis=-1)
        return tf.matmul(self.attention, v)

    def get_config(self):
        config = super(SelfAttention, self).get_config().copy()
        config.update(dict(attention_size=self.attention_size, scaled=self.scaled, masked=self.masked))
        return config


# multi cells layer Gru
class GRUs(tf.keras.layers.Layer):
    def __init__(self, units, num_cells=1, go_backwards=False, dropout=0.,
                 return_sequences=True, return_state=False, name=None, l2=0., activation='tanh', *args, **kwargs):
        super(GRUs, self).__init__(name=name)
        self.l2 = None if l2 == 0 else tf.keras.regularizers.L2(l2)
        self.num_cells = num_cells
        self.dim = units
        self.dropout = dropout
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.activation = activation
        self.states = None

        def gruCell():
            return tf.keras.layers.GRU(units, dropout=self.dropout, go_backwards=self.go_backwards,
                                       return_state=True, return_sequences=True, stateful=False,
                                       kernel_regularizer=self.l2, bias_regularizer=self.l2, activation=activation)

        self._layers_name = ['GruCell_' + str(i) for i in range(num_cells)]
        for name in self._layers_name:
            self.__setattr__(name, gruCell())

    def get_config(self):
        config = super(GRUs, self).get_config().copy()
        config.update({
            'num_cells': self.num_cells,
            'dropout': self.dropout,
            'go_backwards': self.go_backwards,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            '_layers_name': self._layers_name,
            'units': self.dim,
            'activation': self.activation
        })
        return config

    def call(self, inputs, **kwargs):
        seq = inputs
        state = None
        for name in self._layers_name:
            cell = self.__getattribute__(name)
            (seq, state) = cell(seq, initial_state=state)
        self.states = state
        if self.return_state:
            if self.return_sequences:
                return [seq, state]
            return [seq[:, -1, :], state]
        if self.return_sequences:
            return seq
        return seq[:, -1, :]
