import numpy as np
from keras import layers
import keras.backend as K
import tensorflow as tf
from keras import initializers, layers, models, optimizers
from keras.layers import Reshape, Conv1D, RepeatVector, merge


def CapsNet(input_shape, n_class, routings, wavenet_layers, n_column=3, dim_capsule=8, n_capsules_per_column=3, lr=0.02,
            lam_recon=0.395, decoder=None):
    """
    An adapted capsule network model operating in 1D with causal, dilated convolutions and skip connections in primarycaps
    :param input_shape: data shape, 2d, [audio_len, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param wavenet_layers: wavenet layers in primary capsules
    :param n_column: (optional) number of routing columns for routing caps
    :param dim_capsule: dimension of all capsules
    :param n_capsules_per_column: number of capsules
    :param lr: learning rate
    :param lam_recon: weight of reconstruction in loss function
    :param decoder: decoder model to use. If None, default to MLP decoder. Must be keras Sequential model.
    :return:
    """
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 2: Conv1D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    # dim_capsule = 8
    primarycaps = PrimaryCap(x, wavenet_layers=wavenet_layers, dim_capsule=dim_capsule, n_channels=32, kernel_size=10,
                             strides=1, padding='causal', activation='relu', name_concat='1')

    primarycaps.set_shape((None,) + input_shape[:-1] + (dim_capsule,))

    def slice(x, idx_start, idx_end):
        return x[:, idx_start:idx_end, :]

    def digicaps_from_columns(input_tensor, n_slice, n_capsules_per_column):
        from math import ceil

        assert input_tensor.shape[
                   1].value % n_slice == 0, "Size of input tensor must divide by n_routing_columns exactly. Any of these should be fine: {}".format(
            [x for x in range(1, 100) if input_tensor.shape[1].value % x == 0])
        n_elem = ceil(input_shape[0] / n_slice)

        # slice primary caps layer to begin each of the routing columns
        primarycaps_slices = [
            layers.Lambda(slice, name='primarycap_slice{}'.format(s),
                          arguments={'idx_start': s * n_elem, 'idx_end': (s + 1) * n_elem})(input_tensor) for s in
            range(n_slice)
        ]

        # note: routing columns all share weights (should detect same features anywhere in the input audio range)
        digicap = CapsuleLayer(num_capsule=n_capsules_per_column, dim_capsule=dim_capsule, routings=routings,
                               name='routing_caps')
        digicaps_slices = [digicap(primary) for primary in primarycaps_slices]

        # reshape to add dim for slice # to concatenate along
        digi_reshape_layer = layers.Reshape((1, n_capsules_per_column, dim_capsule))
        digicaps_slices = [digi_reshape_layer(digicap) for digicap in digicaps_slices]

        # Concatenate for output
        if n_slice > 1:
            digicaps = layers.Concatenate(axis=1)(digicaps_slices)
        else:
            digicaps = digicaps_slices[0]

        return digicaps

    if n_column == 1:
        digitcaps = CapsuleLayer(num_capsule=n_capsules_per_column, dim_capsule=dim_capsule, routings=routings,
                                 name='routing_caps')(primarycaps)
        out_caps = Length(name='capsnet')(digitcaps)
    else:
        digitcaps = digicaps_from_columns(primarycaps, n_slice=n_column, n_capsules_per_column=n_capsules_per_column)
        digitcaps = layers.Flatten()(digitcaps)
        digitcaps.set_shape((None, np.prod(digitcaps._shape_as_list()[1:])))  # specify the shape
        out_caps = layers.Dense(n_class, activation='softmax', name='capsnet_softmax')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    if decoder is None:  # default to feedforward dense decoder
        decoder_layers = [100, 100]
        # Shared Decoder model in training and prediction
        decoder = models.Sequential(name='decoder')
        decoder.add(
            layers.Dense(decoder_layers[0], activation='sigmoid', input_dim=np.prod(digitcaps.shape.as_list()[1:])))
        for dl in decoder_layers[1:]:
            decoder.add(layers.Dense(dl, activation='relu'))
        decoder.add(layers.Dense(input_shape[0], activation='sigmoid'))
        decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, dim_capsule))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

    from loss.loss import margin_loss

    # compile the model
    train_model.compile(optimizer=optimizers.Adam(lr=lr),
                        loss=[margin_loss, 'mse'],
                        loss_weights=[1., lam_recon],
                        metrics=['acc', 'mse'])

    return train_model, eval_model, manipulate_model


def wavenetBlock(n_atrous_filters, atrous_filter_size, atrous_rate, skip_out_dim=1):
    def f(input_):
        # tile residuals to match output dim of skip_out for merge (capsule implementation)
        # note: tile only required in first wavenet block: after that, skip_out_dim channels
        # are already provided
        # Todo: clean up this horrible hack: using reshape instead of tile to work around "AttributeError: 'Tensor' object has no attribute '_keras_history'" using keras.backend.tile functionality
        if int(input_.shape[2]) < skip_out_dim:
            # residual = tile(input_, (1,1,skip_out_dim) )
            thing1 = Reshape(target_shape=[int(input_.shape[1])])(input_)
            thing2 = RepeatVector(skip_out_dim)(thing1)
            residual = Reshape(target_shape=[int(thing2.shape[2]), int(thing2.shape[1])])(thing2)
        else:
            residual = input_

        tanh_out = Conv1D(
            filters=n_atrous_filters,
            kernel_size=atrous_filter_size,
            dilation_rate=atrous_rate,
            padding='causal',
            activation='tanh')(input_)

        sigmoid_out = Conv1D(
            filters=n_atrous_filters,
            kernel_size=atrous_filter_size,
            dilation_rate=atrous_rate,
            padding='causal',
            activation='sigmoid')(input_)

        merged = merge.Multiply()([tanh_out, sigmoid_out])
        skip_out = Conv1D(skip_out_dim, 1, activation='relu', border_mode='same')(merged)
        out = merge.Add()([skip_out, residual])
        return out, skip_out

    return f


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """

    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, -1), num_classes=x.get_shape().as_list()[-1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        if mask.shape.ndims == (inputs.shape.ndims - 2):
            masked = K.batch_flatten(inputs * K.expand_dims(K.expand_dims(mask, -1), 1))
        else:
            masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))

        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors + K.epsilon()


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, wavenet_layers, dim_capsule, n_channels, kernel_size, strides, padding, name_concat='',
               activation='relu'):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    # Layer 1: Wavenet convolutional layers!

    A, B = wavenetBlock(n_atrous_filters=64, atrous_filter_size=2, atrous_rate=2,
                        skip_out_dim=dim_capsule)(inputs)
    skip_connections = [B]
    for i in range(wavenet_layers):
        A, B = wavenetBlock(n_atrous_filters=64, atrous_filter_size=2, atrous_rate=2 ** ((i + 2) % 9),
                            skip_out_dim=dim_capsule)(A)
        skip_connections.append(B)
    net = layers.merge.Add()(skip_connections)
    output_wn = layers.Activation(activation)(net)

    # output = layers.Conv1D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
    #                         name='primarycap_conv1d'+name_concat, activation=activation)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape' + name_concat)(output_wn)
    return layers.Lambda(squash, name='primarycap_squash' + name_concat)(outputs)