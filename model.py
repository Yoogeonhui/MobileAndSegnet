import tensorflow as tf


def weight_variable(kernel_size, channel, multiplier = 1):
    initial = tf.truncated_normal([kernel_size, kernel_size, channel, multiplier])
    return tf.Variable(initial)


def bias_variable(channel):
    initial = tf.truncated_normal([channel])
    return tf.Variable(initial)


def dwconv(input_tensor ,kernel_size, channel, strides, multiplier=1, bias = False, batchnorm = True):
    dwconv_weight = weight_variable(kernel_size, channel, multiplier)
    dwconv_bias = bias_variable(channel)
    dwconv1 = tf.nn.depthwise_conv2d(input_tensor, dwconv_weight, strides= [1,strides,strides,1], padding='SAME')
    if bias:
        dwconv1 += dwconv_bias
    batchnorm2 = dwconv1
    if batchnorm:
        batchnorm2 = tf.layers.batch_normalization(dwconv1)
    relu3 = tf.nn.relu6(batchnorm2)
    return relu3


def conv_bn(input_tensor, kernel_size, out_channel, is_relu = True, strides = 1, padding = 'SAME'):
    conv1 = tf.layers.conv2d(input_tensor, filters = out_channel, kernel_size = kernel_size, strides = strides, padding=padding)
    batchnorm2 = tf.layers.batch_normalization(conv1)
    relu3 = batchnorm2
    if is_relu:
        relu3 = tf.nn.relu6(batchnorm2)
    return relu3


def conv1x1_bn(input_tensor, out_channel, is_relu = True):
    return conv_bn(input_tensor, 1, out_channel, is_relu = is_relu)


class Model:

    def __init__(self, input_width, input_height, output_channel):
        self.input_tensor = tf.placeholder(tf.float32,[None, input_height, input_width, 3])
        self.label_tensor = tf.placeholder(tf.int32, [None, input_height, input_width])
        self.output_channel = output_channel
        self.model = self.build_model(self.input_tensor)
        self.label_result = tf.argmax(self.model, 3)

    def get_loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_tensor, logits=self.model)
        return tf.reduce_mean(loss)

    def mobile_block(self, input_tensor, in_channel, out_channel, strides = 1, expand_ratio = 1, kernel_size = 3):
        mid_channel= int(in_channel * expand_ratio)
        convbn1 = conv1x1_bn(input_tensor, mid_channel)
        dwconv2 = dwconv(convbn1, kernel_size, mid_channel, strides)
        convbn3 = conv1x1_bn(dwconv2, out_channel, False)
        if in_channel == out_channel and strides == 1:
            convbn3 += input_tensor
        return convbn3

    def build_model(self, input_tensor, first_conv_channel = 32):
        conv_bn1 = conv_bn(input_tensor, 3, first_conv_channel)
        input_channel = first_conv_channel
        down_sample_list = [
            # t, c, n ,s
            [1, 16, 1, 1],
            [4, 24, 2, 2],
            [4, 32, 2, 2],
            [6, 64, 3, 2],
            [6, 130, 5,1]
        ]
        up_sample_list= [
            # t, c, n, bool(is_Upsample)
            [6, 64, 3, True],
            [6, 32, 3, True],
            [6, 24, 2, True],
            [2, self.output_channel, 1, False]
        ]
        next_tensor = conv_bn1
        for t, c, n, s in down_sample_list:
            output_channel = c
            for i in range(n):
                if i == 0:
                    next_tensor = self.mobile_block(next_tensor, input_channel, output_channel, s, t)
                else:
                    next_tensor = self.mobile_block(next_tensor, input_channel, output_channel, 1, t)
                input_channel = output_channel

        for t, c, n ,u in up_sample_list:
            output_channel = c
            for i in range(n):
                if i == 0:
                    if u:
                        next_size = next_tensor.get_shape()
                        next_tensor = tf.image.resize_images(next_tensor, [next_size[1] * 2, next_size[2] * 2])
                next_tensor = self.mobile_block(next_tensor, input_channel, output_channel, 1, t)
                input_channel = output_channel

        return next_tensor