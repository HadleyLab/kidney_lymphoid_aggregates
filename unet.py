import keras
from keras import backend as K
from keras import layers
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
import numpy as np

def get_vgg_encoder(img_input, encoder):
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    if encoder == 'vgg19':
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    if encoder == 'vgg19':
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    if encoder == 'vgg19':
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(inputs=img_input, outputs=x)

    # Load ImageNet weights
    if encoder == 'vgg19':
        pretrained_model = VGG19(include_top=False)
    else:
        pretrained_model = VGG16(include_top=False)
    for layer, pretrained_layer in zip(
            model.layers[2:], pretrained_model.layers[2:]):
        layer.set_weights(pretrained_layer.get_weights())
    imagenet_weights = pretrained_model.layers[1].get_weights()
    init_bias = imagenet_weights[1]
    init_kernel = np.average(imagenet_weights[0], axis=2)
    init_kernel = np.reshape(
        init_kernel,
        (init_kernel.shape[0],
            init_kernel.shape[1],
            1,
            init_kernel.shape[2]))
    init_kernel = np.dstack([init_kernel] * img_input.shape.as_list()[-1])  # input image is grayscale
    model.layers[1].set_weights([init_kernel, init_bias])

    return model

def DecoderBlockv2(x, e, middle_channels, out_channels,
                  activation='relu'):
    x = layers.UpSampling2D((2,2), interpolation='bilinear')(x)
#     x = layers.ZeroPadding2D()(x)
    if e is not None:
        x = layers.concatenate([x, e])
    x = layers.Conv2D(middle_channels, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(trainable=True)(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(out_channels, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(trainable=True)(x)
    x = layers.Activation(activation)(x)

    return x

def construct_decoder(enc, activation = 'selu'):
    center = enc.get_layer("block4_pool").output
    deconv1 = DecoderBlockv2(center, enc.get_layer("block3_pool").output, 512, 256,
                            activation=activation)
    deconv2 = DecoderBlockv2(deconv1, enc.get_layer("block2_pool").output, 384, 192,
                            activation=activation)
    deconv3 = DecoderBlockv2(deconv2, enc.get_layer("block1_pool").output, 256, 128,
                            activation=activation)
    deconv4 = DecoderBlockv2(deconv3, enc.get_layer("block1_conv2").output, 128, 64,
                             activation=activation)
    logit = layers.Conv2D(1, (1, 1),
                         padding='same',
                         kernel_initializer='he_normal')(deconv4)
    score_ = layers.Activation('sigmoid')(logit)
    return score_

def construct_unet(width, height, encoder = 'vgg16', freeze_encoder=True):

    img_input = layers.Input(shape = (width, height, 3))
    enc = get_vgg_encoder(img_input, encoder)

    if freeze_encoder:
        for la in enc.layers:
            la.trainable=False

    score_= construct_decoder(enc, activation = 'selu')
    unet = Model(img_input, score_)

    return unet
