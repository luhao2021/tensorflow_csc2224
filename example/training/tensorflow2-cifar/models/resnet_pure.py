'''
ResNet18/34/50/101/152 in TensorFlow2.

Reference:
[1] He, Kaiming, et al. 
    "Deep residual learning for image recognition." 
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
'''
from turtle import window_height
import tensorflow as tf
from tensorflow.keras import layers
import sys

@tf.function(experimental_compile=True)
def conv2d(x, W, b, strides=1, use_bias=False):
    # Conv2D wrapper, with bias and relu activation.
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    if use_bias:
      x = tf.nn.bias_add(x, b)
    # return tf.nn.relu(x)
    return x



@tf.function(experimental_compile=True)
def batchnorm(x):
    mean, variance = tf.nn.moments(x, axes=[0])
    return tf.nn.batch_normalization(x, mean, variance, 0.0, 1.0, 0.001)


random_normal = tf.initializer.RandomNormal()

weights = {
  # 3 x 3 kernel, input channel is 64, 64 filters
  "wc1": tf.Variable(random_normal([3, 3, 64, 64])),
  "weight_bn1": tf.Variable()
}




class BasicBlock(tf.keras.Model):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.nn.relu(out)
        return out

class BottleNeck(tf.keras.Model):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out


BuildResNet(BasicBlock, [2, 2, 2, 2], num_classes)

self.in_channels = 64

conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)

conv1 = conv2d()
self.bn1 = layers.BatchNormalization()
self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1)
self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2)
self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2)
self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2)
self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
self.flatten = layers.Flatten()
self.fc = layers.Dense(num_classes, activation='softmax')




class BuildResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes):
        super(BuildResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, strides):
        stride = [strides] + [1]*(num_blocks-1)
        layer = []
        for s in stride:
            layer += [block(self.in_channels, out_channels, s)]
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layer)


def ResNet(model_type, num_classes):
    if model_type == 'resnet18':
        return BuildResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif model_type == 'resnet34':
        return BuildResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    elif model_type == 'resnet50':
        return BuildResNet(BottleNeck, [3, 4, 6, 3], num_classes)
    elif model_type == 'resnet101':
        return BuildResNet(BottleNeck, [3, 4, 23, 3], num_classes)
    elif model_type == 'resnet152':
        return BuildResNet(BottleNeck, [3, 8, 36, 3], num_classes)
    else:
        sys.exit(ValueError("{:s} is currently not supported.".format(model_type)))
