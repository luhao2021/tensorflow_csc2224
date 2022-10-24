import tensorflow as tf
from nets import tcl
from tensorflow.keras.layers import BatchNormalization, Dense, Conv2D

# policy = tf.keras.mixed_precision.Policy("mixed_float16")
# policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
# policy = tf.keras.mixed_precision.Policy("mixed_cus")
policy = tf.float32
inplace = True
use_sp_mask = False
use_sp_ratio = True
sp_ratio = (2,4)

class Model(tf.keras.Model):
    def __init__(self, num_layers, num_class, name = 'ResNet', trainable = True, **kwargs):
        super(Model, self).__init__(name = name, **kwargs)
        def kwargs(**kwargs):
            return kwargs
        # setattr(tcl.Conv2d, 'pre_defined', kwargs(use_biases = False, activation_fn = None, trainable = trainable, dtype=policy))
        # setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = trainable, dtype=policy))
        # setattr(tcl.FC, 'pre_defined', kwargs(trainable = trainable, dtype=policy))

        self.num_layers = num_layers

        self.Layers = {}
        network_argments = {
            ## ILSVRC
            18 : {'blocks' : [2,2,2,2],'depth' : [64,128,256,512], 'strides' : [1,2,2,2]},
            50 : {'blocks' : [3,4,6,3],'depth' : [64,128,256,512], 'strides' : [1,2,2,2]},

            ## CIFAR
            56 : {'blocks' : [9,9,9],'depth' : [16,32,64], 'strides' : [1,2,2]},
        }
        self.net_args = network_argments[self.num_layers]


        if num_class == 1000:
            if use_sp_mask is True:
                sparse_mask = tf.cast(tf.random.uniform(shape=(3,3,3,self.net_args['depth'][0]))>0.5, dtype=policy)
                self.Layers['conv'] = Conv2D(self.net_args['depth'][0], 7, strides = 2, padding = 'same', sp_mask=sparse_mask, sp_inplace=inplace)
            elif use_sp_ratio is True:
                self.Layers['conv'] = Conv2D(self.net_args['depth'][0], 7, strides = 2, padding = 'same', sp_ratio=sp_ratio, sp_inplace=inplace)
            else:
                self.Layers['conv'] = tcl.Conv2d([7,7], self.net_args['depth'][0], strides = 2, name = 'conv', dtype=policy)
            self.Layers['bn']   = BatchNormalization(name = 'bn', dtype=policy)
            self.maxpool_3x3 = tf.keras.layers.MaxPool2D((3,3), strides = 2, padding = 'SAME', dtype=policy)

        else:
            if use_sp_mask is True:
                sparse_mask = tf.cast(tf.random.uniform(shape=(3,3,3,self.net_args['depth'][0]))>0.5, dtype=policy)
                #print("11111:", self.net_args['depth'][0])
                self.Layers['conv'] = Conv2D(self.net_args['depth'][0], 3, strides = 1, padding = 'same', sp_mask=sparse_mask, sp_inplace=inplace)
            elif use_sp_ratio is True:
                self.Layers['conv'] = Conv2D(self.net_args['depth'][0], 3, strides = 1, padding = 'same', sp_ratio=sp_ratio, sp_inplace=inplace)
            else:
                self.Layers['conv'] = tcl.Conv2d([3,3], self.net_args['depth'][0], name = 'conv', dtype=policy)
            self.Layers['bn']   = BatchNormalization(name = 'bn', dtype=policy)
            # self.Layers['bn']   = tcl.BatchNorm(name = 'bn', dtype=tf.float32)

        self.expansion = 1 if self.num_layers in {18, 56} else 4
        in_depth = self.net_args['depth'][0]
        for i, (nb_resnet_layers, depth, strides) in enumerate(zip(self.net_args['blocks'], self.net_args['depth'], self.net_args['strides'])):
            for j in range(nb_resnet_layers):
                name = 'BasicBlock%d.%d/'%(i,j)
                if j != 0:
                    strides = 1

                if strides > 1 or depth * self.expansion != in_depth:
                    if use_sp_mask is True:
                        sparse_mask = tf.cast(tf.random.uniform(shape=(1,1,in_depth,depth))>0.5, dtype=policy)
                        self.Layers[name + 'conv3'] = Conv2D(depth * self.expansion, 1, strides = strides, padding = 'same', sp_mask=sparse_mask, sp_inplace=inplace)
                        #print("222222:", depth, self.expansion, depth * self.expansion)
                    elif use_sp_ratio is True:
                        self.Layers[name + 'conv3'] = Conv2D(depth * self.expansion, 1, strides = strides, padding = 'same', sp_ratio=sp_ratio, sp_inplace=inplace)
                    else:
                        self.Layers[name + 'conv3'] = tcl.Conv2d([1,1], depth * self.expansion, strides = strides, name = name +'conv3', dtype=policy)
                    self.Layers[name + 'bn3']   = BatchNormalization(name = name + 'bn3', dtype=policy)
                    # self.Layers[name + 'bn3']   = tcl.BatchNorm(name = name + 'bn3', dtype=tf.float32)

                if self.num_layers in {18, 56}:
                    if use_sp_mask is True:
                        #print("33333:", depth, self.expansion, depth * self.expansion)
                        sparse_mask = tf.cast(tf.random.uniform(shape=(3,3,in_depth,depth))>0.5, dtype=policy)
                        self.Layers[name + 'conv1'] = Conv2D(depth, 3, strides = strides, padding = 'same', sp_mask=sparse_mask, sp_inplace=inplace)
                        sparse_mask = tf.cast(tf.random.uniform(shape=(3,3,depth,depth * self.expansion))>0.5, dtype=policy)
                        self.Layers[name + 'conv2'] = Conv2D(depth * self.expansion, 3, strides = 1, padding = 'same', sp_mask=sparse_mask, sp_inplace=inplace)
                    elif use_sp_ratio is True:
                        self.Layers[name + 'conv1'] = Conv2D(depth, 3, strides = strides, padding = 'same', sp_ratio=sp_ratio, sp_inplace=inplace)
                        self.Layers[name + 'conv2'] = Conv2D(depth * self.expansion, 3, strides = 1, padding = 'same', sp_ratio=sp_ratio, sp_inplace=inplace)
                    else:
                        self.Layers[name + 'conv1'] = tcl.Conv2d([3,3], depth, strides = strides, name = name + 'conv1', dtype=policy)
                        self.Layers[name + 'conv2'] = tcl.Conv2d([3,3], depth * self.expansion, name = name + 'conv2', dtype=policy)
                    self.Layers[name + 'bn1']   = BatchNormalization(name = name + 'bn1', dtype=policy)
                    self.Layers[name + 'bn2']   = BatchNormalization(name = name + 'bn2', dtype=policy)
                    # self.Layers[name + 'bn2']   = tcl.BatchNorm( name = name + 'bn2', dtype=tf.float32)

                else:
                    if use_sp_mask is True:
                        sparse_mask = tf.cast(tf.random.uniform(shape=(3,3,in_depth,depth))>0.5, dtype=policy)
                        self.Layers[name + 'conv0'] = Conv2D(depth, 1, strides = 1, padding = 'same', sp_mask=sparse_mask, sp_inplace=inplace)
                        sparse_mask = tf.cast(tf.random.uniform(shape=(3,3,depth,depth))>0.5, dtype=policy)
                        self.Layers[name + 'conv1'] = Conv2D(depth, 3, strides = strides, padding = 'same', sp_mask=sparse_mask, sp_inplace=inplace)
                        sparse_mask = tf.cast(tf.random.uniform(shape=(3,3,depth,depth * self.expansion))>0.5, dtype=policy)
                        self.Layers[name + 'conv2'] = Conv2D(depth * self.expansion, 1, strides = 1, padding = 'same', sp_mask=sparse_mask, sp_inplace=inplace)
                    elif use_sp_ratio is True:
                        self.Layers[name + 'conv0'] = Conv2D(depth, 1, strides = 1, padding = 'same', sp_ratio=sp_ratio, sp_inplace=inplace)
                        self.Layers[name + 'conv1'] = Conv2D(depth, 3, strides = strides, padding = 'same', sp_ratio=sp_ratio, sp_inplace=inplace)
                    else:
                        self.Layers[name + 'conv0'] = tcl.Conv2d([1,1], depth, name = name + 'conv0', dtype=policy)
                        self.Layers[name + 'conv1'] = tcl.Conv2d([3,3], depth, strides = strides, name = name + 'conv1', dtype=policy)
                        self.Layers[name + 'conv2'] = tcl.Conv2d([1,1], depth * self.expansion, name = name + 'conv2', dtype=policy)
                    self.Layers[name + 'bn0']   = BatchNormalization( name = name + 'bn0', dtype=policy)
                    self.Layers[name + 'bn1']   = BatchNormalization( name = name + 'bn1', dtype=policy)
                    self.Layers[name + 'bn2']   = BatchNormalization( name = name + 'bn2', dtype=policy)
                    # self.Layers[name + 'bn2']   = tcl.BatchNorm( name = name + 'bn2', dtype=tf.float32)
                        #param_initializers = {'gamma': tf.keras.initializers.Zeros()})

                in_depth = depth * self.expansion

        if use_sp_mask is True:
            sparse_mask = tf.cast(tf.random.uniform(shape=(512, num_class))>0.5, dtype=policy)
            self.Layers['fc'] = Dense(num_class, name = 'fc', dtype=policy, sp_mask=sparse_mask, sp_inplace=inplace)
        elif use_sp_ratio is True:
            self.Layers['fc'] = Dense(num_class, name = 'fc', dtype=policy, sp_ratio=sp_ratio, sp_inplace=inplace)
        else:
            self.Layers['fc'] = Dense(num_class, name = 'fc', dtype=policy)
            # self.Layers['fc'] = tcl.FC(num_class, name = 'fc', dtype=policy)

    def call(self, x, training=None):
        x = self.Layers['conv'](x)
        x = self.Layers['bn'](x)
        x = tf.nn.relu(x)
        if hasattr(self, 'maxpool_3x3'):
            x = self.maxpool_3x3(x)

        in_depth = self.net_args['depth'][0]

        for i, (nb_resnet_layers, depth, strides) in enumerate(zip(self.net_args['blocks'], self.net_args['depth'], self.net_args['strides'])):
            for j in range(nb_resnet_layers):
                name = 'BasicBlock%d.%d/'%(i,j)
                if j != 0:
                    strides = 1

                if strides > 1 or depth * self.expansion != in_depth:
                    residual = self.Layers[name + 'conv3'](x)
                    residual = self.Layers[name + 'bn3'](residual)
                else:
                    residual = x
                    
                if self.num_layers not in {18, 56}:
                    x = self.Layers[name + 'conv0'](x)
                    x = self.Layers[name + 'bn0'](x)
                    x = tf.nn.relu(x)

                x = self.Layers[name + 'conv1'](x)
                x = self.Layers[name + 'bn1'](x)
                x = tf.nn.relu(x)

                x = self.Layers[name + 'conv2'](x)
                x = self.Layers[name + 'bn2'](x)
                
                x = tf.nn.relu(x + residual)



                in_depth = depth * self.expansion

        x = tf.reduce_mean(x, [1,2])
        x = self.Layers['fc'](x)
        x = tf.cast(x, tf.float32)
        return x
