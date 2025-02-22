"""Train CIFAR-10 with TensorFlow2.0."""
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import time

from models import *
from utils import *

parser = argparse.ArgumentParser(description='TensorFlow2.0 CIFAR-10 Training')
parser.add_argument('--model', required=True, type=str, help='model type')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='number of training epoch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default=[0], type=int, nargs = '+', help='specify which gpu to be used')
args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.model = args.model.lower()


gpus = tf.config.list_physical_devices('GPU')
print(args.gpu, gpus)
tf.config.set_visible_devices([tf.config.list_physical_devices('GPU')[i] for i in args.gpu], 'GPU')
for gpu_id in args.gpu:
    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
devices = ['/gpu:{}'.format(i) for i in args.gpu]
strategy = tf.distribute.MirroredStrategy(devices, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

class Model():
    def __init__(self, model_type, decay_steps, num_classes=10):
        if 'lenet' in model_type:
            self.model = LeNet(num_classes)
        elif 'alexnet' in model_type:
            self.model = AlexNet(num_classes)
        elif 'vgg' in model_type:
            self.model = VGG(model_type, num_classes)
        elif 'resnet' in model_type:
            if 'se' in model_type:
                if 'preact' in model_type:
                    self.model = SEPreActResNet(model_type, num_classes)
                else:
                    self.model = SEResNet(model_type, num_classes)
            else:
                if 'preact' in model_type:
                    self.model = PreActResNet(model_type, num_classes)
                else:
                    self.model = ResNet(model_type, num_classes)
        elif 'densenet' in model_type:
            self.model = DenseNet(model_type, num_classes)
        elif 'mobilenet' in model_type:
            if 'v2' not in model_type:
                self.model = MobileNet(num_classes)
            else:
                self.model = MobileNetV2(num_classes)
        else:
            sys.exit(ValueError("{:s} is currently not supported.".format(model_type)))
        
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        learning_rate_fn = tf.keras.experimental.CosineDecay(args.lr, decay_steps=decay_steps)
        with strategy.scope():
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
        self.weight_decay = 5e-4
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            # Cross-entropy loss
            ce_loss = self.loss_object(labels, predictions)
            # L2 loss(weight decay)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
            loss = ce_loss + l2_loss*self.weight_decay
            loss = tf.nn.compute_average_loss(loss, global_batch_size=args.batch_size)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

        return loss

    @tf.function
    def distributed_train_step(self, images, labels):
        losses = strategy.run(self.train_step, args=(images, labels,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        t_loss = tf.nn.compute_average_loss(t_loss, global_batch_size=args.batch_size)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

        return t_loss

    @tf.function
    def distributed_test_step(self, images, labels):
        losses = strategy.run(self.test_step, args=(images, labels,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
        
    def train(self, train_ds, test_ds, epoch):
        best_acc = tf.Variable(0.0)
        curr_epoch = tf.Variable(0)  # start from epoch 0 or last checkpoint epoch
        ckpt_path = './checkpoints/{:s}/'.format(args.model)
        ckpt = tf.train.Checkpoint(curr_epoch=curr_epoch, best_acc=best_acc,
                                   optimizer=self.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
        
        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint...')
            assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'

            # Restore the weights
            ckpt.restore(manager.latest_checkpoint)
        
        for e in (range(int(curr_epoch), epoch)):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for images, labels in train_ds:
                self.distributed_train_step(images, labels)
                
            for images, labels in test_ds:
                self.distributed_test_step(images, labels)

            template = 'Epoch {:0}, Loss: {:.4f}, Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'
            print (template.format(e+1,
                                   self.train_loss.result(),
                                   self.train_accuracy.result()*100,
                                   self.test_loss.result(),
                                   self.test_accuracy.result()*100))
            
            # Save checkpoint
            if self.test_accuracy.result() > best_acc:
                print('Saving...')
                if not os.path.isdir('./checkpoints/'):
                    os.mkdir('./checkpoints/')
                if not os.path.isdir(ckpt_path):
                    os.mkdir(ckpt_path)
                best_acc.assign(self.test_accuracy.result())
                curr_epoch.assign(e+1)
                manager.save()
    
    def predict(self, pred_ds, best):
        if best:
            ckpt_path = './checkpoints/{:s}/'.format(args.model)
            ckpt = tf.train.Checkpoint(model=self.model)
            manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
            
            # Load checkpoint
            print('==> Resuming from checkpoint...')
            assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'
            ckpt.restore(manager.latest_checkpoint)
        
        self.test_accuracy.reset_states()
        for images, labels in pred_ds:
            self.distributed_test_step(images, labels)
        print ('Prediction Accuracy: {:.2f}%'.format(self.test_accuracy.result()*100))

def main():
    # Data
    print('==> Preparing data...')
    train_images, train_labels, test_images, test_labels = get_dataset()
    mean, std = get_mean_and_std(train_images)
    train_images = normalize(train_images, mean, std)
    test_images = normalize(test_images, mean, std)

    train_ds = dataset_generator(train_images, train_labels, args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).\
            batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    test_ds = strategy.experimental_distribute_dataset(test_ds)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    decay_steps = int(args.epoch*len(train_images)/args.batch_size)
    
    # Train
    print('==> Building model...')
    with strategy.scope():
        model = Model(args.model, decay_steps)

    time_start = time.time()
    print("start time:", time_start)
    model.train(train_ds, test_ds, args.epoch)
    
    # Evaluate
    model.predict(test_ds, best=True)
    time_end = time.time()
    print("Total time:", time_end - time_start)

if __name__ == "__main__":
    main()
