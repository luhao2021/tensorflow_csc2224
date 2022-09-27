import os
import time

import tensorflow as tf  # TF 2.0
import tensorflow_datasets as tfds

from model import Generator, Discriminator
from utils import generator_loss, discriminator_loss, save_imgs, preprocess_image 

random_seed=123
from numpy.random import seed 
seed(random_seed)

tf.random.set_seed(random_seed)
import random
random.seed(random_seed)


import os
os.environ['PYTHONHASHSEED'] = '0'

def train():
    # data, info = tfds.load("lsun/bedroom", with_info=True, data_dir='/data/tensorflow_datasets')
    data, info = tfds.load("mnist", with_info=True, data_dir='tensorflow_datasets')
    train_data = data['train']

    if not os.path.exists('./images'):
        os.makedirs('./images')

    # settting hyperparameter
    latent_dim = 100
    epochs = 800
    batch_size = 200
    buffer_size = 6000
    save_interval = 50

    img_shape = (28, 28, 1)

    num_classes = info.features['label'].num_classes

    generator = Generator(num_classes)
    discriminator = Discriminator(num_classes)

    gen_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    disc_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

    train_dataset = train_data.map(lambda x: preprocess_image(x, img_shape, num_classes)).shuffle(buffer_size).batch(batch_size)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function(experimental_compile=True)
    def compiled_step(noise, images, labels):
        with tf.GradientTape(persistent=True) as tape:
            generated_images = generator(noise, labels)

            real_output = discriminator(images, labels)
            generated_output = discriminator(generated_images, labels)

            gen_loss = generator_loss(cross_entropy, generated_output)
            disc_loss = discriminator_loss(cross_entropy, real_output, generated_output)

        grad_gen = tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)
        return gen_loss, disc_loss, grad_gen, grad_disc



    @tf.function
    def train_step(images, labels):
        noise = tf.random.normal([batch_size, latent_dim]) 
        gen_loss, disc_loss, grad_gen, grad_disc = compiled_step(noise, images, labels)
        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss

    seed = tf.random.normal([16, latent_dim])

    for epoch in range(1, epochs + 1):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0

        for images, labels in train_dataset:
            gen_loss, disc_loss = train_step(images, labels)

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch, time.time() - start, total_gen_loss / batch_size, total_disc_loss / batch_size))
        # if epoch % save_interval == 0:
        #     save_imgs(epoch, generator, seed)


if __name__ == "__main__":
    train()
