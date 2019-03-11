# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import argparse
import sys
from models import Generator, Discriminator

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import matplotlib.pyplot as plt

# Code by Parag Mital (github.com/pkmital/CADL)
def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones((images.shape[1] * n_plots + n_plots + 1, images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m

np.random.seed(0)
tf.set_random_seed(0)

NUM_ITERATIONS=1000
FLAGS = None

def generate_identity():
    return tf.ones([1,784])

def uniform_data_gen(batch_size):
    return np.random.uniform(0.0,1.0,(batch_size,100))

def normal_data_gen(batch_size):
    return np.random.normal(0.0,1.0,(batch_size,100))

def train_gan(args):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True)

    gen=Generator()
    disc=Discriminator()

    keep_prob=tf.placeholder(tf.float32, None)
    is_training=tf.placeholder(tf.bool, None)

    x = tf.placeholder(tf.float32, [None, 784])
    y_labels = tf.placeholder(tf.int32, [None,10])
    y = disc.classify(tf.reshape(x,[-1,28,28,1]),keep_prob,is_training)

    noise=tf.placeholder(tf.float32,[None,100])

    fake_x=gen.generate(noise,keep_prob,is_training)
    blah1=fake_x
    fake_y=disc.classify(fake_x,keep_prob,is_training,reuse=True)

    #generators
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    disc_reg=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-8), disc_vars)
    gen_reg=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-8), gen_vars)

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y), logits=y))+\
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_y), logits=fake_y))

    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_y), logits=fake_y))

    disc_train_step = tf.train.AdamOptimizer(0.0005,beta1=0.5).minimize(disc_loss,var_list=disc_vars)
    gen_train_step = tf.train.AdamOptimizer(0.0005,beta1=0.5).minimize(gen_loss,var_list=gen_vars)

    init=tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver=tf.train.Saver()
    if "--load" in args or "--test" in args:
        saver.restore(sess,"./models_unsupervised_55/mnist_test_model")

    show_z=normal_data_gen(16)
    g_loss=0
    d_loss=0
    past_glosses=[]
    past_dlosses=[]


    #Train
    batches_in_epoch=55000//55
    num_epochs=5
    if "--test" not in args:
        start=time.time()
        for _ in range(batches_in_epoch*num_epochs+1):
            print("Iteration ",_," of ",str(batches_in_epoch))
            batch_xs, batch_ys = mnist.train.next_batch(55)
            uniform_noise=normal_data_gen(55)
            d_loss, d_steps = sess.run([disc_loss, disc_train_step], feed_dict={keep_prob:0.7, is_training:True, \
                                    x: batch_xs, y_labels: batch_ys, noise: uniform_noise})

            uniform_noise=normal_data_gen(55)
            g_loss,g_steps=sess.run([gen_loss, gen_train_step], feed_dict={keep_prob:0.7, is_training:True, noise: uniform_noise})

            print("Discriminator loss: ",d_loss, d_steps)
            print("Generator loss: ",g_loss, g_steps)

            if _%10 == 0:
                past_glosses.append(g_loss)
                past_dlosses.append(d_loss)

            if _%50 == 0:
                print("\nTESTING DISCRIMINATOR ON MNIST DATA\n")
                for it in range(0):
                    batch_xs, batch_ys = mnist.train.next_batch(16)
                    #image_read.draw_ascii(np.asarray(batch_xs).reshape(-1),printOut=True)
                    output=sess.run([y], feed_dict={keep_prob:1.0, is_training:False, x: batch_xs, y_labels: batch_ys, noise: uniform_data_gen(16)})

                    batch_xs=np.reshape(batch_xs,(-1,28,28,1))
                    imgs = [img[:,:,0] for img in batch_xs]
                    m = montage(imgs)
                    gen_img = m
                    plt.axis('off')
                    plt.imshow(gen_img,cmap="gray")
                    plt.show()

                    print("y guess: ",output)
                    print("y val: ",batch_ys)
                    batch_xs, batch_ys = mnist.train.next_batch(1)
                print("\nTESTING GENERATORS OUTPUT\n")
                for it in range(1):
                    x_val,y_val=sess.run([fake_x,fake_y],feed_dict={keep_prob:1.0, is_training:False, noise: show_z})
                    print("y val: ",y_val)
                    imgs = [img[:,:,0] for img in x_val]
                    m = montage(imgs)
                    gen_img = m
                    plt.axis('off')
                    plt.imshow(gen_img,cmap="gray")
                    plt.savefig("./output/it%d.png"%_)
                    plt.clf()
                    #plt.show()

                    plt.plot(np.linspace(0,len(past_dlosses),len(past_dlosses)),past_dlosses,label="dloss")
                    plt.plot(np.linspace(0,len(past_glosses),len(past_glosses)),past_glosses,label="gloss")
                    plt.title('DCGAN Loss')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig("./output/progress.png")
                    plt.clf()
                #   plt.show()

                if "--save" in args:
                  saver.save(sess, './models_unsupervised_55/mnist_test_model')
        end=time.time()
        print("time elapsed:", str(end-start))
    #Testing discriminator
    


def main(_):
    train_gan(_)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)