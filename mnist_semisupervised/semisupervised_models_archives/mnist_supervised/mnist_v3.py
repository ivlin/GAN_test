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

import image_read
import numpy as np
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

tf.set_random_seed(0)

NUM_ITERATIONS=1000
FLAGS = None

def generate_identity():
    return tf.ones([1,784])

def uniform_data_gen(batch_size):
    return tf.random_uniform([batch_size,100])

class Network:
    #weights will simply consist of a duple of weight-bias dicts
    def __init__(self, weights=None):
        self.layers=weights
        if weights==None:
            self.layers=[]
            with tf.variable_scope("GAN/Generator"):
                self.layers.append({'w':tf.Variable(tf.random_normal([100,100],stddev=1.0),name="h1"),
                                'b':tf.Variable(tf.random_normal([1,100],stddev=0.0),name="h1_bias"),
                                'activation':tf.nn.sigmoid})
                self.layers.append({'w':tf.Variable(tf.random_normal([100,256],stddev=0.1),name="h2"),
                                'b':tf.Variable(tf.random_normal([1,256],stddev=0.0),name="h2_bias"),
                                'activation':tf.nn.sigmoid})
                self.layers.append({'w':tf.Variable(tf.random_normal([256,512],stddev=0.1),name="h3"),
                                'b':tf.Variable(tf.random_normal([1,512],stddev=0.0),name="h3_bias"),
                                'activation':tf.nn.sigmoid})
                self.layers.append({'w':tf.Variable(tf.random_normal([512,1024],stddev=0.1),name="h4"),
                                'b':tf.Variable(tf.random_normal([1,1024],stddev=0.0),name="h4_bias"),
                                'activation':tf.nn.sigmoid})
                self.layers.append({'w':tf.Variable(tf.random_normal([1024,2048],stddev=0.1),name="h5"),
                                'b':tf.Variable(tf.random_normal([1,2048],stddev=0.0),name="h5_bias"),
                                'activation':tf.nn.sigmoid})
                #output
                self.layers.append({'w':tf.Variable(tf.random_normal([2048,784],stddev=1.0),name="out"),
                                'b':tf.Variable(tf.random_normal([1,784],stddev=0.0),name="out_bias")})
    def calculate(self,x):
        intermediate=x
        for layer in self.layers:
            if 'activation' not in layer:
                intermediate=(tf.matmul(intermediate,layer['w'])+layer['b'])
            else:
                intermediate=layer['activation'](tf.matmul(intermediate,layer['w'])+layer['b'])
        return intermediate
    def toString(self):
        print(layers)

def generator(x,reuse=False,weights=None):
    if weights==None:
        with tf.variable_scope("GAN/Generator",reuse=reuse):
            # Create the model
            #convolution
            conv1=tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.leaky_relu)
            pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
            #hidden layer 1
            h1=tf.Variable(tf.random_normal([100,784],stddev=1.0),name="h1")
            h1_b=tf.Variable(tf.random_normal([1,784],stddev=0.0),name="h1_bias")
            y=tf.matmul(pool1,h1)+h1_b
    return y

def discriminator(x,reuse=False,weights=None):
    #x=tf.convert_to_tensor(x,dtype=tf.float32)

    x=tf.reshape(x,[-1,28,28,1])

    #for ind in xrange(len(x)):
    #    x[ind]=tf.reshape(x[ind],[-1,28,28,1])
    if weights==None:
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            # Create the model
            #hidden layer 1
            conv1=tf.layers.conv2d(inputs=x, filters=1, kernel_size=[5,5], padding="same", activation=tf.nn.leaky_relu)
            #pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

            pool2=tf.reshape(conv1,[-1,784])

            h1=tf.Variable(tf.random_normal([784,10],stddev=0.0000001),name="h1")
            h1_b=tf.Variable(tf.random_uniform([1,10],minval=-0.00,maxval=0.0),name="h1_bias")
            y=tf.matmul(pool2,h1)+h1_b
    return y

def train_gan(args):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    x = tf.placeholder(tf.float32, [None, 784])
    y_labels = tf.placeholder(tf.int64, [None])
    y = discriminator(x)

    batch_size=tf.Variable(100)
    gen=Network()
    fake_x=gen.calculate(uniform_data_gen(batch_size))
    fake_y=discriminator(fake_x,True)

    #generators
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    #disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy(labels=tf.ones_like(y), logits=y))+\
    #    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy(labels=tf.zeros_like(fake_y), logits=fake_y))
    disc_loss = tf.losses.sparse_softmax_cross_entropy(labels=y_labels, logits=y)
    #gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_y),logits=fake_y))

    #gen_train_step = tf.train.AdamOptimizer(0.0001).minimize(gen_loss,var_list=gen_vars)
    disc_train_step = tf.train.AdamOptimizer(0.0001).minimize(disc_loss,var_list=disc_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()

    #Train
    for _ in xrange(100):
        print("Iteration ",_)
        batch_xs, batch_ys = mnist.train.next_batch(100)
        for d_to_g in xrange(5):
            d_loss,d_steps=sess.run([disc_loss, disc_train_step], feed_dict={x: batch_xs, y_labels: batch_ys})
        #g_loss,g_steps=sess.run([gen_loss, gen_train_step], feed_dict={x: batch_xs_mod, y_labels: batch_ys_mod})
        print("Discriminator loss: ",d_loss, d_steps)
        #print("Generator loss: ",g_loss, g_steps)

    if "--save" in args:
        saver.save(sess, './models/mnist_test_model',global_step=1000)

    print("\nTESTING DISCRIMINATOR ON MNIST DATA\n")
    batch_xs, batch_ys = mnist.train.next_batch(1)
    image_read.draw_ascii(np.asarray(batch_xs).reshape(-1),printOut=True)
    output=sess.run([y], feed_dict={x: batch_xs, y_labels: batch_ys})
    print("y guess: ",output)
    print("y val: ",batch_ys)

    print("\nTESTING GENERATORS OUTPUT\n")
    x_val,y_val=sess.run([fake_x,fake_y],feed_dict={batch_size:1})
    image_read.draw_ascii(np.asarray(x_val).reshape(-1),printOut=True)
    print("y val: ",y_val)

def main(_):
    if "--load" in _:
        load_gan(_)
    else:
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