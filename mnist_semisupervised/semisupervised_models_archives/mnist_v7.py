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

def agenerator(z_prior,reuse=False):
    h1_size=150
    h2_size=300
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        # Fully Connected Layer 1 (100 (latent-vector) -> 150 (h1_size))
        w1 = tf.Variable(tf.truncated_normal([100, h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([h1_size]), name="g_b1", dtype=tf.float32)
        h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)

        # Fully Connected Layer 2 (150 (h1_size) -> 300 (h2_size))
        w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)
        b2 = tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

        # Fully Connected Layer 3 (300 (h1_size) -> input_height * input_width (img_size))
        w3 = tf.Variable(tf.truncated_normal([h2_size, 784], stddev=0.1), name="g_w3", dtype=tf.float32)
        b3 = tf.Variable(tf.zeros([784]), name="g_b3", dtype=tf.float32)
        h3 = tf.matmul(h2, w3) + b3

        x_generate = tf.nn.tanh(h3)

        g_params = [w1, b1, w2, b2, w3, b3]

    return x_generate

def generator(x,reuse=False,weights=None):
    if weights==None:
        with tf.variable_scope("GAN/Generator",reuse=reuse):
            intermediate=tf.layers.dense(x,units=784)
            intermediate2=tf.reshape(intermediate,[-1,28,28,1])
            '''
            intermediate1=tf.layers.dense(intermediate,units=200,activation=tf.nn.leaky_relu)
            intermediate2=tf.layers.dense(intermediate1,units=300,activation=tf.nn.leaky_relu)
            intermediate3=tf.layers.dense(intermediate2,units=400,activation=tf.nn.leaky_relu)
            intermediate4=tf.layers.dense(intermediate3,units=784,activation=tf.nn.tanh)
            #intermediate2=tf.reshape(intermediate1,[-1,28,28,1])
            # Create the model
            '''
            conv1=tf.layers.conv2d_transpose(inputs=intermediate2, filters=15, kernel_size=[5,5], padding="same", activation=tf.nn.leaky_relu)
            conv2=tf.layers.conv2d_transpose(inputs=conv1, filters=15, kernel_size=[5,5], padding="same", activation=tf.nn.leaky_relu)
            conv3=tf.layers.conv2d_transpose(inputs=conv2, filters=15, kernel_size=[3,3], padding="same", activation=tf.nn.leaky_relu)

            comb=tf.layers.conv2d_transpose(inputs=conv3, filters=1, kernel_size=[3,3], padding="same")#(tf.reshape(conv3,[-1,int(100*20)]),units=784)

            y=tf.reshape(comb,[-1,784])
    return y

#testing - just use an 11th class to classify noise
def discriminator(x,reuse=False,weights=None):
    #x=tf.reshape(x,[-1,28,28,1])
    if weights==None:
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            h1=tf.layers.dense(x,units=300,activation=tf.nn.leaky_relu)
            d1=tf.nn.dropout(h1, 0.5)
            h2=tf.layers.dense(d1,units=150,activation=tf.nn.leaky_relu)
            d2=tf.nn.dropout(h2, 0.5)
            h3=tf.layers.dense(d2,units=11)
            y=tf.nn.dropout(h3, 0.7)

            #y=tf.nn.softmax(d3)
            '''
            # Create the model
            #hidden layer 1
            conv1=tf.layers.conv2d(inputs=x, filters=15, kernel_size=[5,5], padding="same", activation=tf.nn.leaky_relu)
            pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
            pool1_flat=tf.reshape(pool1,[-1,int(784/4*15)])
            h1=tf.Variable(tf.random_normal([int(784/4*15),11],stddev=0.0000001),name="h1")
            h1_b=tf.Variable(tf.random_uniform([1,11],minval=-0.00,maxval=0.0),name="h1_bias")
            y=tf.matmul(pool1_flat,h1)+h1_b
            '''
            print(y.get_shape() )
    return y

def train_gan(args):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    x = tf.placeholder(tf.float32, [None, 784])
    y_labels = tf.placeholder(tf.int32, [None])
    y = discriminator(x)
    noise= tf.placeholder(tf.float32,[None,100])

    batch_size=tf.Variable(100)
    #gen=generator()
    fake_x=generator(noise)
    fake_y=discriminator(fake_x,True)

    #generators
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    disc_loss = tf.losses.sparse_softmax_cross_entropy(labels=y_labels, logits=y)+\
        tf.losses.sparse_softmax_cross_entropy(labels=10*tf.ones_like(y_labels), logits=fake_y)
    gen_loss = tf.losses.sparse_softmax_cross_entropy(labels=7*tf.ones_like(y_labels), logits=fake_y)

    gen_train_step = tf.train.AdamOptimizer(0.0001).minimize(gen_loss,var_list=gen_vars)
    disc_train_step = tf.train.AdamOptimizer(0.0001).minimize(disc_loss,var_list=disc_vars)

    init=tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver=tf.train.Saver()

    g_loss=0
    d_loss=0
    #Train
    for _ in xrange(5000):
        print("Iteration ",_)
        batch_xs, batch_ys = mnist.train.next_batch(100)
        #for d_to_g in xrange(5):
        if g_loss<= 5*d_loss:
            d_loss, d_steps = sess.run([disc_loss, disc_train_step], feed_dict={x: batch_xs, y_labels: batch_ys, noise: uniform_data_gen(100).eval(session=sess)})
        #for i in xrange(5):
        g_loss,g_steps=sess.run([gen_loss, gen_train_step], feed_dict={x: batch_xs, y_labels: batch_ys, noise: uniform_data_gen(100).eval(session=sess)})
        print("Discriminator loss: ",d_loss, d_steps)
        print("Generator loss: ",g_loss, g_steps)
        if _%50 == 0:
            print("\nTESTING DISCRIMINATOR ON MNIST DATA\n")
            for it in xrange(5):
                batch_xs, batch_ys = mnist.train.next_batch(1)
                image_read.draw_ascii(np.asarray(batch_xs).reshape(-1),printOut=True)
                output=sess.run([y], feed_dict={x: batch_xs, y_labels: batch_ys, noise: uniform_data_gen(100).eval(session=sess)})
                print("y guess: ",output)
                print("y val: ",batch_ys)
                batch_xs, batch_ys = mnist.train.next_batch(1)
            print("\nTESTING GENERATORS OUTPUT\n")
            for it in xrange(2):
                x_val,y_val=sess.run([fake_x,fake_y],feed_dict={noise: uniform_data_gen(1).eval(session=sess)})
                image_read.draw_ascii(np.asarray(x_val).reshape(-1),printOut=True)
                print("y val: ",y_val)
    if "--save" in args:
        saver.save(sess, './models/mnist_test_model',global_step=1000)
    print("\nTESTING DISCRIMINATOR ON MNIST DATA\n")
    for it in xrange(5):
        batch_xs, batch_ys = mnist.train.next_batch(1)
        image_read.draw_ascii(np.asarray(batch_xs).reshape(-1),printOut=True)
        output=sess.run([y], feed_dict={x: batch_xs, y_labels: batch_ys, noise: uniform_data_gen(100).eval(session=sess)})
        print("y guess: ",output)
        print("y val: ",batch_ys)
        batch_xs, batch_ys = mnist.train.next_batch(1)
    print("\nTESTING GENERATORS OUTPUT\n")
    for it in xrange(2):
        x_val,y_val=sess.run([fake_x,fake_y],feed_dict={noise: uniform_data_gen(1).eval(session=sess)})
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