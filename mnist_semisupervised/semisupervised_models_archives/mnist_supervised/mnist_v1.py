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
                '''
                #hidden layer 1
                self.layers.append({'w':tf.Variable(tf.random_normal([100,100],stddev=0.000001),name="h1"),
                                'b':tf.Variable(tf.random_normal([1,100],stddev=0.0),name="h1_bias")})
                #hidden layer 2
                self.layers.append({'w':tf.Variable(tf.random_normal([100,256],stddev=0.000001),name="h2"),
                                'b':tf.Variable(tf.random_normal([1,256],stddev=0.0),name="h2_bias"),
                                'activation':tf.nn.leaky_relu})
                #hidden layer 3
                self.layers.append({'w':tf.Variable(tf.random_normal([256,512],stddev=0.000001),name="h3"),
                                'b':tf.Variable(tf.random_normal([1,512],stddev=0.0),name="h3_bias"),
                                'activation':tf.nn.leaky_relu})
                #hidden layer 4
                self.layers.append({'w':tf.Variable(tf.random_normal([512,1024],stddev=0.000001),name="h4"),
                                'b':tf.Variable(tf.random_normal([1,1024],stddev=0.0),name="h4_bias"),
                                'activation':tf.nn.leaky_relu})
                '''
                #output
                self.layers.append({'w':tf.Variable(tf.random_normal([100,784],stddev=0.00001),name="out"),
                                'b':tf.Variable(tf.random_normal([1,784],stddev=0.0),name="out_bias")})
    def calculate(self,x):
        intermediate=x
        for layer in self.layers:
            if 'activation' not in layer:
                intermediate=(tf.matmul(intermediate,layer['w'])+layer['b'])
            else:
                intermediate=layer['activation'](tf.matmul(intermediate,layer['w'])+layer['b'])
        return intermediate,self.layers[0]['w']
    def toString(self):
        print(layers)

def generator(x,weights=None):
    if weights==None:
        with tf.variable_scope("GAN/Generator"):
            #hidden layer 1
            h1=tf.Variable(tf.random_normal([100,100],stddev=0.01),name="h1")
            h1_b=tf.Variable(tf.random_normal([1,100],stddev=0.0001),name="h1_bias")
            #h1=tf.get_variable("h1",[100,100],dtype=tf.float32,initializer=tf.initializers.random_uniform)
            #h1_b=tf.get_variable("h1_bias",[1,100],initializer=tf.initializers.random_uniform)
            h1_act=tf.matmul(x,h1)+h1_b
            #hidden layer 2
            h2=tf.Variable(tf.random_normal([100,256],stddev=0.01),name="h2")
            h2_b=tf.Variable(tf.random_normal([1,256],stddev=0.0001),name="h2_bias")
            #h2=tf.get_variable("h2",[100,256],initializer=tf.initializers.random_uniform)
            #h2_b=tf.get_variable("h2_bias",[1,256],initializer=tf.initializers.random_uniform)
            h2_act=tf.nn.leaky_relu(tf.matmul(h1_act,h2)+h2_b)
            #hidden layer 3
            h3=tf.Variable(tf.random_normal([256,512],stddev=0.01),name="h3")
            h3_b=tf.Variable(tf.random_normal([1,512],stddev=0.0001),name="h3_bias")
            #h3=tf.get_variable("h3",[256,512],initializer=tf.initializers.random_uniform)
            #h3_b=tf.get_variable("h3_bias",[1,512],initializer=tf.initializers.random_uniform)
            h3_act=tf.nn.leaky_relu(tf.matmul(h2_act,h3)+h3_b)
            #hidden layer 4
            h4=tf.Variable(tf.random_normal([512,1024],stddev=0.01),name="h4")
            h4_b=tf.Variable(tf.random_normal([1,1024],stddev=0.0001),name="h4_bias")
            #h4=tf.get_variable("h4",[512,1024],initializer=tf.initializers.random_uniform)
            #h4_b=tf.get_variable("h4_bias",[1,1024],initializer=tf.initializers.random_uniform)
            h4_act=tf.nn.leaky_relu(tf.matmul(h3_act,h4)+h4_b)
            #output
            out=tf.get_variable("out",[1024,784],initializer=tf.initializers.random_uniform)
            y=tf.matmul(h4_act,out)
    return y,h1_act

def discriminator(x,reuse=False,weights=None):
    if weights==None:
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            # Create the model
            #hidden layer 1
            h1=tf.Variable(tf.random_uniform([784,10],minval=-0.01,maxval=0.01),name="h1")
            h1_b=tf.Variable(tf.random_uniform([1,10],minval=-0.01,maxval=0.01),name="h1_bias")
            #h1=tf.get_variable("h1",[784,10],initializer=tf.initializers.random_uniform)
            #h1_b=tf.get_variable("h1_bias",[1,10],initializer=tf.initializers.random_uniform)
            h1_act=tf.nn.softmax(tf.matmul(x,h1)+h1_b)
            #hidden layer 2
            #h2=tf.get_variable("h2",[512,256],initializer=tf.initializers.random_uniform)
            #h2_b=tf.get_variable("h2_bias",[1,1],initializer=tf.initializers.random_uniform)
            #h2_act=tf.nn.sigmoid(tf.matmul(h1_act,h2)+h2_b)
            #hidden layer 3
            #h3=tf.get_variable("h3",[256,128],initializer=tf.initializers.random_uniform)
            #h3_b=tf.get_variable("h3_bias",[1,128],initializer=tf.initializers.random_uniform)
            #h3_act=tf.nn.sigmoid(tf.matmul(h2_act,h3)+h3_b)
            #output
            #out=tf.get_variable("out",[512,10],initializer=tf.initializers.random_uniform)
            y=h1_act#tf.matmul(h1_act,out)
    else:
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            h1=weights[0]
            h1_b=weights[1]
            y=tf.nn.softmax(tf.matmul(x,h1)+h1_b)
    return y

def train_gan(args):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    x = tf.placeholder(tf.float32, [None, 784])
    y_labels = tf.placeholder(tf.int64, [None])
    y = discriminator(x)

    generators=[]
    blah=[]
    fake_x_vals=[]
    fake_y_vals=[]

    batch_size=tf.Variable(100)

    for ind in xrange(10):
        generators.append(Network())
        a,b=generators[-1].calculate(uniform_data_gen(batch_size))
        fake_x_vals.append(a)
        blah.append(b)
        fake_y_vals.append(discriminator(fake_x_vals[-1]))

    #fake_y=discriminator(fake_x,reuse=True)

    #generators
    generators_loss=[]
    generators_optimizers=[]
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    for ind in xrange(len(generators)):
        generators_loss.append(tf.losses.sparse_softmax_cross_entropy(labels=ind*tf.ones([100],dtype=tf.int64), logits=fake_y_vals[ind]))
        generators_optimizers.append(tf.train.GradientDescentOptimizer(0.1).minimize(generators_loss[ind],var_list=gen_vars))

    counter=tf.Variable(0)
    for i in generators_optimizers:
        if i==None:
            counter+=1

    #disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.ones_like(y), logits=y)+\
    #    tf.nn.softmax_cross_entropy_with_logits(labels=tf.zeros_like(fake_y), logits=fake_y))

    disc_loss=tf.losses.sparse_softmax_cross_entropy(labels=y_labels, logits=y)
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")
    disc_train_step = tf.train.GradientDescentOptimizer(0.1).minimize(disc_loss,var_list=disc_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()

    #Train
    for _ in xrange(1000):
        print("Iteration ",_)
        batch_xs, batch_ys = mnist.train.next_batch(100)
        for d_to_g in xrange(10):
            r,g,d_loss,d_steps=sess.run([y,y_labels,disc_loss, disc_train_step], feed_dict={x: batch_xs, y_labels: batch_ys})
        qwert,asd,g_loss,g_steps=sess.run([counter,blah,generators_loss, generators_optimizers], feed_dict={x: batch_xs, y_labels: batch_ys})
        print("Discriminator loss: ",d_loss, d_steps,r.shape,g.shape)
        print(asd)
        #print("Generator loss: ",g_loss, g_steps)

    if "--save" in args:
        saver.save(sess, './models/mnist_test_model',global_step=1000)

    print("\nTESTING DISCRIMINATOR ON MNIST DATA\n")
    batch_xs, batch_ys = mnist.train.next_batch(1)
    image_read.draw_ascii(np.asarray(batch_xs).reshape(-1),printOut=True)
    output=sess.run([y], feed_dict={x: batch_xs, y_labels: batch_ys})

    print("\nTESTING GENERATORS OUTPUT\n")
    ignore,output=sess.run([counter,fake_x_vals],feed_dict={batch_size:1})
    for gen in xrange(len(output)):
        print("GENERATOR ",gen)
        image_read.draw_ascii(np.asarray(output[gen]).reshape(-1),printOut=True)

def load_gan(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    sess=tf.Session()
    saver=tf.train.import_meta_graph('./models/mnist_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint("./models"))
    graph=tf.get_default_graph()

    x = tf.placeholder(tf.float32, [None, 784])
    y_labels=tf.placeholder(tf.int64, [None])

    discriminator=Network([{'w':graph.get_tensor_by_name("GAN/Discriminator/h1:0"),\
                    'b':graph.get_tensor_by_name("GAN/Discriminator/h1_bias:0"),\
                    'activation':tf.nn.softmax}])
    testout,xyz=discriminator.calculate(x)

    batch_xs, batch_ys = mnist.train.next_batch(1)
    image_read.draw_ascii(np.asarray(batch_xs).reshape(-1),printOut=True)
    output=sess.run([testout], feed_dict={x: batch_xs, y_labels: batch_ys})
    print(output)
    print(type(batch_ys))

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