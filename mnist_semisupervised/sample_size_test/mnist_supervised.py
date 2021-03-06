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

import numpy as np
import argparse
import sys

import input_data

import tensorflow as tf

import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)

NUM_ITERATIONS=1000
FLAGS = None
batch_size=110
output_dir="./output/sup_train_size/00350"

class Discriminator:
    def __init__(self):
        pass

    def classify(self,x,keep,is_training,reuse=False):
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            norm_x=x-0.5

            a1=tf.nn.dropout(tf.layers.conv2d(norm_x,filters=4,kernel_size=(3,3),strides=(1,1),padding="SAME",activation=tf.nn.leaky_relu,name="conv1"),keep,name="drop1")#28
            n1=tf.layers.batch_normalization(a1,training=is_training,name="batch_norm1")

            a2=tf.nn.dropout(tf.layers.conv2d(n1,filters=32,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="conv2"),keep,name="drop2")#28
            n2=tf.layers.batch_normalization(a2,training=is_training,name="batch_norm2")
            
            a4=tf.nn.dropout(tf.layers.conv2d(n2,filters=512,kernel_size=(3,3),strides=(1,1),padding="SAME",activation=tf.nn.leaky_relu,name="conv4"),keep,name="drop4")#14
            n4=tf.layers.batch_normalization(a4,training=is_training,name="batch_norm4")
            
            a5=tf.nn.dropout(tf.layers.conv2d(n4,filters=1024,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="conv5"),keep,name="drop5")#14
            n5=tf.layers.batch_normalization(a5,training=is_training,name="batch_norm5")

            y=tf.layers.dense(tf.reshape(n5,[-1,1024*7*7]),units=10,name="out")
        return y

def train_gan(args):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True,train_size=350)

    disc=Discriminator()

    keep_prob=tf.placeholder(tf.float32, None)
    is_training=tf.placeholder(tf.bool, None)

    x = tf.placeholder(tf.float32, [None, 784])
    y_labels = tf.placeholder(tf.int32, [None,10])
    y = disc.classify(tf.reshape(x,[-1,28,28,1]),keep_prob,is_training)

    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_labels, logits=y))

    disc_train_step = tf.train.AdamOptimizer(0.0005,beta1=0.5).minimize(disc_loss)#,var_list=disc_vars)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
    classifier_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #correct_prediction = tf.equal(tf.argmax(tf.cast(y, tf.int32), 1), y_labels)
    #correct_prediction = tf.cast(correct_prediction, tf.float32)
    #accuracy = tf.reduce_mean(correct_prediction)

    init=tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver=tf.train.Saver()
    if "--test" not in args:        
        d_loss=0
        past_dlosses=[]

        train_loss=[]
        train_acc=[]
        test_loss=[]
        test_acc=[]

        training_log=open("%s/training_log"%(output_dir),"w")
        training_log.write("accuracy,loss,\n")
        testing_log=open("%s/testing_log"%(output_dir),"w")
        testing_log.write("accuracy,loss,\n")

        #Train
        batches_in_epoch=55000//batch_size
        num_epochs=5
        for _ in range(num_epochs*batches_in_epoch+1):
            print("Iteration ",_," of ",batches_in_epoch)
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            accuracy, d_loss, d_steps = sess.run([classifier_accuracy, disc_loss, disc_train_step], feed_dict={keep_prob:0.7, is_training:True, \
                                    x: batch_xs, y_labels: batch_ys})
            print("Discriminator loss: ",d_loss, d_steps," accuracy: ",accuracy)

            if _%10 == 0:

                training_log.write("%f,%f,\n"%(accuracy, d_loss))

                past_dlosses.append(d_loss)
            if _%50 == 0:
                if "--save" in args:
                    saver.save(sess, './models/mnist_test_model')
                
                train_loss.append(d_loss)
                train_acc.append(accuracy)

                #Testing classifier
                accuracy_current = []
                loss_current = []
                for iteration in range(20):
                    batch = mnist.test.next_batch(500, shuffle=False)
                    acc,loss=sess.run([classifier_accuracy,disc_loss],feed_dict={x: batch[0], 
                                                 y_labels: batch[1], 
                                                 keep_prob: 1.0,
                                                 is_training:True})
                    accuracy_current.append(acc)
                    loss_current.append(loss)
                
                test_acc.append(np.mean(accuracy_current))
                test_loss.append(np.mean(loss_current))

                testing_log.write("%f,%f,\n"%(np.mean(accuracy_current),np.mean(loss_current)))

                plt.plot(np.linspace(0,len(train_loss),len(train_loss)),train_loss,label="training loss")
                plt.plot(np.linspace(0,len(test_loss),len(test_loss)),test_loss,label="test loss")
                plt.plot(np.linspace(0,len(train_acc),len(train_acc)),train_acc,label="training accuracy")
                plt.plot(np.linspace(0,len(test_acc),len(test_acc)),test_acc,label="testing accuracy")
                
                plt.title("Training Progress")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig("%s/progress.png"%output_dir)
                #plt.show()
                plt.clf()
    else:
        saver.restore(sess,"./models/mnist_test_model")

    #Testing classifier
    accuracy_l = []
    for _ in range(20):
        batch = mnist.test.next_batch(500, shuffle=False)
        accuracy_l.append(classifier_accuracy.eval(session=sess,feed_dict={x: batch[0], 
                                                 y_labels: batch[1], 
                                                 keep_prob: 1.0,
                                                 is_training:True}))
    print(accuracy_l)
    print('test accuracy %g' % np.mean(accuracy_l))
    

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