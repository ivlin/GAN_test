# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,"./models_softmax_batch_110_0005/mnist_test_model"
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
import time

#custom libraries
from models_ss import Generator, Discriminator
import input_data

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
batch_size=110
model_file="./models/mnist_test_model"#"./models_semisupervised_55/mnist_test_model"
output_dir="./output/semisup_train_size/55000"
def generate_identity():
    return tf.ones([1,784])

def uniform_data_gen(batch_size):
    return np.random.uniform(0.0,1.0,(batch_size,100))

def normal_data_gen(batch_size):
    return np.random.normal(0.0,1.0,(batch_size,100))

def calculate_loss(true_logit,true_prob,true_features,labels,fake_logit,fake_prob,fake_features):#labels WILL be onehot
    #add "FAKE" class
    ss_labels=tf.pad(labels,[[0,0],[0,1]],"CONSTANT")
    #ss_labels=tf.concat([labels,tf.zeros([tf.shape(labels)[0], 1])],axis=1)
    #Classifier Loss= L_unsupervised+L_supervised
    #L_supervised=cross-entropy loss of true labelled data (true data labelled true but misclassified)
    #L_unsupervised=cross-entropy loss of (1- prob true data labelled fake)+ 
    #               cross-entropy loss of (fake data labelled as fake)
    #              =cross entropy loss of (true data labelled true) + cross entropy loss of (fake data labelled fake)
    #Note this is the standard GAN minimax equation:
    #L_unsupervised=CE-loss of (D(X) correctly labelled) + CE-loss of D(G(z)) correctly labelled)
    l_sup=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ss_labels,logits=true_logit))
    l_unsup=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(true_prob)[0]),logits=true_prob[:,-1]))+\
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros(tf.shape(fake_prob)[0]),logits=fake_prob[:,-1]))
    disc_loss=l_sup+l_unsup
    #Generator class
    #minimize the l2 distance between two features
    l_unsup_gen=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(true_prob)[0]),logits=true_prob[:,-1]))+\
                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros(tf.shape(fake_prob)[0]),logits=fake_prob[:,-1]))
    gen_loss=tf.reduce_mean(tf.square(true_features-fake_features))
    return gen_loss, disc_loss, l_sup

def train_gan(args):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, train_size=55000)

    gen=Generator()
    disc=Discriminator()

    keep_prob=tf.placeholder(tf.float32, None)
    is_training=tf.placeholder(tf.bool, None)

    x = tf.placeholder(tf.float32, [None, 784])
    y_labels = tf.placeholder(tf.int32, [None,10])
    y, y_prob, y_feature = disc.classify(tf.reshape(x,[-1,28,28,1]),keep_prob,is_training)

    noise=tf.placeholder(tf.float32,[None,100])

    fake_x=gen.generate(noise,keep_prob,is_training)
    fake_y, fake_y_prob, fake_y_feature= disc.classify(fake_x,keep_prob,is_training,reuse=True)

    #generators
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    #disc_reg=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-8), disc_vars)
    #gen_reg=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-8), gen_vars)

    gen_loss, disc_loss, sup_loss = calculate_loss(y, y_prob, y_feature, y_labels, fake_y, fake_y_prob, fake_y_feature)

    disc_train_step = tf.train.AdamOptimizer(0.0005,beta1=0.5).minimize(disc_loss,var_list=disc_vars)
    gen_train_step = tf.train.AdamOptimizer(0.0005,beta1=0.5).minimize(gen_loss,var_list=gen_vars)

    #classifier accuracy
    y_labels_mod=tf.pad(y_labels,[[0,0],[0,1]])
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels_mod, 1))
    classifier_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #discriminator accuracy - assumes all data is fake
    fake_test_labels=tf.concat([tf.zeros([tf.shape(fake_y)[0], 10]), tf.ones([tf.shape(fake_y)[0],1])], 1)
    predict_fake = tf.equal(tf.argmax(fake_y, 1), tf.argmax(fake_test_labels, 1))
    fake_id_accuracy = tf.reduce_mean(tf.cast(predict_fake, tf.float32))

    #discriminator accuracy - assumes all data is real
    real_test_labels=tf.concat([tf.zeros([tf.shape(y)[0], 10]), tf.ones([tf.shape(y)[0],1])], 1)
    predict_real = tf.equal(tf.argmax(y, 1), tf.argmax(real_test_labels, 1))
    real_id_accuracy = 1- tf.reduce_mean(tf.cast(predict_real, tf.float32))


    init=tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver=tf.train.Saver()
    if "--load" in args or "--test" in args:
        saver.restore(sess,model_file)

    show_z=normal_data_gen(16)
    g_loss=0
    d_loss=0
    past_glosses=[]
    past_dlosses=[]

    training_log=open("%s/training_log"%(output_dir),"w")
    training_log.write("accuracy,loss,\n")
    testing_log=open("%s/testing_log"%(output_dir),"w")
    testing_log.write("accuracy,loss,\n")
    
    #Train
    if "--test" not in args:
        start=time.time()

        batches_in_epoch=55000//batch_size
        num_epochs=5
        for _ in range(num_epochs*batches_in_epoch+1):
        #for _ in range(50):
            print("Iteration ",_," of ",str(batches_in_epoch))
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            uniform_noise=normal_data_gen(batch_size)
            acc, d_loss, d_steps, g_loss, g_steps = sess.run([classifier_accuracy, disc_loss, disc_train_step,gen_loss, gen_train_step], \
                                        feed_dict={keep_prob:0.7, is_training:True, \
                                        x: batch_xs, y_labels: batch_ys, noise: uniform_noise})
        
            #uniform_noise=normal_data_gen(batch_size)
            #g_loss,g_steps=sess.run([gen_loss, gen_train_step], feed_dict={keep_prob:0.7, is_training:True,\
            #                        x: batch_xs, y_labels: batch_ys, noise: uniform_noise})
        
            print("Discriminator loss: ",d_loss, d_steps)
            print("Generator loss: ",g_loss, g_steps)

            if _%10 == 0:
                training_log.write("%f,%f,\n"%(acc,d_loss))
                past_glosses.append(g_loss)
                past_dlosses.append(d_loss)


            if _%50 == 0:
                #Testing classifier
                accuracy_l = []
                loss_1 = []
                for iteration in range(20):
                    batch = mnist.test.next_batch(500, shuffle=False)
                    acc,loss=sess.run([classifier_accuracy,sup_loss],feed_dict={x: batch[0], 
                                                 y_labels: batch[1], 
                                                 keep_prob: 1.0,
                                                 is_training:True})
                    accuracy_l.append(acc)
                    loss_1.append(loss)
                    #accuracy_l.append(classifier_accuracy.eval(session=sess,feed_dict={x: batch[0], 
                    #                             y_labels: batch[1], 
                    #                             keep_prob: 1.0,
                    #                             is_training:True}))
                    #print(accuracy_l)
                    #print('test classifier accuracy %g' % np.mean(accuracy_l))
                testing_log.write("%f,%f,\n"%(np.mean(accuracy_l),np.mean(loss_1)))

                print("\nTESTING GENERATORS OUTPUT\n")
                for it in range(1):
                    x_val,y_val=sess.run([fake_x,fake_y],feed_dict={keep_prob:1.0, is_training:False, noise: show_z})
                    print("y val: ",y_val)
                    imgs = [img[:,:,0] for img in x_val]
                    m = montage(imgs)
                    gen_img = m
                    plt.axis('off')
                    plt.imshow(gen_img,cmap="gray")
                    plt.savefig("%s/it%d.png"%(output_dir,_))
                    plt.clf()
                    #plt.show()

                    plt.plot(np.linspace(0,len(past_dlosses),len(past_dlosses)),past_dlosses,label="dloss")
                    plt.plot(np.linspace(0,len(past_glosses),len(past_glosses)),past_glosses,label="gloss")
                    plt.title("DCGAN Loss")
                    plt.xlabel("Iteration")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig("%s/progress.png"%output_dir)
                    plt.clf()
                    #plt.show()

                if "--save" in args:
                    saver.save(sess, model_file)
        end=time.time()
        print("time elapse: ", str(end-start))
    
    #Testing classifier
    accuracy_l = []
    for _ in range(20):
        batch = mnist.test.next_batch(500, shuffle=False)
        accuracy_l.append(classifier_accuracy.eval(session=sess,feed_dict={x: batch[0], 
                                                 y_labels: batch[1], 
                                                 keep_prob: 1.0,
                                                 is_training:True}))
    print(accuracy_l)
    print('test classifier accuracy %g' % np.mean(accuracy_l))
    #Testing discriminator
    fake_identification = []
    real_identification = []
    for _ in range(20):
        batch_1 = mnist.test.next_batch(250, shuffle=False)
        
        real_identification.append(real_id_accuracy.eval(session=sess,feed_dict={x: batch_1[0], 
                                                 y_labels: batch_1[1], 
                                                 keep_prob: 1.0,
                                                 is_training:True}))
        fake_identification.append(fake_id_accuracy.eval(session=sess,feed_dict={noise: normal_data_gen(250),  
                                                 keep_prob: 1.0,
                                                 is_training:True}))
    print("success in identifying fakes")
    print(fake_identification)
    print("success in identifying reals")
    print(real_identification)
    print('test classifier accuracy %g' % np.mean(fake_identification+real_identification))


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