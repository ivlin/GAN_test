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
import os
from models import Generator, Discriminator

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)

NUM_ITERATIONS=1000
FLAGS = None

def mont2(images):
    tensor_shape = images.shape#get_shape().as_list()
    img_h = tensor_shape[1]
    img_w = tensor_shape[2]
    n_plots=0
    if tensor_shape[0] is not None:
        n_plots = int(np.ceil( tensor_shape[0]**0.5 ))
    m = np.ones( (tensor_shape[1] * n_plots + n_plots + 1, tensor_shape[2] * n_plots + n_plots + 1, tensor_shape[3])) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < tensor_shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return tf.image.encode_jpeg( tf.convert_to_tensor(m, tf.uint8), format='rgb', quality=100 ) #m

# Code by Parag Mital (github.com/pkmital/CADL)
def montage(images):
    #if isinstance(images, list):
    #    images = np.array(images)
    tensor_shape = images.get_shape().as_list()#images.get_shape().as_list()
    img_h = tensor_shape[1]
    img_w = tensor_shape[2]
    n_plots=0
    if tensor_shape[0] is not None:
        n_plots = int(np.ceil( tensor_shape[0]**0.5 ))
    m = np.ones((tensor_shape[1] * n_plots + n_plots + 1, tensor_shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < tensor_shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    '''
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
    '''
    return m

def generate_identity():
    return tf.ones([1,784])

def uniform_data_gen(batch_size):
    return np.random.uniform(0.0,1.0,(batch_size,512))

def normal_data_gen(batch_size):
    return np.random.normal(0.0,1.0,(batch_size,512))

def next_batch(data_iterator, data_size, batch_num, batch_size):
    return data_iterator.get_next()

#load code taken from https://stackoverflow.com/questions/50562287/tensorflow-read-and-decode-batch-of-images user Aldream
def parse_imgs(filename):
    return tf.squeeze (tf.image.resize_nearest_neighbor( [tf.image.decode_jpeg( tf.read_file(filename), channels=3 )], size=[64,64] ))

def load_training_images(args, batch_size, data_dir="./car_data"):
    filenames=[os.path.join(dp, f) for dp, dn, filenames in os.walk(data_dir) for f in filenames if os.path.splitext(f)[1] == '.jpg']

    dataset = tf.data.Dataset.from_tensor_slices((filenames))
    dataset = dataset.map(parse_imgs).batch(batch_size)
    iterator = dataset.make_initializable_iterator()

    return iterator, len(filenames)

def setup_output_directory(args):
    output_dir="tmp"
    if "--out" in args:
        output_dir = args[args.index("--out")+1]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(("%s/model"%output_dir))
        os.mkdir(("%s/output"%output_dir))
        os.mkdir(("%s/output/train"%output_dir))
        os.mkdir(("%s/output/test"%output_dir))
    return output_dir

def get_jpg(data):
    return tf.image.encode_jpeg(data, format='rgb', quality=100)

def train_gan(args):
    gen=Generator()
    disc=Discriminator()

    drop_prob=tf.placeholder(tf.float32, None) #dropout keep probability
    is_training=tf.placeholder(tf.bool, None)

    x = tf.placeholder(tf.float32, [None, 64, 64, 3])
    y_labels = tf.placeholder(tf.int32, [None,1])
    y = disc.classify(tf.reshape(x,[-1,64,64,3]),drop_prob,is_training)

    noise=tf.placeholder(tf.float32,[None,512])
    fake_x=gen.generate(noise,drop_prob,is_training)
    fake_y=disc.classify(fake_x,drop_prob,is_training,reuse=True)

    jpg_test_fake = tf.image.encode_jpeg( tf.cast(fake_x[0],tf.uint8), format='rgb', quality=100 ) #tf.map_fn(get_jpg, int_fake_x)

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


    #load data
    batch_size=50
    output_dir=setup_output_directory(args)
    training_data_it, dataset_size = load_training_images(args, batch_size)
    print(dataset_size)

    sess.run(training_data_it.initializer)

    if "--load" in args or "--test" in args:
        saver.restore(sess,("%s/model"%(output_dir)))

    show_z=normal_data_gen(16)
    g_loss=0
    d_loss=0
    past_glosses=[]
    past_dlosses=[]

    #Train
    batches_in_epoch=dataset_size//batch_size

    num_epochs=25

    training_loss=open("%s/training_loss"%output_dir,"w")
    training_loss.write("generator loss, discriminator loss,\n")
    training_loss.close()

    current_batch = next_batch(training_data_it, dataset_size, 0, batch_size) #current_batch initialized outside loop to prevent resource exhaustion

    if "--test" not in args:
        start=time.time()

        for batch_num in range(batches_in_epoch*num_epochs+1):
            print("Iteration ",batch_num," of ",str(batches_in_epoch))

            try:
                batch_xs = sess.run(current_batch)#next_batch(training_data_it, dataset_size, batch_num, batch_size)
            except tf.errors.OutOfRangeError:
                sess.run(training_data_it.initializer)
                batch_xs = sess.run(current_batch)

            batch_ys = tf.ones([tf.shape(batch_xs)[0],1])
            uniform_noise=normal_data_gen(batch_size)

            d_loss, d_steps = sess.run([disc_loss, disc_train_step], feed_dict={drop_prob:0.3, is_training:True, \
                                    x: batch_xs,
                                    y_labels: sess.run(batch_ys),
                                    noise: uniform_noise})
            #print(batch_xs)
            uniform_noise=normal_data_gen(batch_size)
            g_loss,g_steps=sess.run([gen_loss, gen_train_step], feed_dict={drop_prob:0.3, is_training:True, noise: uniform_noise})

            print("Discriminator loss: ",d_loss, d_steps)
            print("Generator loss: ",g_loss, g_steps)

            if batch_num%10 == 0:
                past_glosses.append(g_loss)
                past_dlosses.append(d_loss)

                training_loss=open("%s/training_loss"%output_dir,"a")
                training_loss.write(str(g_loss))
                training_loss.write(",")
                training_loss.write(str(d_loss))
                training_loss.write(",\n")
                training_loss.close()

            if batch_num%50 == 0:
                print("\nGenerator Progress\n")

                for it in range(1):
                    x_val,y_val,singleton=sess.run([fake_x,fake_y, jpg_test_fake],feed_dict={drop_prob:0.0, is_training:False, noise: show_z})
                    print("y val: ",y_val)
                    imgs = [img[:,:,0] for img in x_val]

                    montaged=sess.run(mont2(x_val))

                    with open("%s/output/test/test_it%04d_single.jpg"%(output_dir,batch_num), "w") as f:
                        f.write(singleton)
                    with open("%s/output/test/test_it%04d_multi.jpg"%(output_dir,batch_num), "w") as f:
                        f.write(montaged)

                    #gen_img = montage(imgs)
                    #plt.axis('off')
                    #plt.imshow(gen_img)
                    #plt.savefig("%s/output/test_it%d.png"%(output_dir,batch_num))
                    #plt.clf()
                    #plt.show()

                    plt.plot(np.linspace(0,len(past_dlosses),len(past_dlosses)),past_dlosses,label="dloss")
                    plt.plot(np.linspace(0,len(past_glosses),len(past_glosses)),past_glosses,label="gloss")
                    plt.title('DCGAN Loss')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    #plt.savefig("%s/output/progress%d.png"%(output_dir,batch_num))
                    plt.clf()
                    #plt.show()

                for it in range(1):
                    x_val,y_val,singleton=sess.run([fake_x,fake_y, jpg_test_fake],feed_dict={x:batch_xs, drop_prob:0.3, is_training:False, noise: show_z})

                    montaged=sess.run(mont2(x_val))

                    with open("%s/output/train/train_it%04d_single.jpg"%(output_dir,batch_num), "w") as f:
                        f.write(singleton)
                    with open("%s/output/train/train_it%04d_multi.jpg"%(output_dir,batch_num), "w") as f:
                        f.write(montaged)

                    print("y val: ",y_val)
                    imgs = [img[:,:,0] for img in x_val]

                    #gen_img = montage(imgs)
                    #plt.axis('off')
                    #plt.imshow(gen_img)
                    #plt.savefig("%s/output/train_it%04d.jpg"%(output_dir,batch_num))
                    #plt.clf()
                    #plt.show()

                saver.save(sess, ('%s/model_it%d'%(output_dir,batch_num)), write_meta_graph=False)

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
      default='/tmp/tensorflow/stanford_car/input_data',
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)