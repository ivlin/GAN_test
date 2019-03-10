import sys, math
#ams math
import numpy as np
import tensorflow as tf
#visualization
import seaborn as sb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
sb.set()

BATCH_SIZE=100
NUM_FEATURES=2
D_TO_G_STEPS=5 #k - ratio of discriminator to generator optimization steps
GENERATOR_LAYERS=[10,10,10]
DISCRIMINATOR_LAYERS=[10,10]
NUM_ITERATIONS=2000

help_clause="""USAGE: python my_gan.py [--generator-layers (int list)] [--discriminator (int list)] [--batch-size (int)] [--features (int)] [--k-ratio (int)]
        --generator-layers          takes a python-formatted list of integers representing the width of each hidden layer
        --discriminator-layers      takes a python-formatted list of integers representing the width of each hidden layer
        --batch-size                takes an integer representing the batch size
        --k-ratio                   takes an integer representing the ratio of discriminator updates to generator updates
        --features                  takes an integer representing the number of features (visualization not available for more than 2 features)
        --iterations                takes an integer representing the number of iterations
"""


# ARG PARSING
def load_params(arglist):
    global GENERATOR_LAYERS, DISCRIMINATOR_LAYERS, BATCH_SIZE, NUM_FEATURES, D_TO_G_STEPS
    for i in xrange(len(arglist)):
        if arglist[i]=="--help" or arglist[i]=="-h":
            print help_clause
        if arglist[i]=="--generator-layers":
            GENERATOR_LAYERS=[int(num) for num in arglist[i+1][1:-1].split(",")]
        if arglist[i]=="--discriminator-layers":
            DISCRIMINATOR_LAYERS=[int(num) for num in arglist[i+1][1:-1].split(",")]
        if arglist[i]=="--batch-size":
            BATCH_SIZE=int(arglist[i+1])
        if arglist[i]=="--k-ratio":
            D_TO_G_STEPS=int(arglist[i+1])
        if arglist[i]=="--features":
            NUM_FEATURES=int(arglist[i+1])
        if arglist[i]=="--iterations":
            NUM_ITERATIONS=int(arglist[i+1])
#PART 1: DATA GENERATION

def generate_true_data(num_samples):
    sample_input=np.random.normal(0,5,(num_samples, NUM_FEATURES))
    for i in sample_input:
        i[1]=math.cos(i[0])
    return sample_input

def generate_uniform_data(num_samples):
    return np.random.uniform(-1.0,1.0,size=[num_samples, NUM_FEATURES])

#PART 2: GENERATOR AND DISCRIMINATOR
def minibatch(input_v, num_kernels=5, kernel_dim=3):
    # initializes a num_kernels by kernel_dim size tensor with weights generated of stddev 0.02
    # linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    with tf.variable_scope("minibatch",reuse=tf.AUTO_REUSE):
        #tensor of size input_rows x num_kernels*kernel_dim
        tens=tf.get_variable('minibatch_tensor', shape=[input_v.get_shape()[1], num_kernels*kernel_dim], \
            initializer=tf.random_normal_initializer(stddev=0.2))
        bias=tf.get_variable('minibatch_bias', num_kernels*kernel_dim)
    #apply the tensor and bias - produce a num_kernels*num_rows

    product = tf.matmul(input_v,tens)+bias
    #reshapes the transformed matrix into num_kernels x kernel_dim matrix
    activation = tf.reshape(product, (-1, num_kernels, kernel_dim))
    #add another column to the end of each activation row
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    #compute the L1 norm
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    #exponentiate
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input_v, minibatch_features], 1)

def generator(input_data, layer_sizes, reuse=False):
    #create shared variables that can be reaccessed later
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        layers=[input_data]
        for layer_size in layer_sizes:
            #tf.layers defines the interface for a densely defined layer that performs activation(input*kernel+bias)
            #returns input_data with the last dimension of size units
            layers.append(tf.layers.dense(inputs=layers[-1],\
                units=layer_size,\
                activation=tf.nn.leaky_relu))
        output=tf.layers.dense(layers[-1], NUM_FEATURES)
    return output

def discriminator(input_data, layer_sizes, reuse=False):
    #create shared variables that can be reaccessed later
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        layers=[input_data]
        for layer_size in layer_sizes:
            #tf.layers defines the interface for a densely defined layer that performs activation(input*kernel+bias)
            #returns input_data with the last dimension of size units
            layers.append(tf.layers.dense(inputs=layers[-1],\
                units=layer_size,\
                activation=tf.nn.leaky_relu))
        #layers.append(minibatch(layers[-1]))
        output=tf.layers.dense(layers[-1], 1)
    return output

if __name__=="__main__":
    ####################
    # LOAD PARAMS FROM USER INPUT
    ####################
    load_params(sys.argv)

    np.random.seed(0)
    tf.set_random_seed(0)
    ####################
    # VARIABLE INTIALIZATION STEP
    ####################
    #use placeholders to simulate operations on net input
    noise=tf.placeholder(tf.float32, [None, NUM_FEATURES])
    train_data=tf.placeholder(tf.float32, [None, NUM_FEATURES])
    #call the parameters that are to be optimized - the shared variables defined in gen and disc
    generator_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    discriminator_variables= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    ####################
    # FORWARD PROP
    ####################
    #generate data from generator
    generated_data=generator( noise, GENERATOR_LAYERS )
    #run discriminator
    generated_guesses=discriminator( generated_data, DISCRIMINATOR_LAYERS )
    #set reuse to be true since we're using same net - not updating between runs
    real_guesses=discriminator( train_data, DISCRIMINATOR_LAYERS, True)

    ####################
    # LOSS CALCULATION
    ####################
    #calculate losses - currently sigmoid cross entropy with logits
    #1->real data, 0->fake data
    sample_losses=tf.nn.sigmoid_cross_entropy_with_logits(logits=real_guesses,labels=tf.ones_like(real_guesses))\
                + tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_guesses,labels=tf.zeros_like(generated_guesses))
    discriminator_loss=tf.reduce_mean(sample_losses)
    generator_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_guesses,labels=tf.ones_like(generated_guesses)))

    ####################
    # BACKPROP + UPDATE STEP
    ####################
    #Call shared variables defined in layer calculations
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    #Update rule using rmsprop
    gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(generator_loss,var_list = gen_vars)
    disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(discriminator_loss,var_list = disc_vars)

    ####################
    # TENSORFLOW SESSION INITIALIZATION
    ####################
    init = tf.global_variables_initializer()#create global var initializer
    sess=tf.Session()                       #create new session for operation
    sess.run(init)                          #run the initializer

    ####################
    # INITIALIZE LOGGING
    ####################
    f = open('loss_logs.csv','w')
    f.write('Iteration,Discriminator Loss,Generator Loss\n')

    ####################
    # TRAINING
    ####################
    iterations=NUM_ITERATIONS
    #run the training
    for it in xrange(iterations):
        true_data=generate_true_data(BATCH_SIZE)
        uniform_noise=generate_uniform_data(BATCH_SIZE)
        #train discriminator for k steps
        for i in xrange(D_TO_G_STEPS):
            #1st argument: fetch: runs necessary graph fragments to generate each tensor in fetch
            abc, d_loss, ds=sess.run([generated_guesses,discriminator_loss,disc_step],feed_dict={noise:uniform_noise, \
                train_data:true_data})
            #print abc

        #train generator
        true_data=generate_true_data(BATCH_SIZE)
        d_loss, ds, g_data, g_loss, gs=sess.run([discriminator_loss,disc_step,generated_data,generator_loss,gen_step],\
            feed_dict={noise:generate_uniform_data(BATCH_SIZE), train_data:true_data})
        print it, " iteration: discriminator losss:", d_loss, " generator loss:", g_loss

        if it%10 == 0:
            f.write("%d,%f,%f\n"%(it,d_loss,g_loss))
        if it%100 == 0 and NUM_FEATURES==2:
            plt.figure()

            xax = plt.scatter(true_data[:,0], true_data[:,1],color="r")
            gax = plt.scatter(g_data[:,0], g_data[:,1],color="b")

            plt.legend((xax,gax), ("Real Data","Generated Data"))
            #plt.legend((xax), ("Real Data"))
            plt.title('Samples at Iteration %d'%it)
            plt.tight_layout()
            plt.savefig('./iterations/iteration_%d.png'%it)
            plt.close()
    f.close()