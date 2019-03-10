import tensorflow as tf

class Generator:
    def __init__(self):
        pass

    def generate(self,x,keep,is_training):
        with tf.variable_scope("GAN/Generator",reuse=False):
            i_flat=tf.nn.dropout(tf.layers.dense(x,units=16*512),keep,name="drop0")
            i1=tf.reshape(i_flat,[-1,4,4,512])

            conv1=tf.nn.dropout(tf.layers.conv2d_transpose(i1,filters=1024,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="dconv1"), keep, name="drop1")#16
            n1=tf.layers.batch_normalization(conv1,training=is_training,name="batch_norm1")

            conv2=tf.nn.dropout(tf.layers.conv2d_transpose(n1,filters=512,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="dconv2"), keep, name="drop2")#32
            n2=tf.layers.batch_normalization(conv2,training=is_training,name="batch_norm2")

            conv3=tf.nn.dropout(tf.layers.conv2d_transpose(n2,filters=32,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="dconv3"), keep, name="drop3")#64
            n3=tf.layers.batch_normalization(conv3,training=is_training,name="batch_norm3")

            out=tf.layers.conv2d_transpose(n3,filters=3,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.sigmoid,name="dconv4")#128x128
        return out

class Discriminator:
    def __init__(self):
        pass

    def classify(self,x,keep,is_training,reuse=False):
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            norm_x=tf.image.per_image_standardization(x)

            a1=tf.nn.dropout(tf.layers.conv2d(norm_x,filters=4,kernel_size=(3,3),strides=(1,1),padding="SAME",activation=tf.nn.leaky_relu,name="conv1"),keep,name="drop1")#28
            n1=tf.layers.batch_normalization(a1,training=is_training,name="batch_norm1")

            a2=tf.nn.dropout(tf.layers.conv2d(n1,filters=32,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="conv2"),keep,name="drop2")#28
            n2=tf.layers.batch_normalization(a2,training=is_training,name="batch_norm2")

            a4=tf.nn.dropout(tf.layers.conv2d(n2,filters=512,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="conv4"),keep,name="drop4")#14
            n4=tf.layers.batch_normalization(a4,training=is_training,name="batch_norm4")

            a5=tf.nn.dropout(tf.layers.conv2d(n4,filters=1024,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="conv5"),keep,name="drop5")#14
            n5=tf.layers.batch_normalization(a5,training=is_training,name="batch_norm5")

            y=tf.layers.dense(tf.reshape(n5,[-1,1024*4*4]),units=1,name="out")
            y_prob=tf.nn.sigmoid(y)

        return y