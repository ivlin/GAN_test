import tensorflow as tf

class Generator:
    def __init__(self):
        pass

    def generate(self,x,drop,is_training):
        with tf.variable_scope("GAN/Generator",reuse=False):
            i_flat=tf.nn.dropout(tf.layers.dense(x,units=16*512),drop,name="drop0")
            i1=tf.reshape(i_flat,[-1,4,4,512])

            conv1=tf.nn.dropout(tf.keras.layers.Conv2DTranspose(filters=1024,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="dconv1")(i1), (1-drop), name="drop1")#16
            n1=tf.keras.layers.BatchNormalization(name="batch_norm1")(conv1,training=is_training)

            conv2=tf.nn.dropout(tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="dconv2")(n1), (1-drop), name="drop2")#128
            n2=tf.keras.layers.BatchNormalization(name="batch_norm2")(conv2,training=is_training)

            conv3=tf.nn.dropout(tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="dconv3")(n2), (1-drop), name="drop3")#128
            n3=tf.keras.layers.BatchNormalization(name="batch_norm3")(conv3,training=is_training)

            out=tf.keras.layers.Conv2DTranspose(filters=3,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=(lambda x: 255 * tf.nn.sigmoid(x)),name="dconv4")(n3)#128x128
        return out

class Discriminator:
    def __init__(self):
        pass

    def classify(self,x,drop,is_training,reuse=False):
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            norm_x=tf.image.per_image_standardization(x)

            a1=tf.nn.dropout(tf.keras.layers.Conv2D(filters=4,kernel_size=(3,3),strides=(1,1),padding="SAME",activation=tf.nn.leaky_relu,name="conv1")(norm_x),(1-drop),name="drop1")#28
            n1=tf.keras.layers.BatchNormalization(name="batch_norm1")(a1,training=is_training)

            a2=tf.nn.dropout(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="conv2")(n1),(1-drop),name="drop2")#28
            n2=tf.keras.layers.BatchNormalization(name="batch_norm2")(a2,training=is_training)

            a4=tf.nn.dropout(tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="conv4")(n2),(1-drop),name="drop4")#14
            n4=tf.keras.layers.BatchNormalization(name="batch_norm4")(a4,training=is_training)

            a5=tf.nn.dropout(tf.keras.layers.Conv2D(filters=1024,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="conv5")(n4),(1-drop),name="drop5")#14
            n5=tf.keras.layers.BatchNormalization(name="batch_norm5")(a5,training=is_training)

            y=tf.layers.dense(tf.reshape(n5,[-1,1024*4*4]),units=1,name="out")
            y_prob=tf.nn.sigmoid(y)

        return y