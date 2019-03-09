# GAN_test

This repository is a collection of testing with generative adversarial networks.

## Generative Adversarial Networks

Generative adversarial networks are a generative model of learning  that has shown especially promising results with image data. Two competing neural networks are trained adversarially. The generator learns to create data points that match the training data; a combination of real and generated data are then labelled as such and then fed to the discriminator network. The discriminator learns to correctly identify the fakes while the generator in turn improves the quality of its images.

## MNIST Semisupervised Learning

Here I use MNIST to try and test the generator and discriminator of a semisupervised DCGAN. Typically, the discriminator network only classifies datapoints as fake or real. This means that it serves very little purpose outside of providing feedback to improve the generator. However, with some modification, the discriminator can be modified to also complete classification tasks. Rather than labelling data points as real or fake, real data points can be assigned true calsses. The discriminator will then produce a vector of size c+1 indicating the likelihood between the c classes and being a fake.

##
