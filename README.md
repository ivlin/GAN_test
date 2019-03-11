# GAN_test

This repository is a collection of testing with generative adversarial networks.

## Generative Adversarial Networks

Generative adversarial networks are a generative model of learning  that has shown especially promising results with image data. Two competing neural networks are trained adversarially. The generator learns to create data points that match the training data; a combination of real and generated data are then labelled as such and then fed to the discriminator network. The discriminator learns to correctly identify the fakes while the generator in turn improves the quality of its images.

## Stanford Car

Dataset: Stanford car dataset (https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder)

Preliminary results:

Preliminary results shows issues with color balance. However, greyscale images show promise.

![Alt text](/stanford_car/blkwht1/output/test_it.gif?raw=true "testing output")
![Alt text](/stanford_car/color1/output/test_it.gif?raw=true "testing output")


## MNIST Semisupervised Learning

Dataset: MNIST

Here I use MNIST to try and test the generator and discriminator of a semisupervised DCGAN. Typically, the discriminator network only classifies datapoints as fake or real. This means that it serves very little purpose outside of providing feedback to improve the generator. However, with some modification, the discriminator can be modified to also complete classification tasks. Rather than labelling data points as real or fake, real data points can be assigned true calsses. The discriminator will then produce a vector of size c+1 indicating the likelihood between the c classes and being a fake.

Example: Comparison of Semisupervised Discriminator Ability vs Supervised Classifier

![Alt text](/mnist_semisupervised/sample_size_test/output_images/train_size_comp/comp_acc.gif?raw=true "accuracy")
![Alt text](/mnist_semisupervised/sample_size_test/output_images/train_size_comp/comp_loss.gif?raw=true "loss")

### Comparisons

This folder consists of attempts to tune the batch size hyperparameter. This contains:
 - mnist_supervised.py - a control MNIST classifier
 - mnist_unsupervised.py - a traditional DCGAN where the discriminator only attempts to distinguish between "real" and "fake"
 - mnist_semisupervised.py - a modified version of the DCGAN where the discriminator attempts to distinguish between "fake" or one of the ten real numbers