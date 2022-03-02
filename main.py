import torch

# Import MNIST and FashionMNIST from torchvision as datasets.
import data.mnist as mnist
import data.fashion_mnist as fashion_mnist

# Import CNNs with two different architectures.
import net.cnn.one as cnn_one
import net.cnn.two as cnn_two

# Import utilities for displaying images.
import image.display as display

# Import training utilities.
import net.train as train_utils

#Import methods for noise images and analysis
import noise.utils as noise_utils

#Import methods for noise images and analysis
import noise.classification_image_classifier as classification_image_classifier

LOAD_SAVED_MODELS = True
SAVED_MODELS_LOCATION = "./net/saved_nets"

#For Reproducability
torch.manual_seed(0)

# Get trainloaders for datasets.
mnist_train_loader = mnist.get_train_loader()
fashion_mnist_train_loader = fashion_mnist.get_train_loader()

# Get testloaders for datasets.
mnist_test_loader = mnist.get_test_loader()
fashion_mnist_test_loader = fashion_mnist.get_test_loader()

# Dislay one batch from MNIST Trainset
mnist_train_images, mnist_train_labels = iter(mnist_train_loader).next()
display.display_images_grid(mnist_train_images, mnist_train_labels, mnist.get_classes(), mnist.get_batch_size())

# Dislay one batch from FashionMNIST Trainset
fashion_mnist_train_images, fashion_mnist_train_labels = iter(fashion_mnist_train_loader).next()
display.display_images_grid(fashion_mnist_train_images, fashion_mnist_train_labels, fashion_mnist.get_classes(), fashion_mnist.get_batch_size())

# Create CNNs for MNIST
mnist_cnn_one = cnn_one.create_net(net_name="mnist_cnn_one")
mnist_cnn_two = cnn_two.create_net(net_name="mnist_cnn_two")

# Create CNNs for FashionMNIST
fashion_mnist_cnn_one = cnn_one.create_net(net_name="fashion_mnist_cnn_one")
fashion_mnist_cnn_two = cnn_two.create_net(net_name="fashion_mnist_cnn_two")


if LOAD_SAVED_MODELS:
    # Load saved model state
    mnist_cnn_one.load_state_dict(torch.load(SAVED_MODELS_LOCATION + "/mnist_cnn_one"))
    mnist_cnn_two.load_state_dict(torch.load(SAVED_MODELS_LOCATION + "/mnist_cnn_two"))
    fashion_mnist_cnn_one.load_state_dict(torch.load(SAVED_MODELS_LOCATION + "/fashion_mnist_cnn_one"))
    fashion_mnist_cnn_two.load_state_dict(torch.load(SAVED_MODELS_LOCATION + "/fashion_mnist_cnn_two"))
    
else:
    # Train MNIST nets
    train_utils.train(mnist_train_loader, mnist_cnn_one)
    train_utils.train(mnist_train_loader, mnist_cnn_two)
    
    # Train FashionMNIST nets
    train_utils.train(fashion_mnist_train_loader, fashion_mnist_cnn_one)
    train_utils.train(fashion_mnist_train_loader, fashion_mnist_cnn_two)

    torch.save(mnist_cnn_one.state_dict(), SAVED_MODELS_LOCATION + "/mnist_cnn_one")
    torch.save(mnist_cnn_two.state_dict(), SAVED_MODELS_LOCATION + "/mnist_cnn_two")
    torch.save(fashion_mnist_cnn_one.state_dict(), SAVED_MODELS_LOCATION + "/fashion_mnist_cnn_one")
    torch.save(fashion_mnist_cnn_two.state_dict(), SAVED_MODELS_LOCATION + "/fashion_mnist_cnn_two")

# Dislay one batch from MNIST Inference
mnist_test_images, mnist_test_labels = iter(mnist_test_loader).next()
print("Mnist Cnn 1")
display.display_images_grid(mnist_test_images, torch.max(mnist_cnn_one(mnist_test_images)[0], 1)[1], mnist.get_classes(), mnist.get_batch_size())
print("Mnist Cnn 2")
display.display_images_grid(mnist_test_images, torch.max(mnist_cnn_two(mnist_test_images)[0], 1)[1], mnist.get_classes(), mnist.get_batch_size())

# Dislay one batch from FashionMNIST Inference
fashion_mnist_test_images, fashion_mnist_test_labels = iter(fashion_mnist_test_loader).next()
print("Fashion Mnist Cnn 1")
display.display_images_grid(fashion_mnist_test_images, torch.max(fashion_mnist_cnn_one(fashion_mnist_test_images)[0], 1)[1], fashion_mnist.get_classes(), fashion_mnist.get_batch_size())
print("Fashion Mnist Cnn 2")
display.display_images_grid(fashion_mnist_test_images, torch.max(fashion_mnist_cnn_two(fashion_mnist_test_images)[0], 1)[1], fashion_mnist.get_classes(), fashion_mnist.get_batch_size())

# Test MNIST nets
print("Mnist Cnn 1")
train_utils.get_network_accuracy(mnist_test_loader, mnist_cnn_one)
print("Mnist Cnn 2")
train_utils.get_network_accuracy(mnist_test_loader, mnist_cnn_two)

# Test FashionMNIST nets
print("Fashion Mnist Cnn 1")
train_utils.get_network_accuracy(fashion_mnist_test_loader, fashion_mnist_cnn_one)
print("Fashion Mnist Cnn 2")
train_utils.get_network_accuracy(fashion_mnist_test_loader, fashion_mnist_cnn_two)

# Test MNIST nets at class level
print("Mnist Cnn 1")
train_utils.get_predicted_classes_accuracy(mnist_test_loader, mnist_cnn_one, mnist.get_classes())
print("Mnist Cnn 2")
train_utils.get_predicted_classes_accuracy(mnist_test_loader, mnist_cnn_two, mnist.get_classes())

# Test FashionMNIST nets at class level
print("Fashion Mnist Cnn 1")
train_utils.get_predicted_classes_accuracy(fashion_mnist_test_loader, fashion_mnist_cnn_one, fashion_mnist.get_classes())
print("Fashion Mnist Cnn 2")
train_utils.get_predicted_classes_accuracy(fashion_mnist_test_loader, fashion_mnist_cnn_two, fashion_mnist.get_classes())

#Get CNN 1 image classifications for MNIST
mnist_cnn_one_classification_images = noise_utils.generate_classification_images(mnist_cnn_one, mnist.get_batch_size(), mnist.get_classes())

#Get CNN 2 image classifications for MNIST
mnist_cnn_two_classification_images = noise_utils.generate_classification_images(mnist_cnn_two, mnist.get_batch_size(), mnist.get_classes())

#Get CNN 1 image classifications for FashionMNIST
fashion_mnist_cnn_one_classification_images = noise_utils.generate_classification_images(fashion_mnist_cnn_one, fashion_mnist.get_batch_size(), fashion_mnist.get_classes())

#Get CNN 2 image classifications for FashionMNIST
fashion_mnist_cnn_two_classification_images = noise_utils.generate_classification_images(fashion_mnist_cnn_two, fashion_mnist.get_batch_size(), fashion_mnist.get_classes())

# Test CNN 1 classification images on MNIST
mnist_average_noise_map_classifier_one = classification_image_classifier.Classifier(mnist_cnn_one_classification_images)
train_utils.get_network_accuracy(mnist_test_loader, mnist_average_noise_map_classifier_one)
train_utils.get_predicted_classes_accuracy(mnist_test_loader, mnist_average_noise_map_classifier_one, mnist.get_classes())
display.display_confusion_matrix(mnist_test_loader, mnist_average_noise_map_classifier_one, mnist.get_classes())
 
# Test CNN 2 classification images on MNIST
mnist_average_noise_map_classifier_two = classification_image_classifier.Classifier(mnist_cnn_two_classification_images)
train_utils.get_network_accuracy(mnist_test_loader, mnist_average_noise_map_classifier_two)
train_utils.get_predicted_classes_accuracy(mnist_test_loader, mnist_average_noise_map_classifier_two, mnist.get_classes())
display.display_confusion_matrix(mnist_test_loader, mnist_average_noise_map_classifier_two, mnist.get_classes())

# Test CNN 1 classification images on FashionMNIST
fashion_mnist_average_noise_map_classifier_one = classification_image_classifier.Classifier(fashion_mnist_cnn_one_classification_images)
train_utils.get_network_accuracy(fashion_mnist_test_loader, fashion_mnist_average_noise_map_classifier_one)
train_utils.get_predicted_classes_accuracy(fashion_mnist_test_loader, fashion_mnist_average_noise_map_classifier_one, fashion_mnist.get_classes())
display.display_confusion_matrix(fashion_mnist_test_loader, fashion_mnist_average_noise_map_classifier_one, fashion_mnist.get_classes())

# Test CNN 2 classification images on FashioinMNIST
fashion_mnist_average_noise_map_classifier_two = classification_image_classifier.Classifier(fashion_mnist_cnn_two_classification_images)
train_utils.get_network_accuracy(fashion_mnist_test_loader, fashion_mnist_average_noise_map_classifier_two)
train_utils.get_predicted_classes_accuracy(fashion_mnist_test_loader, fashion_mnist_average_noise_map_classifier_two, fashion_mnist.get_classes())
display.display_confusion_matrix(fashion_mnist_test_loader, fashion_mnist_average_noise_map_classifier_two, fashion_mnist.get_classes())

# STA CNN 1 MNIST First Layer Noise
noise_utils.generate_sta_images_from_noise_first_conv_layer(mnist_cnn_one)

# STA CNN 1 MNIST Last Layer Noise
noise_utils.generate_sta_images_from_noise_last_conv_layer(mnist_cnn_one)

# STA CNN 1 MNIST First Layer Data
noise_utils.generate_sta_images_from_data_first_conv_layer(mnist_cnn_one, mnist_train_loader)

# STA CNN 1 MNIST Last Layer Data
noise_utils.generate_sta_images_from_data_last_conv_layer(mnist_cnn_one, mnist_train_loader)


# STA CNN 2 MNIST First Layer Noise
noise_utils.generate_sta_images_from_noise_first_conv_layer(mnist_cnn_two)

# STA CNN 2 MNIST Last Layer Noise
noise_utils.generate_sta_images_from_noise_last_conv_layer(mnist_cnn_two)

# STA CNN 2 MNIST First Layer Data
noise_utils.generate_sta_images_from_data_first_conv_layer(mnist_cnn_two, mnist_train_loader)

# STA CNN 2 MNIST Last Layer Data
noise_utils.generate_sta_images_from_data_last_conv_layer(mnist_cnn_two, mnist_train_loader)


# STA CNN 1 FashionMNIST First Layer Noise
noise_utils.generate_sta_images_from_noise_first_conv_layer(fashion_mnist_cnn_one)

# STA CNN 1 FashionMNIST Last Layer Noise
noise_utils.generate_sta_images_from_noise_last_conv_layer(fashion_mnist_cnn_one)

# STA CNN 1 FashionMNIST First Layer Data
noise_utils.generate_sta_images_from_data_first_conv_layer(fashion_mnist_cnn_one, fashion_mnist_train_loader)

# STA CNN 1 FashionMNIST Last Layer Data
noise_utils.generate_sta_images_from_data_last_conv_layer(fashion_mnist_cnn_one, fashion_mnist_train_loader)


# STA CNN 2 FashionMNIST First Layer Noise
noise_utils.generate_sta_images_from_noise_first_conv_layer(fashion_mnist_cnn_two)

# STA CNN 2 FashionMNIST Last Layer Noise
noise_utils.generate_sta_images_from_noise_last_conv_layer(fashion_mnist_cnn_two)

# STA CNN 2 FashionMNIST First Layer Data
noise_utils.generate_sta_images_from_data_first_conv_layer(fashion_mnist_cnn_two, fashion_mnist_train_loader)

# STA CNN 2 FashionMNIST Last Layer Data
noise_utils.generate_sta_images_from_data_last_conv_layer(fashion_mnist_cnn_two, fashion_mnist_train_loader)
