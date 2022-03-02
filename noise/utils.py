import torch
import image.display as display
from functools import lru_cache

@lru_cache
def get_noise_images(total_num_images=100_000, image_size=(28, 28), r1=-1, r2=1):
    return  torch.rand(total_num_images, 1, image_size[0], image_size[1]).uniform_(r1, r2)

def noise_images_to_classification_images(noise_images, total_num_noise_images, net, output_classes, image_size=(1, 28, 28)):
    sum_predicted_noise_images_for_class = torch.zeros((len(output_classes),) + image_size)
    num_predicted_noise_images_for_class = torch.zeros(len(output_classes))
    noise_image_predicted_classes = torch.max(net(noise_images)[0], dim=1).indices
    
    for i in range(total_num_noise_images):
        sum_predicted_noise_images_for_class[noise_image_predicted_classes[i]] += noise_images[i]
        num_predicted_noise_images_for_class[noise_image_predicted_classes[i]] += 1
        
    for i in range(len(output_classes)):
        if num_predicted_noise_images_for_class[i] != 0:
            sum_predicted_noise_images_for_class[i] = torch.div(sum_predicted_noise_images_for_class[i], num_predicted_noise_images_for_class[i])
        
    return sum_predicted_noise_images_for_class


def generate_classification_images(net, batch_size, classes):
    total_num_noise_images = 100_000
    noise_images = get_noise_images(total_num_images=total_num_noise_images)
    image_classifcations = noise_images_to_classification_images(noise_images, total_num_noise_images, net, classes)
    for i in range(len(classes)):
        display.display_images_grid(image_classifcations[i:i+1], None, classes, batch_size, title=[f"Net: {net.get_name()}, Class: {classes[i]}"])
    return image_classifcations

def predict_image_classification_map_class(net, classification_images, classes):
    predictions = torch.max(net(classification_images)[0], dim=1).indices
    for i in range(len(classes)):
        print(f"Classification Image {classes[i]} predicted as {classes[predictions[i]]} by net: {net.get_name()}")
    return predictions

def generate_sta_images_from_noise_first_conv_layer(net):
    noise_images = get_noise_images()
    _, first_conv_layer_output, _ = net(noise_images)
    avg_first_conv_layer_output = torch.mean(first_conv_layer_output, dim=0)
    
    with torch.no_grad(): 
        for i in range(avg_first_conv_layer_output.size()[0]):
            display.display_images_grid(avg_first_conv_layer_output[i].unsqueeze(0).unsqueeze(0), None, None, None, title=[f"conv_1 channel: {i+1}"])
            
def generate_sta_images_from_noise_last_conv_layer(net):
    noise_images = get_noise_images()
    _, _, last_conv_layer_output = net(noise_images)
    avg_last_conv_layer_output = torch.mean(last_conv_layer_output, dim=0)
    
    with torch.no_grad():            
        for i in range(avg_last_conv_layer_output.size()[0]):
            display.display_images_grid(avg_last_conv_layer_output[i].unsqueeze(0).unsqueeze(0), None, None, None, title=[f"conv_last channel: {i+1}"])
    
def generate_sta_images_from_data_first_conv_layer(net, train_loader):
    sum_first_conv_layer_output = 0
    total_num_images = 0
    for data in train_loader:
        images, labels = data
        _, first_conv_layer_output, _ = net(images)
        total_num_images += len(images)
        sum_first_conv_layer_output += first_conv_layer_output
    
    avg_first_conv_layer_output = torch.sum(sum_first_conv_layer_output, dim=0) / total_num_images
    
    with torch.no_grad(): 
        for i in range(avg_first_conv_layer_output.size()[0]):
            display.display_images_grid(avg_first_conv_layer_output[i].unsqueeze(0).unsqueeze(0), None, None, None, title=[f"conv_1 channel: {i+1}"])
            
def generate_sta_images_from_data_last_conv_layer(net, train_loader):
    sum_last_conv_layer_output = 0
    total_num_images = 0
    for data in train_loader:
        images, labels = data
        _, _, last_conv_layer_output = net(images)
        total_num_images += len(images)
        sum_last_conv_layer_output += last_conv_layer_output
    
    avg_last_conv_layer_output = torch.sum(sum_last_conv_layer_output, dim=0) / total_num_images
    
    with torch.no_grad():             
        for i in range(avg_last_conv_layer_output.size()[0]):
            display.display_images_grid(avg_last_conv_layer_output[i].unsqueeze(0).unsqueeze(0), None, None, None, title=[f"conv_last channel: {i+1}"])