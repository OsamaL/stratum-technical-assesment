import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import confusion_matrix
import torch
import seaborn as sns


def imshow(img, title=None):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()[0]
    plt.imshow(npimg, cmap="viridis", aspect='equal')
    if title:
     plt.title(title)
    plt.show()
    
def display_images_grid(images, labels, classes, batch_size, title=None):
    imshow(torchvision.utils.make_grid(images), title=title if title else [f'{classes[labels[j]]:5s}' for j in range(batch_size)])
    

def get_confusion_matrix(testloader, net, classes):
    confusion_matrix_true_labels = [0] * (len(iter(testloader).next()[1]) * len(testloader))
    confusion_matrix_prediction_labels = [0] * (len(iter(testloader).next()[1]) * len(testloader))
    index = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)[0]
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted)):
                confusion_matrix_true_labels[index] = classes[labels[i]]
                confusion_matrix_prediction_labels[index] = classes[predicted[i]]
                index += 1
    return confusion_matrix(confusion_matrix_true_labels, confusion_matrix_prediction_labels, labels=list(classes))



def display_confusion_matrix(test_loader, net, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = get_confusion_matrix(test_loader, net, classes)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(list(classes), rotation=45); ax.yaxis.set_ticklabels(list(classes), rotation=0, horizontalalignment='right');