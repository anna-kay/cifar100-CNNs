import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools

from scipy import interp
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier



img_rows, img_cols =32, 32
channels=3
input_shape=(img_rows, img_cols, channels)

#-----------------------------------------------------------------------------------------------------------------#
#----------------------------------- Preparing Data for Training -------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

def unpickle(file):
    
    with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
    return dict
    

    
def reshape_images_data_shape(raw_images_data):
    
    #Reshape to 4-dimensions - [image_number, channel, height, width]
    images_data=raw_images_data.reshape([-1, channels, img_rows, img_cols])
    #4D array - [image_number, height, width, channel]
    images_data=images_data.transpose([0,2,3,1])
    return images_data
    
    
    
def load_data(file):
    
    data = unpickle(file)
    
    # Get raw images data
    images_data =data[b'data']
    # Reshape images_data
    reshaped_images_data =reshape_images_data_shape(images_data)
    
    # Get lists of filenames and labels
    filenames = [t.decode('utf8') for t in data[b'filenames']]
    fine_labels = data[b'fine_labels']
    coarse_labels= data[b'coarse_labels']
    # Turn lists into numpy arrays
    filenames=np.array(filenames)
    fine_labels=np.array(fine_labels)
    coarse_labels=np.array(coarse_labels)
        
    return reshaped_images_data, filenames, fine_labels, coarse_labels
    
    
    
def select_superclasses_indices(a,b, coarse_labels):
    
    indices=list()

    for index in range(len(coarse_labels)):
        if (coarse_labels[index]==a or coarse_labels[index]==b):
            index_to_be_added=index
            indices.append(index_to_be_added)
    
    return indices



def select_items(indices, data, filenames, fine_labels, coarse_labels):
    
    selected_filenames=filenames[indices]
    selected_fine_labels=fine_labels[indices]
    selected_coarse_labels=coarse_labels[indices]
    
    selected_data=data[indices, :]

    return selected_data, selected_filenames, selected_fine_labels, selected_coarse_labels



def change_class_encoding_numbers(class_encoding):
    
    # [cloud, forest, mountain, plain, sea, camel, cattle, chimpanzee, elephant, kangaroo]
    # [23, 33, 49, 60, 71, 15, 19, 21, 31, 38]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]    
  
    initial_encoding=[23, 33, 49, 60, 71, 15, 19, 21, 31, 38]
    final_encoding=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 
    d = dict(zip(initial_encoding, final_encoding))
    
    class_encoding_list=[d.get(e, e) for e in class_encoding]
    class_encoding=np.array(class_encoding_list)
    
    return class_encoding



def change_superclass_encoding_numbers(superclass_encoding):
    
    # [large natural outdoor scenes, large omnivores and herbivores]
    # [10, 11]
    # [0, 1]    
  
    initial_encoding=[10, 11]
    final_encoding=[0, 1]
 
    d = dict(zip(initial_encoding, final_encoding))
    
    superclass_encoding_list=[d.get(e, e) for e in superclass_encoding]
    superclass_encoding=np.array(superclass_encoding_list)
    
    return superclass_encoding



def classes_to_superclasses(predicted_classes):
    
    # [cloud, forest, mountain, plain, sea, camel, cattle, chimpanzee, elephant, kangaroo]
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    superclasses = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    
    d = dict(zip(classes, superclasses))
    
    predicted_superclasses_list=[d.get(e, e) for e in predicted_classes]
    predicted_superclasses=np.array(predicted_superclasses_list)
    
    return predicted_superclasses



def get_train_data():
    
    data, filenames, fine_labels, coarse_labels = load_data('cifar-100-python/train')
    indices=select_superclasses_indices(10, 11, coarse_labels)
    train_data, train_filenames, train_fine_labels, train_coarse_labels = \
    select_items(indices, data, filenames, fine_labels, coarse_labels)
    
    train_classes=change_class_encoding_numbers(train_fine_labels)
    train_superclasses=change_superclass_encoding_numbers(train_coarse_labels)
        
    return train_data, train_filenames, train_fine_labels, train_coarse_labels, train_classes, train_superclasses



def get_test_data():
    
    data, filenames, fine_labels, coarse_labels = load_data('cifar-100-python/test')
    indices=select_superclasses_indices(10, 11, coarse_labels)
    test_data, test_filenames, test_fine_labels, test_coarse_labels = \
    select_items(indices, data, filenames, fine_labels, coarse_labels)

    test_classes=change_class_encoding_numbers(test_fine_labels)
    test_superclasses=change_superclass_encoding_numbers(test_coarse_labels)
    
    np.save('test_data.npy',test_data)
    np.save('test_classes.npy',test_classes)
    
    return test_data, test_filenames, test_fine_labels, test_coarse_labels, test_classes, test_superclasses



def create_train_validation_partition(X, y, validation_size):
    
    train_partition_data, validation_partition_data, train_partition_classes, validation_partition_classes = \
    train_test_split(X, y, test_size=validation_size, stratify=y)
    
    np.save('train_partition_data.npy',train_partition_data)
    np.save('train_partition_classes.npy',train_partition_classes)
    np.save('validation_partition_data.npy',validation_partition_data)
    np.save('validation_partition_classes.npy',validation_partition_classes)
    
    return train_partition_data, validation_partition_data, train_partition_classes, validation_partition_classes


#-----------------------------------------------------------------------------------------------------------------#
#----------------------------------- Classification Report -------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

def plot_history(history):
    mean_squared_error_list = [s for s in history.history.keys() if 'mean_squared_error' in s and 'val' not in s]
    val_mean_squared_error_list = [s for s in history.history.keys() if 'mean_squared_error' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(mean_squared_error_list) == 0:
        print('mean_squared_error is missing in history')
        return 
    
    ## As MSE always exists
    epochs = range(1,len(history.history[mean_squared_error_list[0]]) + 1)
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    ## MSE
    plt.figure(1)
    for l in mean_squared_error_list:
        plt.plot(epochs, history.history[l], 'b', label='Training MSE (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_mean_squared_error_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation MSE (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    
    return 
 
    

def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
                                       
    # This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return 


    
def print_classification_report(test_classes, test_superclasses):
    
    y_test_classes = np.argmax(test_classes, axis=1) # Convert one-hot to index
    
    y_pred_classes = model.predict_classes(test_data)
    y_pred_superclasses = classes_to_superclasses(y_pred_classes)
      
    # Report for classes
    print("______________________________CLASSES______________________________\n")
    
    Accuracy= accuracy_score(y_test_classes, y_pred_classes)
    print("Total Classes Accuracy= ",Accuracy,"\n")
    
    print(classification_report(y_test_classes, y_pred_classes))
    
    cnf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    plot_confusion_matrix(cnf_matrix, classes=[0,1,2,3,4,5,6,7,8,9])
     
    # Report for superclasses
    print("______________________________SUPERCLASSES______________________________\n")
    
    Accuracy= accuracy_score(test_superclasses, y_pred_superclasses)
    print("Total Superclasses Accuracy= ",Accuracy,"\n")
    
    print(classification_report(test_superclasses, y_pred_superclasses))
    
    cnf_matrix = confusion_matrix(test_superclasses, y_pred_superclasses)
    plot_confusion_matrix(cnf_matrix, classes=[0,1])

#-----------------------------------------------------------------------------------------------------------------#
#--------------------------------------------- ROCS --------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

def plot_specific_class_ROC(class_index):
   
    plt.figure()
    lw = 2
    plt.plot(fpr[class_index], tpr[class_index], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[class_index])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - class %i' % class_index)
    plt.legend(loc="lower right")
    plt.show()
    
    return

#-----------------------------------------------------------------------------------------------------------------#

def plot_multiclass_ROC(n_classes):
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'purple', 'red', 'green', 'dimgrey', 'yellow', 'deepskyblue', 'navy'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.show()

    return

#-----------------------------------------------------------------------------------------------------------------#

def compute_and_plot_ROC(y_test, y_score):

    y = label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9])
    n_classes = y.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    print("------------------------ROC per Class------------------------\n")
    
    for i in range(n_classes):
        plot_specific_class_ROC(i)
    
    print("------------------------Multiclass ROC------------------------\n")
    
    plot_multiclass_ROC(n_classes)
    
    return