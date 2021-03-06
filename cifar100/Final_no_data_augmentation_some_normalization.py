
# coding: utf-8

# In[1]:


# imports

import numpy as np
from scipy import interp
import keras
import itertools
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.models import Model

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.multiclass import OneVsRestClassifier

from keras.models import load_model
import matplotlib.pyplot as plt

# My imports
from my_functions import get_train_data, get_test_data, create_train_validation_partition, classes_to_superclasses

# Image parameters
IMG_ROWS, IMG_COLS =32, 32
CHANNELS=3
INPUT_SHAPE=(IMG_ROWS, IMG_COLS, CHANNELS)


# In[2]:


# Load train set
train_data, train_filenames, train_fine_labels, train_coarse_labels, train_classes, train_superclasses = get_train_data()

# Load test set
test_data, test_filenames, test_fine_labels, test_coarse_labels, test_classes, test_superclasses = get_test_data()


# In[3]:


# Normalize data
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255


# In[4]:


# Turn classes into one-hot-vectors
train_classes = np_utils.to_categorical(train_classes, 10)
test_classes = np_utils.to_categorical(test_classes, 10)

# Create train - validation partitions from the inital train set
train_partition_data, validation_partition_data, train_partition_classes, validation_partition_classes = create_train_validation_partition(train_data, train_classes, 0.1)

# Call data generators
train_datagen  = ImageDataGenerator()

valid_datagen = ImageDataGenerator()    

train_datagen.fit(train_partition_data)
valid_datagen.fit(validation_partition_data)


# In[5]:


# Create model

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.summary()


# In[6]:


opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy', 'mse'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=4, min_delta=0.001, cooldown=5, min_lr=0.0001)
mcp_save = ModelCheckpoint('.mdl_wts_no_augmentation.hdf5', save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')

history = model.fit_generator(
            train_datagen.flow(train_partition_data, train_partition_classes, batch_size=128),
            steps_per_epoch=200,
            epochs=26,
            validation_data=valid_datagen.flow(validation_partition_data, validation_partition_classes, batch_size=12),
            validation_steps=50,
            callbacks=[early_stopping, reduce_lr, mcp_save])

scores = model.evaluate(test_data, test_classes, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Test MSE:', scores[2])


# In[9]:


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


# In[10]:


print_classification_report(test_classes, test_superclasses)


# In[11]:


plot_history(history)


# In[12]:


y_test=test_classes
y_test_classes = np.argmax(test_classes, axis=1) # Convert one-hot to index

y_pred_classes = model.predict_classes(test_data)

y_pred_probabilities = model.predict(test_data)


# In[13]:


from sklearn.preprocessing import label_binarize


# In[14]:


y = label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9])
n_classes = y.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_probabilities.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

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


# In[15]:


for i in range(n_classes):
    plot_specific_class_ROC(i)


# In[16]:


from itertools import cycle


# In[17]:


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


# In[18]:


lw=2
plot_multiclass_ROC(n_classes)

