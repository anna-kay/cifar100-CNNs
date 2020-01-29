
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import PIL.Image
import urllib
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


# In[2]:


model = ResNet50(weights='imagenet')


# In[3]:


page=requests.get("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07609840")
#print(page.content)
soup=BeautifulSoup(page.content, 'html.parser')
#print(soup)
str_soup=str(soup)
#print(str_soup)
split_urls=str_soup.split('\r\n')
#print(len(split_urls))
#print(split_urls)


# In[4]:


mkdir\content\sweets


# In[5]:


def url_to_image(url):
    #download the image, convert it to a NumPy array, and then read
    #it into OpenCV format
    resp=urllib.request.urlopen(url)
    image=np.asarray(bytearray(resp.read()), dtype="uint8")
    image=cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


# In[6]:


n_of_training_images=50


# In[7]:


for progress in range(n_of_training_images):
    if (progress%20==0):
        print(progress)
    if not split_urls[progress]==None: 
        try:
            I=url_to_image(split_urls[progress])
            if (len(I.shape))==3: #check if the image has width, length and channels
                save_path='C:\content\sweets\img'+str(progress)+'.jpg'
                cv2.imwrite(save_path, I)
        except:
            None


# In[8]:


#x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
empty=(cv2.imread('C:\content\empty.jpg', 1))


# In[9]:


empty_image=image.img_to_array(empty)


# In[10]:


for i in range(50):
    img_path = 'C:\content\sweets\img'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    #x is a <class 'numpy.ndarray'>
    print('\n\n',img_path)
    #cv2.imshow(str(i), x)
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    img = cv2.imread('C:\content\sweets\img'+str(i)+'.jpg', 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    #label = decode_predictions(preds)
    #retrieve the most likely result, e.g. highest probability
    #label = label[0][0]
    #print the classification
    #print('%s (%.2f%%)' % (label[1], label[2]*100))

