
# coding: utf-8

# In[1]:

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline


# In[2]:

import glob
f_list=glob.glob('../linux_sim/log/*.csv')
dataset=[]
collection=0.05
for f in f_list:
    print(f)
    with open(f) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        #first_line = reader[0]
        prev_source_path =None
        for line in reader :
            if prev_source_path is None:
                prev_source_path=line[0]
            
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = '../linux_sim//IMG/' + filename
            measurement = float(line[3])
            
            filename = prev_source_path.split('/')[-1]
            prev_path =  '../linux_sim//IMG/' + filename
            dataset.append((current_path,prev_path,measurement))

            
            source_path = line[1]
            filename = source_path.split('/')[-1]
            current_path = '../linux_sim//IMG/' + filename
            
            filename = prev_source_path.split('/')[-1]
            prev_path =  '../linux_sim//IMG/' + filename
            #dataset.append((current_path,prev_path,measurement+collection))

            
            source_path = line[2]
            filename = source_path.split('/')[-1]
            current_path = '../linux_sim//IMG/' + filename
            
            filename = prev_source_path.split('/')[-1]
            prev_path =  '../linux_sim//IMG/' + filename            
            #dataset.append((current_path,prev_path,measurement-collection))

            prev_source_path=source_path


# In[3]:

a=cv2.imread(dataset[0][0])
b=cv2.imread(dataset[0][1])
np.concatenate((a, b), axis=2).shape


# In[4]:

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(dataset, test_size=0.2)


# In[5]:

from sklearn.utils import shuffle
import sklearn


# In[6]:

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            i=0
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = cv2.imread(name)
                prev_name = batch_sample[1]
                prev_image = cv2.imread(prev_name)
                    
                m_image=np.concatenate((image, prev_image), axis=2)
                angle = float(batch_sample[2])
                images.append(m_image)
                angles.append(angle*1.5)
                i+=1

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[7]:

from keras.models import Sequential
from keras.layers import Flatten , Dense,Convolution2D,Lambda,Cropping2D,Cropping3D,Dropout
from keras.callbacks import EarlyStopping


# In[ ]:

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)


# In[ ]:

ch, row, col = 6, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col,ch),
        output_shape=(row, col,ch)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36, 5, 5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48, 5, 5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64, 3, 3,activation="relu"))
model.add(Convolution2D(64, 3, 3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.7))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
early_stop = EarlyStopping(patience=3)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=10 ,callbacks=[early_stop])


# In[ ]:

model.save('model.h5')


# In[ ]:




# In[ ]:



