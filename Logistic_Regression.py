
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np 
import pandas as pd
from math import floor, ceil
import tensorflow as tf
tf.reset_default_graph()


# In[2]:


#Accuracy on test set; 
def return_pred(pred_logits): 
    sig = 1/(1+np.exp(-pred_logits))
    res = np.zeros(sig.shape)
    res[sig > 0.5] = 1
    return res


# In[3]:


#For more information on the dataset: 
#https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names
data = np.genfromtxt("breast-cancer-wisconsin.data",delimiter=',',dtype=int,missing_values=-1)
data = data.astype(float)
N = data.shape[0]


# In[4]:


#Convert the categorization to 2-> 0 (benign),4-> 1 (malignant)
x = np.zeros((N,1),dtype=float)
x[data[:,10] > 2] = 1
data = np.hstack((data,x))
print(data.shape)


# In[5]:


#Define parameters
batch_size = 50
learning_rate = 0.01
n_epochs = 50
frac = 0.30 #fraction of original dataset used to train, the rest for test


# In[6]:


#Split into train and test datasets
#80 percent of the shape will be used for training, the test for testing
np.random.shuffle(data)
train_data = data[:floor(frac*N)]
test_data = data[ceil(frac*N)-1:]
# Delete the training ID and original label (index 10) 
train_data = train_data[:,1:]
train_data = np.delete(train_data,9,axis=1)
test_data  = test_data[:,1:]
test_data  = np.delete(test_data,9,axis=1)
#Split into predictors and labels; there are 9 predictors and 1 label
train_x,train_y = train_data[:,:9],train_data[:,9].reshape(-1,1)
test_x,test_y   = test_data[:,:9],test_data[:,9].reshape(-1,1)
#Save the number of observations used to train or test
N_train = train_data.shape[0]
N_test = test_data.shape[0]
# #Convert to tensor for appropriate type conversion later
# train_x,train_y = tf.convert_to_tensor(train_x,dtype=tf.float64),tf.convert_to_tensor(train_y,dtype=tf.float64)
# test_x,test_y = tf.convert_to_tensor(test_x),tf.convert_to_tensor(test_y)


# In[7]:


#Convert data into tensor
train = tf.data.Dataset.from_tensor_slices((train_x,train_y))
train = train.shuffle(N_train)
train = train.batch(batch_size)
test  = tf.data.Dataset.from_tensor_slices((test_x,test_y))
test  = test.shuffle(N_test)
# test  = test.batch(batch_size)


# In[8]:


#Create iterator 
iterator = tf.data.Iterator.from_structure(train.output_types,train.output_shapes)
feat,label = iterator.get_next()
train_init = iterator.make_initializer(train)
# test_init = iterator.make_initializer(test)


# In[9]:


#Step 1: Create weights and biases
w = tf.get_variable(name="weights",shape=(9,1),dtype=tf.float64,
                    initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001))
b = tf.get_variable(name="biases",shape=(1,1),dtype=tf.float64,initializer=tf.zeros_initializer())             


# In[10]:


#Step 2: Build the model
logits = tf.matmul(feat,w) + b


# In[11]:


#Step 3: Define loss function
entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=label,name="entropy")
loss = tf.reduce_mean(entropy, name="loss")


# In[12]:


#Step 4: Define training model 
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[13]:


#Step 5: Train the model
# preds = tf.zeros_like(logits)
# update = tf.where(tf.greater(logits,0.5))
# preds = preds.assign(update,1)
preds = tf.round(tf.sigmoid(logits))
correct = tf.equal(tf.argmax(preds,1),tf.argmax(label,1))
accuracy = tf.reduce_sum(tf.cast(correct,dtype=tf.float32))


# In[14]:


#output the graph
writer = tf.summary.FileWriter('./graphs/logreg',tf.get_default_graph())


# In[15]:


#RUN THE MODEL 
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())#MUST INITILIZE ALL VARIABLES
     # train the model n_epochs times
    for i in range(n_epochs): 	
#         print("Iteration:",i)
        sess.run(train_init)	# drawing samples from train_data
        sess.run([optimizer,loss])
    
    #test the model after getting the finalized set of weights
    final_w = sess.run(w)
    final_b = sess.run(b)
#     print(result.shape)
#     print(test_y.T)
writer.close()


# In[16]:


test_pred = return_pred(np.matmul(test_x,final_w) + final_b)
test_acc = np.sum(test_pred == test_y)/N_test
print("Accuracy on test set:{}%".format(round(test_acc*100,2)))

