
# coding: utf-8

# In[22]:


import pandas as pd
train = pd.read_csv(r'C:\Users\Admin\Desktop\K-NN practical implementation\Train.csv')
train.head()


# In[23]:


train.info()


# In[ ]:


#Problem Statement
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. 
Also, certain attributes of each product and store have been defined.
The aim is to build a predictive model and find out the sales of each product at a particular store.

Using this model, BigMart will try to understand the properties of products 
and stores which play a key role in increasing sales.

 

Please note that the data may have missing values as some stores might not 
report all the data due to technical glitches. Hence, it will be required to treat them accordingly.


# In[ ]:


#The aim is to build a predictive model and find out the sales of each product at a particular store.


# In[24]:


train.columns[train.isnull().any()]


# In[33]:


miss = train.isnull().sum()/len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
miss


# In[34]:


#visualising missing values
import seaborn as sns
import matplotlib.pyplot as plt
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index

#plot the missing value count
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=miss)
plt.xticks(rotation = 90)
sns.plt.show()


# In[12]:


train['Item_Weight'].unique()


# In[35]:


train['Outlet_Size'].unique()


# In[36]:


#2. Impute missing values

train.isnull().sum()
#missing values in Item_weight and Outlet_size needs to be imputed
mean = train['Item_Weight'].mean() #imputing item_weight with mean
train['Item_Weight'].fillna(mean, inplace =True)

mode = train['Outlet_Size'].mode() #imputing outlet size with mode
train['Outlet_Size'].fillna(mode[0], inplace =True)


# In[41]:


miss = train.isnull().sum()/len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
miss


# In[40]:


train.head()


# In[42]:


#Deal with categorical variables and drop the id columns

train.drop(['Item_Identifier','Outlet_Identifier'],axis=1, inplace=True)
train = pd.get_dummies(train)




# In[43]:


train.info()


# In[48]:


#create the test train data

from sklearn.model_selection import train_test_split
trainn , test = train_test_split(train, test_size = 0.3)

x_train = trainn.drop('Item_Outlet_Sales', axis=1)
y_train = trainn['Item_Outlet_Sales']

x_test = test.drop('Item_Outlet_Sales', axis = 1)
y_test = test['Item_Outlet_Sales']


# In[50]:


#Preprocessing – Scaling the features

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)


# In[57]:


#Let us have a look at the error rate for different k values

#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
rmse_val = [] #to store rmse values for different k
for K in range(15):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[53]:


#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()


# In[64]:


# Predictions on the test dataset

#reading test and submission files
test = pd.read_csv(r'C:\Users\Admin\Desktop\K-NN practical implementation\Test.csv')
#submission = pd.read_csv(r'C:\Users\Admin\Desktop\K-NN practical implementation\SampleSubmission.csv')
#submission['Item_Identifier'] = test['Item_Identifier']
#submission['Outlet_Identifier'] = test['Outlet_Identifier']

#preprocessing test dataset
test.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)
test['Item_Weight'].fillna(mean, inplace =True)
test = pd.get_dummies(test)
test_scaled = scaler.fit_transform(test)
test = pd.DataFrame(test_scaled)

#predicting on the test set and creating submission file
predict = model.predict(test)
submission['Item_Outlet_Sales'] = predict
submission.to_csv(r'C:\Users\Admin\Desktop\K-NN practical implementation\SampleSubmission2.csv',index=False)

