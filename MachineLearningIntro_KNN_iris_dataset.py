
# coding: utf-8

# In[39]:


#another Implementation of KNN as per the data science course 
import pandas as pd
#from statsmodels.api import datasets
from sklearn import datasets ## Get dataset from sklearn
## Import the dataset from sklearn.datasets
iris = datasets.load_iris()
## Create a data frame from the dictionary 
species=[]
for x in iris.target:
    species.append(iris.target_names[x])

#print(species)
#we need the data in iris data set as values of the data frame,the columns are sepal_length,'Sepal_Width', 'Petal_Length', 'Petal_Width'
iris = pd.DataFrame(iris['data'], columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
#print(iris)

#grouping the values according to species 
iris['Species'] = species
iris['count'] = 1
iris[['Species', 'count']].groupby('Species').sum()

from sklearn.preprocessing import scale
import pandas as pd
num_cols = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
iris_scaled = scale(iris[num_cols])
iris_scaled = pd.DataFrame(iris_scaled, columns = num_cols)
print(iris_scaled.describe().round(3))

levels = {'setosa':0, 'versicolor':1, 'virginica':2}
iris_scaled['Species'] = [levels[x] for x in iris['Species']]
iris_scaled.head()

## Split the data into a training and test set by Bernoulli sampling
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(3456)
iris_split = train_test_split(np.asmatrix(iris_scaled), test_size = 75)
print(iris_split)

iris_train_features = iris_split[0][:, :4]
iris_train_labels = np.ravel(iris_split[0][:, 4])

iris_test_features = iris_split[1][:, :4]
iris_test_labels = np.ravel(iris_split[1][:, 4])
#print(iris_train_features.shape)
#print(iris_train_labels.shape)
#print(iris_test_features.shape)
#print(iris_test_labels.shape)

from sklearn.neighbors import KNeighborsClassifier
KNN_mod = KNeighborsClassifier(n_neighbors = 3)
KNN_mod.fit(iris_train_features, iris_train_labels)

iris_test = pd.DataFrame(iris_test_features, columns = num_cols)
iris_test['predicted'] = KNN_mod.predict(iris_test_features)
iris_test['correct'] = [1 if x == z else 0 for x, z in zip(iris_test['predicted'], iris_test_labels)]
accuracy = 100.0 * float(sum(iris_test['correct'])) / float(iris_test.shape[0])
print(accuracy)

"""
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

def plot_iris(iris, col1, col2):
    sns.lmplot(x = col1, y = col2, data = iris, hue = "Species", fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Iris species shown by color')
    plt.show()
plot_iris(iris, 'Petal_Width', 'Sepal_Length')
plot_iris(iris, 'Sepal_Width', 'Sepal_Length')

## Define and train the KNN model
levels = {0:'setosa', 1:'versicolor', 2:'virginica'}
iris_test['Species'] = [levels[x] for x in iris_test['predicted']]
markers = {1:'^', 0:'o'}
colors = {'setosa':'blue', 'versicolor':'green', 'virginica':'red'}
def plot_shapes(df, col1,col2,  markers, colors):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = plt.figure(figsize=(6, 6)).gca() # define plot axis
    for m in markers: # iterate over marker dictionary keys
        for c in colors: # iterate over color dictionary keys
            df_temp = df[(df['correct'] == m)  & (df['Species'] == c)]
            sns.regplot(x = col1, y = col2, 
                        data = df_temp,  
                        fit_reg = False, 
                        scatter_kws={'color': colors[c]},
                        marker = markers[m],
                        ax = ax)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Iris species by color')
    return 'Done'
plot_shapes(iris_test, 'Petal_Width', 'Sepal_Length', markers, colors)
plot_shapes(iris_test, 'Sepal_Width', 'Sepal_Length', markers, colors)
"""


# In[113]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  


# In[2]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)  


# In[3]:


dataset


# In[4]:


X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 4].values


# In[5]:


X


# In[6]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


# In[7]:


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 


# In[8]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  


# In[9]:


y_pred = classifier.predict(X_test) 


# In[10]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


# In[11]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[15]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')
plt.show(plt.ylabel('Mean Error'))


# In[17]:


y


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


# In[20]:


X_train.sum()


# In[21]:


X_test.sum()


# In[27]:


y_test


# In[28]:


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 


# In[29]:


X_train


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


# In[31]:


X_train


# In[32]:


X_test


# In[33]:


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


# In[34]:


X_train


# In[76]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(X_train, y_train)  


# In[77]:


y_pred = classifier.predict(X_test)  


# In[78]:


y_pred


# In[79]:


y_pred = classifier.predict(X_test) 
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


# In[69]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 10):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[71]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')
plt.show(plt.ylabel('Mean Error'))


# In[82]:


#plt.figure(figsize=(10,5))
plt.plot(range(1,10),error,color='red')


# In[83]:



plt.show(plt.plot(range(1,10),error,color='red'))


# In[114]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("Fruit_data_with_colors.csv")
data


# In[115]:


data.info()


# In[116]:


data.isnull().sum()


# In[127]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)


# In[128]:


dataset


# In[129]:


x=dataset.iloc[::-1].values


# In[130]:


x


# In[131]:


y=dataset.iloc[:,4].values


# In[132]:


y


# In[133]:


from sklearn.model_selection import train_test_split


# In[135]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


# In[136]:


x_train


# In[137]:


x_test


# In[138]:


y_train


# In[139]:


y_test


# In[141]:


from sklearn.neighbors import KNeighborsClassifier


# In[142]:


classifier=KNeighborsClassifier(n_neighbors=5)


# In[144]:


X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
classifier.fit(X_train,y_train)


# In[145]:


y_pred=classifier.predict(x_test)
from sklearn.metrics import classification_report , confusion_matrix
print(classification_report(y_test,y_pred))


# In[146]:


print(confusion_matrix(y_test,y_pred))


# In[148]:


error=[]
for i in (1,30):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i=Knn.Predict(x_test)
    error.append(np.mean(pred_i != y_test))


# In[149]:


for i in range(1, 10):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.Predict(x_test)
    error.append(np.mean(pred_i != y_test))


# In[150]:


for i in range(1, 10):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[152]:


plt.show(plt.plot(range(1,10)),error,color='red')


# In[153]:


plt.plot(range(1,10),error,color='red')


# In[154]:


plt.show(plt.plot(range(1,10),error,color='red'))

