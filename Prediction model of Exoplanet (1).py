#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


kepler_data = pd.read_csv(r"C:\Users\hp\Downloads\kepler_data.csv")


# In[3]:


kepler_data


# In[4]:


# Drop irrelevant column
kepler_data = kepler_data.drop(columns=['kepler_name','koi_teq_err2','koi_teq_err1'])


# In[5]:


kepler_data=kepler_data.dropna()


# In[6]:


kepler_data


# In[7]:


# Convert categorical column to numeric
kepler_data['koi_disposition'] = kepler_data['koi_disposition'].astype('category').cat.codes


# In[8]:


kepler_data['koi_disposition']


# In[9]:


missing_values = kepler_data.isnull().sum()
missing_values


# In[10]:


# Select relevant columns for training
X = kepler_data.drop(columns=['koi_disposition', 'kepid', 'kepoi_name'])
y = kepler_data['koi_disposition']


# In[11]:


X


# In[12]:


X = X.apply(pd.to_numeric, errors='coerce')


# In[13]:


X.fillna(0, inplace=True)


# In[14]:


X.info()


# In[15]:


y 


# In[16]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


X_train


# In[18]:


# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}


# In[19]:


rf = RandomForestClassifier()


# In[20]:


# Train the classifier
rf.fit(X_train, y_train)


# In[21]:



# Instantiate the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
print("Starting GridSearchCV...")


# In[22]:


# Perform the grid search
grid_search.fit(X_train, y_train)


# In[26]:


# Get the best parameters
best_params = grid_search.best_params_


# In[27]:


# Train the model using the best parameters
best_rf = RandomForestClassifier(**best_params)
best_rf.fit(X_train, y_train)


# In[30]:


# Predict on the test set
y_pred = best_rf.predict(X_test)


# In[31]:


y_pred


# In[32]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[33]:


# Print classification report
print(classification_report(y_test, y_pred))


# In[47]:


label_Encoder = LabelEncoder()


# In[50]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




