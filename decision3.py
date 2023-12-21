#prepare
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#data path
file_path = "C:\\Workspaces\\Exam\\"
file_name = "car_data.csv" 

###preprocess's data
df = pd.read_csv(file_path + file_name)
df = df.drop('User ID', axis=1)
df = df.dropna()


#encoding
encoders = []
for i in range(0, len(df.columns) - 1):
    enc = LabelEncoder()
    df.iloc[:,i] = enc.fit_transform(df.iloc[:,i])
    encoders.append(enc)


###Model decistion tree
x = df.iloc[:, 0:3]
y = df['Purchased']
model = DecisionTreeClassifier( criterion='entropy' )
model.fit(x,y)

# predict model
x_pred = ['Male','24','58000']
for i in range(0, len(df.columns) - 1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
x_pred_adj = np.array(x_pred).reshape(-1, 3)


y_pred = model.predict(x_pred_adj)
print("Predication: ", y_pred[0])
score = model.score(x, y)
print("Accuracy: ", "{:.2f}".format(score))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model.fit(x_train, y_train)

# Test the model
y_pred = model.predict(x_test)
print("Predication: ", y_pred[0])
score = model.score(x_test, y_test)
print("Accuracy: ", "{:.2f}".format(score))

### convert to tree graph diagram

# convert to list
feature = x.columns.tolist()
Data_class = y.tolist()

'''
##plot data into graph
plt.figure(figsize=(25,20))
_ = plot_tree(model, 
              feature_names=feature,
              class_names=Data_class,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=12)

plt.show()
'''

## Feature importance
feature_importances = model.feature_importances_
feature_names = df.columns.tolist()[:3]

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x= feature_importances, y=feature_names)
print(feature_importances)


