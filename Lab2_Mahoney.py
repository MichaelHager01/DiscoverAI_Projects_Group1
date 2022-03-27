import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import required sklearn libraries for Decision Tree Classifier
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Read in the data using panda's read_csv method
my_data = pd.read_csv("SupervisedLearning/DrugSelection/drug200.csv")

#TODO: Write code to inspect the first five rows of the dataframe
my_data

#TODO: Write code to inspect the shape of the data frame
my_data.shape

#TODO: Write code to display information about the data frame
my_data.values

#TODO: Write code to display statistics about the data frame
my_data.describe()

#TODO: Write code to drop rows having missing values
#need help here
my_data.dropna()

#TODO: Write code to declare X
# Hint: remove the column containing the target of this prediction problem
# Note: To run the next section, X is expected to be an array. 
# You can get an array from a data frame with: X = X.values
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
#TODO: Write code to inspect the first five rows of X
# Note: If X is an array, instead of using the head() function,
# you will need to use array functionality to output the first five values.
X[0:5]

# Define a label encoder for the sex feature to be 0 or 1
# X is expected to be an array here. If it's a dataframe, get the array version by running:
# X = X.values

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 

#TODO: Write code to encode the BP feature in X[:,2]
# from 'Low', 'NORMAL', 'HIGH', to 0, 1, 2
le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2] = le_BP.transform(X[:,2])

# Define a label encoder for the Chol feature to be 0 or 1
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

#TODO: Write code to declare y
# Hint: this is the column containing the target of this prediction problem
Y = my_data["Drug"]
#TODO: Write code to inspect the first five rows of y
Y[0:5]





