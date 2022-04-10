#import pandas, scipy and sklearn packages

import pandas as pd
import scipy.stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from itertools import repeat
import numpy as np

df = pd.read_csv('SupervisedLearning/supervised_test.csv')

#TODO: Write code below to inspect the first five rows of the data frame
df.head()

# Run this section to inspect X
X = df.drop(columns = ['genre'])

#TODO: Write code to inspect X
X

# Uncomment this section to inpect y
y = df['genre']

#TODO: Write code to inspect y
y

# Compute the maximum entropy value
k = y.unique().size
maxE = np.log2(k)
p_data = y.value_counts(normalize=True)           # counts occurrence of each value
entropy = scipy.stats.entropy(p_data)  # get entropy from counts

# normalize the value to be between 0 and 1.
normalizedE = entropy/maxE

#TODO: Write code to display the entropy value
entropy

avg_score = 0.0
ntimes = 30

for i in repeat(None, ntimes):

    # train model with 80% of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # prediction using entropy
    # Note: You can replace 'entropy' by 'gini' to get the classifier to use the gini index criterion.
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)

    # compute model accuracy
    avg_score += accuracy_score(y_test, predictions)

avg_score /= ntimes

print('normalized entropy value: %.3f'% normalizedE)
print('average accuracy score: %.3f' % avg_score)


# output visual (can be visualized with visual code)
tree.export_graphviz(model, out_file='SupervisedLearning/EntropySupervisedModel.dot',
                    feature_names=['age', 'gender'],
                    class_names=sorted(y.unique()),
                    label='all',
                    rounded=True,
                    filled=True)



