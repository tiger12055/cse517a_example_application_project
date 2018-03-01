import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn import datasets, linear_model
import sklearn.cross_validation


#load boston dataset
boston = load_boston()


# print data set details
#~ print boston.DESCR

# convert it into pandas
bos = pd.DataFrame(boston.data)

# convert the index to the column names.
bos.columns = boston.feature_names

#~ print bos.head()


# add price to our dataset
bos['price'] = boston.target
#~ print bos.head()


# summary statistics of the dataset 
print(bos.describe())

# training set without price
x = bos.drop('price', axis = 1)

#price
y = boston.target #target
#~ print y


# split the dataset into train and test data with test data at 35%
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(x, y, test_size = 0.35, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# Run Linear Regression
linmo = linear_model.LinearRegression()
linmo.fit(X_train, Y_train)
lmodel = linmo.fit(X_train, Y_train)
Y_pred = linmo.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()

mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
mseprint = 'Mean Squared Error is ' + repr(mse)

print(mseprint)



""" 
cvscores = cross_val_score(lmodel, bos, y, cv=10)
print("cross validated scores")
print(cvscores)

"""


regr = linear_model.LinearRegression()

regr.fit(x,boston.target)

#~ print('Coefficients: \n', regr.coef_)

# check the cofficients to each attribute
output1 = pd.DataFrame(list(zip(x.columns,regr.coef_)))
print(output1)

# we can see RM has the highest correlation with prices so that we can draw graph on this

# draw image
plt.scatter(bos.RM, bos.price)
plt.xlabel("RM")
plt.ylabel("price")
plt.title("Relation between rm and price")
plt.show()



#10 -cross validation
scores = cross_val_score(regr, x, boston.target, cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


