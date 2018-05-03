import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn import datasets, linear_model
import sklearn.cross_validation
from sklearn.semi_supervised import label_propagation
from scipy.sparse.csgraph import connected_components
import math
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score



#load boston dataset
boston = load_boston()

X = boston.data
#~ print X
y = boston.target
#~ print y

for i in range(0, 505):
	y[i] = int(math.ceil(y[i]))
#~ print y
	
# data processing
labeled_data = X[0:100] #len = 100
labeled_target = y[0:100]
#~ print labeled_data
#~ print labeled_target

unlabeled_data = X[101: ]   #len = 405
unlabeled_target = y[101: ]
#~ print len(unlabeled_data) 
#~ print len(unlabeled_target)

indices = np.arange(len(boston.data))

unlable_set = indices[101:]

#~ print unlable_set

unlabeled_set = indices[100:]
#~ print unlabeled_set

y_train = np.copy(y)
y_train[unlabeled_set] = -1
#~ print y_train
#~ print y

lp_model = label_propagation.LabelSpreading(gamma=0.01) #generate the rest of 405 labels by using first 100 labeled data


#~ lp_model = label_propagation.LabelSpreading(kernel='knn', n_neighbors=3, alpha=0.9, tol=0.01)

lp_model.fit(X, y_train)

predicted_labels = lp_model.transduction_[unlabeled_set] #get new labeled data

#~ print predicted_labels

true_labels = y[unlabeled_set]

#~ print true_labels

new_target = list()

# combined the first 100 and the rest of 405 unlabeled data to form new dataset for linear regression
for i in range(0,100):
	new_target.append(int(labeled_target[i]))


for j in range(0,406):
	new_target.append(int(predicted_labels[j]))	
	
#######linear regression on new data #######
yy = boston.target
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, new_target, test_size = 0.35, random_state = 0)
linmo = linear_model.LinearRegression()
linmo.fit(X_train, Y_train)
lmodel = linmo.fit(X_train, Y_train)
Y_pred = linmo.predict(X_test)

#~ print Y_test
#~ print Y_pred


print("Mean squared error using semi label data: %.2f"
      % mean_squared_error(Y_test, Y_pred))

#~ mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)

#~ mse1 = mse
#~ mseprint = 'Mean Squared Error for semi is ' + repr(mse1)

#~ print(mseprint)

##################use only first 100 label data to predict the rest of unlabled data###########################

#~ print labeled_data
#~ print labeled_target
#~ print unlabeled_data
regr = linear_model.LinearRegression()
regr.fit(labeled_data, labeled_target)
diabetes_y_pred = regr.predict(unlabeled_data)


#~ print diabetes_y_pred
#~ print unlabeled_target

print("Mean squared error using first 100 label data: %.2f"
      % mean_squared_error(unlabeled_target, diabetes_y_pred))


#######GP on new data #######


#~ y1 = new_target

#~ seed = 2018
#~ np.random.seed(seed)




#~ # First kernel: RBF
#~ estimators_1 = []
#~ estimators_1.append(('gp', GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=9,normalize_y=True)))




#~ pipeline_1 = Pipeline(estimators_1)


#~ # In[10]:

#~ print("Gaussian process using RBF kernel ...")
#~ print("10-folder CV ...")

#~ # 10-folder CV
#~ kfold_1 = KFold(n_splits=10, random_state=seed)
#~ results_1 = cross_val_score(pipeline_1, X, y1, cv=kfold_1)


#~ print("MSE using RBF kernel: %.2f " % (-1*results_1.mean()))




# Second kernel: Matern
#~ estimators_2 = []
#~ estimators_2.append(('gp', GaussianProcessRegressor(kernel=Matern(), n_restarts_optimizer=9,normalize_y=True)))
#~ pipeline_2 = Pipeline(estimators_2)

#~ print("Gaussian process using Matern kernel ...")
#~ print("10-folder CV ...")

#~ # 10-folder CV
#~ kfold_2 = KFold(n_splits=10, random_state=seed)
#~ results_2 = cross_val_score(pipeline_2, X, y1, cv=kfold_2)
#~ print("MSE using Matern kernel: %.2f " % (-1*results_2.mean()))

