Milestone 3
===========
In Milestone 3 we perform dimensionality reduction using PCA  on the Boston Housing dataset.

We load the features X and standardize it.
We then create the covariance matrix between each variable using this data.
We then store the eigenvalues and eigenvectors of this matrix and then calculate the percentage of the variance that each variable explains.



Using these values, we can determine that the first, second, and third variables (CRIM -> per capita crime rate by town, ZN -> proportion of residential land zoned, INDUS -> proportion of non-retail business acres per town) captures the most variance in the dataset.

We use 2 principal components 

We classify the houses into 4 types based on the MEDV prices

LOW -> black star
MEDIUM -> blue circle
HIGH -> green cross
VERY HIGH -> red square


We then perform linear regression and find that the MSE is 63.92 which is much higher when compared to normal linear regression with all the features where the MSE was 25.744.

Author:
Yuxiang Wang 
Ashwin Kumar
