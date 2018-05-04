Milestone 4
-===========
+============================================
+We compare the different methods by performing 10 fold cross validation on the predictions and performing summary statistics and then using a statistical test such as the t-test.
+
+Linear Regression (A) vs Gaussian Process (B)
+
+![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)
+
+The box captures the middle 50% of the data, outliers are shown as '+' and the red line shows the median. 
+We can see the data indeed has a similar spread from both distributions and is not symmetric about the median.
+We see that A (Linear regression) performs better in handling outliers as well.
+ 
+We also see that both sets of results are Gaussian and have the same variance; this means we can use the Student t-test to see if the difference between the means of the two distributions is statistically significant or not in SciPy, we can use the ttest_ind() function.
+
+We can see that the p-value is much greater than 0.05 thus accepting the null hypothesis and the samples are likely drawn from the same distributions. This shows that both the algorithms are statistically accurate to each other though GP provides a smaller MSE.
+
+
+
+
+
+
+
+
+OPTIONAL SEMI-SUPERVISED LEARNING
+============================================
+
 We used first 100 Boston housing data and its target as our labeled data. Then we used sklearn library function LabelSpreading to learn the rest of unlabeling data through labeled data. Finally, apply linearly regression to new dataset and We got 11.01 as mean square error compared to 989.48 if we only used the first 100 labeled to train our model then predict the rest of data. Moreover, if we compare this result to the milestone1 which we got 25.74 as mean square error, the semi-suprevised still preforms better. Thus, by adding cheap and abundant unlabeled data, we are able to build a better model than using supervised learning alone.
