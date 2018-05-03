Milestone 4
===========
We used first 100 Boston housing data and its target as our labeled data. Then we used sklearn library function LabelSpreading to learn the rest of unlabeling data through labeled data. Finally, apply linearly regression to new dataset and We got 11.01 as mean square error compared to 989.48 if we only used the first 100 labeled to train our model then predict the rest of data. Moreover, if we compare this result to the milestone1 which we got 25.74 as mean square error, the semi-suprevised still preforms better. Thus, by adding cheap and abundant unlabeled data, we are able to build a better model than using supervised learning alone.

