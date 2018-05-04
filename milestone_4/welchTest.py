from pandas import read_csv
from scipy.stats import ttest_ind
from matplotlib import pyplot
from scipy.stats import ks_2samp
# load results1
result1 = read_csv('first.csv', header=None)
values1 = result1.values[:,0]
# load results2
result2 = read_csv('second.csv', header=None)
values2 = result2.values[:,0]
# calculate the significance
value, pvalue = ks_2samp(values1, values2)
print(value, pvalue)
if pvalue > 0.05:
	print('Samples are likely drawn from the same distributions (accept H0)')
else:
	print('Samples are likely drawn from different distributions (reject H0)')
