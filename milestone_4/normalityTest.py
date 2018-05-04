from pandas import read_csv
from scipy.stats import normaltest
from matplotlib import pyplot
result1 = read_csv('first.csv', header=None)
value, p = normaltest(result1.values[:,0])
print(value, p)
if p >= 0.05:
	print('It is likely that result1 is normal')
else:
	print('It is unlikely that result1 is normal')
