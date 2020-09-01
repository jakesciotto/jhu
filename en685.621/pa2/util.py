# -----------------------------------------------------------
# util.py
# 
# Jake Sciotto
# EN685.621 Algorithms for Data Science
# Johns Hopkins University
# Summer 2020
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import gaussian_kde

# debugging options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

LINE = "------------------------------------------------------\n"

################################################
# find_blanks
#
# Returns a dataset without the NaN values.
################################################
def find_blanks(dataset):

	count = []
	columns = list(dataset)

	for i in columns:
		count.append(dataset[i].isna().sum())

	return count

################################################
# remove_outliers
#
# Removes outliers in the range 
# quartile(.05, .95) inclusive
################################################
def remove_outliers(df):

	# take out class from 
	filt_df = df.iloc[:, 0:8]

	low = .05
	high = .95

	# determine the quantile values
	quant_df = filt_df.quantile([low, high])

	# apply across columns-
	filt_df = filt_df.apply(lambda x: x[(x >= quant_df.loc[low, x.name]) & 
                                    (x <= quant_df.loc[high, x.name])], axis=0)

	# add in class and target
	filt_df = pd.concat([filt_df, df.iloc[:, 8], ], axis=1)

	# drop NaN values
	filt_df.dropna(inplace=True)

	return filt_df

################################################
# density
#
# Generates a continuous density function from 
# which we will draw samples and perform a 
# calculation resembling an integration.
################################################
def density(x, cov_factor):
	density = gaussian_kde(x)

	# gaussian_kde function uses a changable function to calculate its bandwidth
	# that we can modify with 

	density.covariance_factor = lambda: cov_factor
	density._compute_covariance()
	return density

################################################
# Bhattacharyya distance method 1
# 
# Db (p,q) = -ln(BC(p, q))
# 
# We find the Bhattacharyya coefficient by
# using form => BC(p, q) = sum_i^n sqrt(p1, p2)
# and return the negative log to get the distance, where
# p1 and p2 are samples in our continuous density function
# and n is the number of steps in integration.
################################################
def bhatta_cont(X1, X2):

	# value to return
	bh_distance = 0

	# put features together for later
	data = np.concatenate((X1, X2))

	# the number of specified steps for our "integration"
	steps = 200

	# generate a continuous density function
	density_func1 = density(X1, .5)
	density_func2 = density(X2, .5)

	# returns the samples between the specified number of steps
	samples = np.linspace(min(data), max(data), steps)

	# p1 and p2 are numbers of members in samples in nth partition
	for s in samples:
		p1 = density_func1(s)
		p2 = density_func2(s)

		# add up our results to get our coefficient 
		bh_distance += sqrt(p1 * p2) * (max(data) - min(data)) / steps

	# return the distance
	return -np.log(bh_distance)