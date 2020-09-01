# -----------------------------------------------------------
# util.py
# 
# Jake Sciotto
# EN685.621 Algorithms for Data Science
# Johns Hopkins University
# Summer 2020
# -----------------------------------------------------------

import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import gaussian_kde

# display every row in pandas dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# constants
LINE = "----------------------------------------------------"

# swap helper
def swap(array, index1, index2):
	temp = array[index1].copy()
	array[index1] = array[index2]
	array[index2] = temp
	return array

################################################
# insertion_sort
#
# A simple insertion sort method that runs in 
# O(n^2) time. Sorts entire rows of an array 
# and swaps appropriately.
################################################
def insertion_sort(array, col_key, f):

	for i in range (1, len(array)):
		
		key = array[i, col_key]
		j   = i - 1

		while j >= 0 and key < array[j, col_key]:
			array = swap(array, j, j + 1)
			j -= 1

		array[j + 1, col_key] = key

	f.write(str(array))
	f.write("\n")

################################################
# get_data_stats
#
# Returns stats features of the dataset.
################################################
def get_data_stats(df, f):

	mean = str(df.iloc[:, 0:4].mean()) + "\n"
	skew = str(df.iloc[:, 0:4].skew()) + "\n"

	f.write("Mean\n")
	f.write(mean)
	f.write("Skew\n")
	f.write(skew)

################################################
# remove_outliers
#
# Removes outliers in the range 
# quartile(.05, .95) inclusive
################################################
def remove_outliers(df):

	# take out class from 
	filt_df = df.iloc[:, 0:4]

	low = .05
	high = .95

	# determine the quantile values
	quant_df = filt_df.quantile([low, high])

	# apply across columns
	filt_df = filt_df.apply(lambda x: x[(x >= quant_df.loc[low, x.name]) & 
                                    (x <= quant_df.loc[high, x.name])], axis=0)
	# add in class and target
	filt_df = pd.concat([filt_df, df.iloc[:, 4:6], ], axis=1)

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

################################################
# Bhattacharyya distance method 2
# 
# Theoretical calculation using formula (under 
# normal distribution)
#
# Db(p,q) = (ln((v_p / v_q) + (v_q + v_p) + 2) / 4) +
#			(((u_p - u_q)^2 / (v_p + v_q)) / 4)
#
################################################
def bhatta_theo(X1, X2):

	# calculate standard deviation
	s1 = np.std(X1)
	s2 = np.std(X2)

	# calculate mean
	m1 = np.mean(X1)
	m2 = np.mean(X2)

	# calculate variance
	v1 = s1 ** 2
	v2 = s2 ** 2

	# use formula for theortical distribution
	bdist = np.log(((v1 / v2 + v2 / v1) + 2) / 4) / 4 + (((m1 - m2) ** 2 / (v1 + v2)) / 4)

	return bdist