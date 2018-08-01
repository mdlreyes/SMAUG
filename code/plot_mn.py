# plot_mn.py
# Make plots.
# 
# Created 22 June 18
# Updated 22 June 18
###################################################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import math
from astropy.io import fits
import pandas

def plot_mn_fe(filenames, outfile, title, snr=None):
	"""Plot [Mn/Fe] vs [Fe/H] for all the stars in a set of files.

	Inputs:
	filename 	-- list of input filenames
	outfile 	-- name of output file
	title 		-- title of graph

	Keywords:
	snr 		-- if not None, then plot points of different S/N ratios in different colors!
	"""

	# Get data from files
	name 	= []
	feh 	= []
	feherr 	= []
	mnh 	= []
	mnherr	= []

	for file in filenames:
		current_name 	= np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')

		data = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=[5,6,8,9])
		current_feh 	= data[:,0]
		current_feherr 	= data[:,1]
		current_mnh 	= data[:,2]
		current_mnherr 	= data[:,3]

		# Append to total data arrays
		name.append(current_name)
		feh.append(current_feh)
		feherr.append(current_feherr)
		mnh.append(current_mnh)
		mnherr.append(current_mnherr)

	# Convert back to numpy arrays
	name 	= np.hstack(name)
	feh 	= np.hstack(feh)
	feherr 	= np.hstack(feherr)
	mnh 	= np.hstack(mnh)
	mnherr  = np.hstack(mnherr)

	# Compute [Mn/Fe]
	mnfe = mnh - feh
	mnfeerr = np.sqrt(np.power(feherr,2.)+np.power(mnherr,2.))

	# Testing: Label some stuff
	outlier = np.where(feh > -1)[0]
	print(name[outlier])
	notoutlier = np.where(mnfeerr < 1)[0]
	'''
	for i in range(len(outlier)):
		idx = outlier[i]
		plt.text(feh[idx], mnfe[idx], name[idx]) 
	'''

	# Errorbar plot
	'''
	# Make plot
	plt.figure()
	plt.title(title)
	plt.xlabel('[Fe/H]')
	plt.ylabel('[Mn/Fe]')
	plt.errorbar(feh[notoutlier],mnfe[notoutlier], xerr=feherr[notoutlier], yerr=mnfeerr[notoutlier], color='k', marker='.', linestyle='None')

	# Plot points with different signal/noise in different colors
	plt.gca().set_color_cycle(['red', 'blue', 'orange', 'cyan'])
	if snr is not None:
		for i in range(len(snr)):

			# Find points with signal/noise greater than whatever value
			snrmask = np.where(np.abs(mnfe/mnfeerr) > snr[i])

			# Plot in a different color
			plt.errorbar(feh[snrmask], mnfe[snrmask], xerr=feherr[snrmask], yerr=mnfeerr[snrmask], marker='o', linestyle='None', label='S/N > '+str(snr[i]))
	plt.legend(loc='best')
	'''

	# Scatter plot
	fig, ax = plt.subplots(figsize=(10,6))
	area = 2*np.reciprocal(np.power(mnfeerr,2.))
	ax.scatter(feh, mnfe, s=area, c='b', alpha=0.5)
	ax.text(0.025, 0.05, 'N = '+str(len(name)), transform=ax.transAxes, fontsize=14)

	# Format plot
	ax.set_title(title, fontsize=18)
	ax.set_xlabel('[Fe/H]', fontsize=16)
	ax.set_ylabel('[Mn/Fe]', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)

	# Output file
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

def main():
	# Sculptor
	#plot_mn_fe(['data/scl1_final.csv','data/scl2_final.csv','data/scl6_final.csv'],'figures/mnfe_scltotal.png','Sculptor',snr=[3,5])

	# Ursa Minor
	#plot_mn_fe(['data/umi1_final.csv','data/umi2_final.csv','data/umi3_final.csv'],'figures/mnfe_umitotal.png','Ursa Minor',snr=[3,5])

	# Draco
	plot_mn_fe(['data/dra1_final.csv','data/dra2_final.csv','data/dra3_final.csv'],'figures/mnfe_dratotal.png','Draco',snr=[3,5])

if __name__ == "__main__":
	main()