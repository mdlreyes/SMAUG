# plot_mn.py
# Make plots.
# 
# Created 22 June 18
# Updated 11 Aug 18
###################################################################

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import math
from astropy.io import fits
import pandas
from matplotlib.ticker import NullFormatter

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

def comparison_plot(filenames, labels, outfile, title):
	"""Compare [Mn/H] vs [Mn/H] for two different files.

	Inputs:
	filenames 	-- list of input filenames (must only have 2 files!)
	labels		-- labels for input filenames
	outfile 	-- name of output file
	title 		-- title of graph
	"""

	# Check that the right number of files is specified
	if len(filenames) != 2:
		print('Must specify 2 input files!')
		return
	if len(labels)!= 2:
		print('Must specify labels for both input files!')

	# Get data
	x_name 	= np.genfromtxt(filenames[0], delimiter='\t', skip_header=1, usecols=0, dtype='str')
	x_mnh 	= np.genfromtxt(filenames[0], delimiter='\t', skip_header=1, usecols=8)
	x_mnherr 	= np.genfromtxt(filenames[0], delimiter='\t', skip_header=1, usecols=9)

	y_name 	= np.genfromtxt(filenames[1], delimiter='\t', skip_header=1, usecols=0, dtype='str')
	y_mnh 	= np.genfromtxt(filenames[1], delimiter='\t', skip_header=1, usecols=8)
	y_mnherr 	= np.genfromtxt(filenames[1], delimiter='\t', skip_header=1, usecols=9)

	# Match catalogs to make sure correct values are being plotted against one another
	x = []
	y = []
	xerr = []
	yerr = []

	for i in range(len(x_name)):
		if x_name[i] in y_name:
			x.append(x_mnh[i])
			xerr.append(x_mnherr[i])

			idx = np.where(y_name == x_name[i])
			y.append(y_mnh[idx][0])
			yerr.append(y_mnherr[idx][0])

	# Definitions for the axes
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.65
	bottom_h = left_h = left + width + 0.02

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]
	rect_histy = [left_h, bottom, 0.2, height]

	# Start with a rectangular figure
	plt.figure(1, figsize=(8, 8))

	axScatter = plt.axes(rect_scatter)
	axHistx = plt.axes(rect_histx)
	axHisty = plt.axes(rect_histy)

	# No labels
	nullfmt = NullFormatter()
	axHistx.xaxis.set_major_formatter(nullfmt)
	axHisty.yaxis.set_major_formatter(nullfmt)

	# Formatting
	axHistx.set_title(title, fontsize=18)
	axScatter.set_xlabel(labels[0], fontsize=16)
	axScatter.set_ylabel(labels[1], fontsize=16)

	# The scatter plot
	axScatter.errorbar(x, y, xerr=xerr, yerr=yerr, marker='o', linestyle='none')
	axScatter.plot(axScatter.get_xlim(), axScatter.get_xlim(), 'k-')

	axScatter.text(-5.75, 1, 'N = '+str(len(x)), fontsize=13)

	# The histograms
	axHistx.set_xlim(axScatter.get_xlim())
	axHisty.set_ylim(axScatter.get_ylim())

	axHistx.axvline(np.average(x), color='r', linestyle='--')
	axHisty.axhline(np.average(y), color='r', linestyle='--')

	axHistx.axvspan(np.average(x) - np.std(x), np.average(x) + np.std(x), color='r', alpha=0.25)
	axHisty.axhspan(np.average(y) - np.std(y), np.average(y) + np.std(y), color='r', alpha=0.25)

	axHistx.hist(x, bins=25)
	axHisty.hist(y, bins=25, orientation='horizontal')

	axHistx.text(-5.75, 8., 'Mean: '+"{:.2f}".format(np.average(x))+'\n'+r'$\sigma$: '+"{:.2f}".format(np.std(x)), fontsize=13)
	axHisty.text(1, 0.75, 'Mean: '+"{:.2f}".format(np.average(y))+'\n'+r'$\sigma$: '+"{:.2f}".format(np.std(y)), fontsize=13)

	print('Median x: '+str(np.median(x)))
	print('Median y: '+str(np.median(y)))

	# Output file
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

	return

def main():
	# Sculptor
	#plot_mn_fe(['data/scl1_final.csv','data/scl2_final.csv','data/scl6_final.csv'],'figures/mnfe_scltotal.png','Sculptor',snr=[3,5])

	# Ursa Minor
	#plot_mn_fe(['data/umi1_final.csv','data/umi2_final.csv','data/umi3_final.csv'],'figures/mnfe_umitotal.png','Ursa Minor',snr=[3,5])

	# Draco
	#plot_mn_fe(['data/dra1_final.csv','data/dra2_final.csv','data/dra3_final.csv'],'figures/mnfe_dratotal.png','Draco',snr=[3,5])

	# Linelist check using globular cluster
	comparison_plot(['data/newlinelist_data/n2419b_blue_final.csv','data/oldlinelist_data/n2419b_blue_final.csv'],['New linelist [Mn/H]', 'Old linelist [Mn/H]'],'figures/n2419b_linelistcheck.png','NGC 2419')

if __name__ == "__main__":
	main()