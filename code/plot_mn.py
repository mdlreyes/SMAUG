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
from astropy.io import fits, ascii
import pandas
from matplotlib.ticker import NullFormatter
from statsmodels.stats.weightstats import DescrStatsW

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

def comparison_plot(filenames, labels, outfile, title, membercheck=None, memberlist=None, maxerror=None, weighted=True):
	"""Compare [Mn/H] vs [Mn/H] for two different files.

	Inputs:
	filenames 	-- list of input filenames (must only have 2 files!)
	labels		-- labels for input filenames
	outfile 	-- name of output file
	title 		-- title of graph

	Keywords:
	membercheck -- do membership check for this object
	memberlist	-- member list
	maxerror	-- if not 'None', throw out any objects with measurement error > maxerror
	weighted 	-- if 'True', compute weighted mean/std; else, compute unweighted mean/std
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
	name_final = []

	for i in range(len(x_name)):
		if x_name[i] in y_name:
			x.append(x_mnh[i])
			xerr.append(x_mnherr[i])

			idx = np.where(y_name == x_name[i])
			y.append(y_mnh[idx][0])
			yerr.append(y_mnherr[idx][0])

			name_final.append(x_name[i])

	# Do membership check
	if membercheck is not None:
		x_new = []
		y_new = []
		xerr_new = []
		yerr_new = []

		table = ascii.read(memberlist)
		memberindex = np.where(table.columns[0] == membercheck)
		membernames = table.columns[1][memberindex]

		for i in range(len(name_final)):
			if name_final[i] in membernames:
				x_new.append(x[i])
				y_new.append(y[i])
				xerr_new.append(xerr[i])
				yerr_new.append(yerr[i])

				print(name_final[i], x[i], y[i], xerr[i], yerr[i])

		x = x_new
		y = y_new
		xerr = xerr_new
		yerr = yerr_new

	# Do check for max errors
	if maxerror is not None:
		x_new = []
		y_new = []
		xerr_new = []
		yerr_new = []

		for i in range(len(x)):
			if (xerr[i] < maxerror) and (yerr[i] < maxerror):
				x_new.append(x[i])
				y_new.append(y[i])
				xerr_new.append(xerr[i])
				yerr_new.append(yerr[i])

		x = x_new
		y = y_new
		xerr = xerr_new
		yerr = yerr_new

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

	# The histograms
	axHistx.set_xlim(axScatter.get_xlim())
	axHisty.set_ylim(axScatter.get_ylim())

	axHistx.axvline(np.average(x), color='r', linestyle='--')
	axHisty.axhline(np.average(y), color='r', linestyle='--')

	axHistx.axvspan(np.average(x) - np.std(x), np.average(x) + np.std(x), color='r', alpha=0.25)
	axHisty.axhspan(np.average(y) - np.std(y), np.average(y) + np.std(y), color='r', alpha=0.25)

	axHistx.hist(x, bins=15)
	axHisty.hist(y, bins=15, orientation='horizontal')

	textx_left = -2.85
	textx_right = 1
	texty_up = 3.75
	texty_down = -1.4
	texty_down_adjscatter = 0.1

	axScatter.text(textx_left, texty_down + texty_down_adjscatter, 'N = '+str(len(x)), fontsize=13)

	if weighted:
		weighted_stats_x = DescrStatsW(x, weights=np.reciprocal(np.asarray(xerr)**2.), ddof=0)
		weighted_stats_y = DescrStatsW(y, weights=np.reciprocal(np.asarray(yerr)**2.), ddof=0)

		axHistx.text(textx_left, texty_up, 'Mean: '+"{:.2f}".format(weighted_stats_x.mean)+'\n'+r'$\sigma$: '+"{:.2f}".format(weighted_stats_x.std), fontsize=13)
		axHisty.text(textx_right, texty_down, 'Mean: '+"{:.2f}".format(weighted_stats_y.mean)+'\n'+r'$\sigma$: '+"{:.2f}".format(weighted_stats_y.std), fontsize=13)

	else:
		axHistx.text(textx_left, texty_up, 'Mean: '+"{:.2f}".format(np.average(x))+'\n'+r'$\sigma$: '+"{:.2f}".format(np.std(x)), fontsize=13)
		axHisty.text(textx_right, texty_down, 'Mean: '+"{:.2f}".format(np.average(y))+'\n'+r'$\sigma$: '+"{:.2f}".format(np.std(y)), fontsize=13)

	print('Median x: '+str(np.median(x)))
	print('Median y: '+str(np.median(y)))

	# Output file
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

	return

def plot_mn_vs_something(filename, quantity, outfile, title, membercheck=None, memberlist=None, maxerror=None):
	"""Compare [Mn/H] vs another quantity for two different files.

	Inputs:
	filename 	-- list of input filename
	quantity 	-- quantity against which to plot [Mn/H]; options: 'temp'
	outfile 	-- name of output file
	title 		-- title of graph

	Keywords:
	membercheck -- do membership check for this object
	memberlist	-- member list
	maxerror	-- if not 'None', throw out any objects with measurement error > maxerror
	"""

	# Get data
	name 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=0, dtype='str')
	mnh 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=8)
	mnherr 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=9)

	if quantity=='temp':
		x 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=3)
		xlabel = 'Temp (K)'

	# Do membership check
	if membercheck is not None:
		name_new = []
		x_new = []
		mnh_new = []
		mnherr_new = []

		table = ascii.read(memberlist)
		memberindex = np.where(table.columns[0] == membercheck)
		membernames = table.columns[1][memberindex]

		for i in range(len(name)):
			if name[i] in membernames:
				x_new.append(x[i])
				mnh_new.append(mnh[i])
				mnherr_new.append(mnherr[i])
				name_new.append(name[i])

		x = x_new
		mnh = mnh_new
		mnherr = mnherr_new
		name = name_new

	# Do check for max errors
	if maxerror is not None:
		x_new = []
		mnh_new = []
		mnherr_new = []
		name_new = []

		for i in range(len(x)):
			if (mnherr[i] < maxerror):
				x_new.append(x[i])
				mnh_new.append(mnh[i])
				mnherr_new.append(mnherr[i])
				name_new.append(name[i])

		x = x_new
		mnh = mnh_new
		mnherr = mnherr_new
		name = name_new

	# Plot stuff
	# Scatter plot
	fig, ax = plt.subplots(figsize=(10,6))
	#area = 2*np.reciprocal(np.power(mnfeerr,2.))
	ax.errorbar(x, mnh, yerr=mnherr, marker='o', linestyle='None')
	ax.text(0.025, 0.05, 'N = '+str(len(x)), transform=ax.transAxes, fontsize=14)

	# Format plot
	ax.set_title(title, fontsize=18)
	ax.set_xlabel(xlabel, fontsize=16)
	ax.set_ylabel('[Mn/H]', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)

	# Print labels
	for i in range(len(x)):
		if mnh[i] > -1.75:
			ax.text(x[i], mnh[i], name[i])

	# Output file
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

	return

def plot_spectrum(filename, starname, outfile, lines):
	"""Plot observed spectra.

    Inputs:
    filename 	-- list of input filename
    starname 	-- name of star to plot
	outfile 	-- name of output file
	lines 		-- Mn lines to mark
    """

	print('Opening ', filename)
	hdu1 = fits.open(filename)
	data = hdu1[1].data

	namearray = data['OBJNAME']
	wavearray = data['LAMBDA']
	fluxarray = data['SPEC']
	ivararray = data['IVAR']
	dlamarray = data['DLAM']

	# Get spectrum of a single star
	nameidx = np.where(namearray == starname)
	wvl  = wavearray[nameidx][0]
	flux = fluxarray[nameidx][0]
	ivar = ivararray[nameidx][0]
	dlam = dlamarray[nameidx][0]

	print(flux)

	# Make plot
	fig, ax = plt.subplots(figsize=(12,6))
	ax.plot(wvl, flux, linestyle='-', color='k', marker='None')

	# Format plot
	ax.set_title(starname, fontsize=18)
	ax.set_xlabel('Wavelength (A)', fontsize=16)
	ax.set_ylabel('Flux', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)

	for i in range(len(lines)):
		plt.axvline(lines[i], color='r', linestyle='--')

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
	comparison_plot(['data/newlinelist_data/n2419b_blue_final.csv','data/oldlinelist_data/n2419b_blue_final.csv'],['New linelist [Mn/H]', 'Old linelist [Mn/H]'],'figures/gc_checks/n2419b_linelistcheck.png','NGC 2419', membercheck='NGC 2419', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=False)
	#plot_mn_vs_something('data/newlinelist_data/n2419b_blue_final.csv', 'temp', 'figures/n2419b_mnh_temp.png','NGC 2419', membercheck='NGC 2419', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.5)

	'''
	newlinelist = [4739.087, 4754.042, 4761.512, 4762.367, 4765.846, 
					4766.418, 4783.427, 4823.524, 5394.677, 5399.5, 
					5407.419, 5420.355, 5432.546, 5516.774, 5537.72, 
					6013.51, 6016.68, 6021.82, 6384.67, 6491.69]
	plot_spectrum('data/gc_checks/ngc2419b_blue/moogify.fits.gz', 'N2419-S1604', 'figures/n2419b_s1604.png', newlinelist)
	plot_spectrum('data/gc_checks/ngc2419b_blue/moogify.fits.gz', 'N2419-S243', 'figures/n2419b_s243.png', newlinelist)
	plot_spectrum('data/gc_checks/ngc2419b_blue/moogify.fits.gz', 'N2419-S327', 'figures/n2419b_s327.png', newlinelist)
	'''

if __name__ == "__main__":
	main()