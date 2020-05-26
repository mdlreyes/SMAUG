# plot_mn.py
# Make plots.
#
# Created 22 June 18
# Updated 11 Aug 18
###################################################################

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import os
import sys
import numpy as np
import math
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
import scipy.optimize as op
import pandas as pd
from matplotlib.ticker import NullFormatter
from statsmodels.stats.weightstats import DescrStatsW
from cycler import cycler

def comparison_plot(file1, file2, label1, label2, outfile, title, file1_mnh=False, file2_mnh=False, membercheck=None, memberlist=None, maxerror=None, weighted=True, checkcoords=False):
	"""Compare [Mn/Fe] vs [Mn/Fe] for two different files.

	Inputs:
	file1, label1 -- single file to compare against, label of filename
	file2, label2 -- single file OR list of files to compare against, label(s) of file(s);
					 if file2 is a list of files, last string in label2 is the main axis label
	outfile 	-- name of output file
	title 		-- title of graph

	Keywords:
	file1_mnh, file2_mnh -- bool or list of bools; if True, need to convert [Mn/H] to [Mn/Fe]
	membercheck -- do membership check for this object
	memberlist	-- member list
	maxerror	-- if not 'None', throw out any objects with measurement error > maxerror
	weighted 	-- if 'True', compute weighted mean/std; else, compute unweighted mean/std
	checkcoords -- if 'True', check both files to see if coordinates overlap
	"""

	# Get data from single file
	y_name 	 = np.genfromtxt(file1, delimiter='\t', skip_header=1, usecols=0, dtype='str')

	if file1_mnh:
		# Convert [Mn/H] to [Fe/H]
		y_mnh 	 = np.genfromtxt(file1, delimiter='\t', skip_header=1, usecols=8)
		y_mnherr = np.genfromtxt(file1, delimiter='\t', skip_header=1, usecols=9)
		y_feh 	 = np.genfromtxt(file1, delimiter='\t', skip_header=1, usecols=5)
		y_feherr = np.genfromtxt(file1, delimiter='\t', skip_header=1, usecols=6)

		y_mnfe 	  = y_mnh - y_feh
		y_mnfeerr = y_mnherr#np.sqrt(np.power(y_mnherr,2.)+np.power(y_feherr,2.))

	else:
		y_mnfe 	  = np.genfromtxt(file1, delimiter='\t', skip_header=1, usecols=8)
		y_mnfeerr = np.genfromtxt(file1, delimiter='\t', skip_header=1, usecols=9)

	# Start with a rectangular figure
	plt.figure(1, figsize=(8, 8))

	# Make histograms if only comparing 2 files
	if len(file2) == 1:
		left, width = 0.1, 0.65
		bottom, height = 0.1, 0.65
		bottom_h = left_h = left + width + 0.02

		rect_scatter = [left, bottom, width, height]
		rect_histx = [left, bottom_h, width, 0.2]
		rect_histy = [left_h, bottom, 0.2, height]

		axScatter = plt.axes(rect_scatter)
		axHistx = plt.axes(rect_histx)
		axHisty = plt.axes(rect_histy)

		# No labels on histograms
		nullfmt = NullFormatter()
		axHistx.xaxis.set_major_formatter(nullfmt)
		axHisty.yaxis.set_major_formatter(nullfmt)

		# Formatting
		axHistx.set_title(title, fontsize=18)
		axScatter.set_xlabel(label2[-1], fontsize=16)
		axScatter.set_ylabel(label1, fontsize=16)

		textx_left = -1.95
		textx_right = -1.55
		texty_up = -1.6
		texty_down = -3.0
		texty_down_adjscatter = 0.1

	else:
		axScatter = plt.gca()
		axScatter.set_xlabel(label2[-1], fontsize=16)
		axScatter.set_ylabel(label1, fontsize=16)

	#axScatter.text(textx_left, texty_down + texty_down_adjscatter, 'N = '+str(len(x)), fontsize=13)

	# Define new property cycles
	new_prop_cycle = cycler('marker', ['o','^','s','D','*','x','+','v'])
	axScatter.set_prop_cycle(new_prop_cycle)

	# Loop over all x-axis files
	for filenum in range(len(file2)):
		x_name   = np.genfromtxt(file2[filenum], delimiter='\t', skip_header=1, usecols=0, dtype='str')

		# Get data
		if file2_mnh[filenum]:
			# Convert [Mn/H] to [Fe/H]
			x_mnh 	 = np.genfromtxt(file2[filenum], delimiter='\t', skip_header=1, usecols=8)
			x_mnherr = np.genfromtxt(file2[filenum], delimiter='\t', skip_header=1, usecols=9)
			x_feh 	 = np.genfromtxt(file2[filenum], delimiter='\t', skip_header=1, usecols=5)
			x_feherr = np.genfromtxt(file2[filenum], delimiter='\t', skip_header=1, usecols=6)

			x_mnfe 	  = x_mnh - x_feh
			x_mnfeerr = x_mnherr #np.sqrt(np.power(x_mnherr,2.)+np.power(x_feherr,2.))

		else:
			x_mnfe 	  = np.genfromtxt(file2[filenum], delimiter='\t', skip_header=1, usecols=8)
			x_mnfeerr = np.genfromtxt(file2[filenum], delimiter='\t', skip_header=1, usecols=9)

		# Match catalogs
		x = []
		y = []
		xerr = []
		yerr = []
		name_final = []

		# If checkcoords==True, match catalogs based on separation
		if checkcoords:
			x_ra 	= np.genfromtxt(file2[filenum], delimiter='\t', skip_header=1, usecols=1, dtype='str')
			x_dec	= np.genfromtxt(file2[filenum], delimiter='\t', skip_header=1, usecols=2, dtype='str')

			y_ra 	= np.genfromtxt(file1, delimiter='\t', skip_header=1, usecols=1, dtype='str')
			y_dec	= np.genfromtxt(file1, delimiter='\t', skip_header=1, usecols=2, dtype='str')

			x_coord = SkyCoord(x_ra, x_dec, frame='icrs', unit='deg')
			y_coord = SkyCoord(y_ra, y_dec, frame='icrs', unit='deg')

			for i in range(len(x_name)):
				idx, sep, _ = x_coord[i].match_to_catalog_sky(y_coord) 

				if sep.arcsec < 10:
					print('Got one! Separation: ', sep.arcsecond, 'Name: ', x_name[i], y_name[idx])
					x.append(x_mnfe[i])
					xerr.append(x_mnfeerr[i])

					#print(x_mnfe[i], y_mnfe[idx])
					y.append(y_mnfe[idx])
					yerr.append(y_mnfeerr[idx])

		# Else, match catalogs to make sure correct values are being plotted against one another
		for i in range(len(x_name)):
			if x_name[i] in y_name:
				x.append(x_mnfe[i])
				xerr.append(x_mnfeerr[i])

				idx = np.where(y_name == x_name[i])
				y.append(y_mnfe[idx][0])
				yerr.append(y_mnfeerr[idx][0])

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

					#print(name_final[i], x[i], y[i], xerr[i], yerr[i])

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
				if (xerr[i] < maxerror) and (yerr[i] < maxerror): # and (x[i] < -1.8) and (y[i] < -1.8):
					x_new.append(x[i])
					y_new.append(y[i])
					xerr_new.append(xerr[i])
					yerr_new.append(yerr[i])

			x = x_new
			y = y_new
			xerr = xerr_new
			yerr = yerr_new

		# Plot points on scatter plot
		axScatter.errorbar(x, y, xerr=xerr, yerr=yerr, marker='o', linestyle='none', label=label2[filenum])
		axScatter.plot(axScatter.get_xlim(), axScatter.get_xlim(), 'k-')

	# Make histograms if necessary
	if len(file2) == 1:

		# Compute values
		if weighted:
			weighted_stats_x = DescrStatsW(x, weights=np.reciprocal(np.asarray(xerr)**2.), ddof=0)
			weighted_stats_y = DescrStatsW(y, weights=np.reciprocal(np.asarray(yerr)**2.), ddof=0)

			axHistx.text(textx_left, texty_up, 'Mean: '+"{:.2f}".format(weighted_stats_x.mean)+'\n'+r'$\sigma$: '+"{:.2f}".format(weighted_stats_x.std), fontsize=13)
			axHisty.text(textx_right, texty_down, 'Mean: '+"{:.2f}".format(weighted_stats_y.mean)+'\n'+r'$\sigma$: '+"{:.2f}".format(weighted_stats_y.std), fontsize=13)

			meanx = weighted_stats_x.mean
			meany = weighted_stats_y.mean
			stdx  = weighted_stats_x.std
			stdy  = weighted_stats_y.std

		else:
			axHistx.text(textx_left, texty_up, 'Mean: '+"{:.2f}".format(np.average(x))+'\n'+r'$\sigma$: '+"{:.2f}".format(np.std(x)), fontsize=13)
			axHisty.text(textx_right, texty_down, 'Mean: '+"{:.2f}".format(np.average(y))+'\n'+r'$\sigma$: '+"{:.2f}".format(np.std(y)), fontsize=13)

			meanx = np.average(x)
			meany = np.average(y)
			stdx  = np.std(x)
			stdy  = np.std(y)

		axHistx.set_xlim(axScatter.get_xlim())
		axHisty.set_ylim(axScatter.get_ylim())

		axHistx.axvline(meanx, color='r', linestyle='--')
		axHisty.axhline(meany, color='r', linestyle='--')

		axHistx.axvspan(meanx - stdx, meanx + stdx, color='r', alpha=0.25)
		axHisty.axhspan(meany - stdy, meany + stdy, color='r', alpha=0.25)

		axHistx.hist(x, bins=10)
		axHisty.hist(y, bins=10, orientation='horizontal')

		print('Median x: '+str(np.median(x)))
		print('Median y: '+str(np.median(y)))

	# Output file
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

	return

def plot_mn_vs_something(filename, quantity, outfile, title, membercheck=None, memberlist=None, maxerror=None, weighted=True, plotfeh=False, sigmasys=False):
	"""Compare [Mn/Fe] vs another quantity for two different files.

	Inputs:
	filename 	-- list of input filename
	quantity 	-- quantity against which to plot [Mn/H]; options: 'temp', 'feh', 'logg'
	outfile 	-- name of output file
	title 		-- title of graph

	Keywords:
	membercheck -- do membership check for this object
	memberlist	-- member list
	maxerror	-- if not 'None', throw out any objects with measurement error > maxerror
	weighted 	-- if 'True', compute weighted mean/std; else, compute unweighted mean/std
	plotfeh 	-- if 'True', plot [Fe/H] instead of [Mn/Fe] on y-axis
	sigmasys 	-- if 'True', compute sigma_sys
	"""

	# Get data
	name 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=0, dtype='str')
	mnh 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=8)
	mnherr 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=9)
	feh 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=5)
	feherr 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=6)
	#feherr 	= np.ones(len(feh))*0.1

	# Convert [Mn/H] to [Mn/Fe]
	mnfe 	= mnh - feh
	mnfeerr = mnherr #np.sqrt(np.power(feherr,2.)+np.power(mnherr,2.))
	print('sigmastat: ', np.average(mnh))

	if quantity=='temp':
		x = np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=3)
		xlabel = r'$T_{eff}$' + ' (K)'

	if quantity=='feh':
		x = feh
		xlabel = '[Fe/H]'

	if quantity=='logg':
		x = np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=4)
		xlabel = 'Log(g) [cm/s'+r'$^{2}$]'

	if plotfeh:
		# Use [Fe/H] instead of [Mn/Fe]
		mnfe = feh
		mnfeerr = feherr

	#print(mnfeerr, feherr)

	# Do membership check
	if membercheck is not None:
		name_new = []
		x_new = []
		mnfe_new = []
		mnfeerr_new = []

		table = ascii.read(memberlist)
		memberindex = np.where(table.columns[0] == membercheck)
		membernames = table.columns[1][memberindex]

		for i in range(len(name)):
			if name[i] in membernames:
				x_new.append(x[i])
				mnfe_new.append(mnfe[i])
				mnfeerr_new.append(mnfeerr[i])
				name_new.append(name[i])

		x = x_new
		mnfe = mnfe_new
		mnfeerr = mnfeerr_new
		name = name_new

	# Do check for max errors
	if maxerror is not None:
		x_new = []
		mnfe_new = []
		mnfeerr_new = []
		name_new = []

		for i in range(len(x)):
			if (mnfeerr[i] < maxerror):
				x_new.append(x[i])
				mnfe_new.append(mnfe[i])
				mnfeerr_new.append(mnfeerr[i])
				name_new.append(name[i])

		x = x_new
		mnfe = mnfe_new
		mnfeerr = mnfeerr_new
		name = name_new

	outliers = np.where((np.asarray(mnfe) > -0.2) & (np.asarray(feh) < -1.6)) #| (np.asarray(mnfe) > 0.1))[0]
	print('Outliers:')
	print(np.asarray(name)[outliers])
	print(np.asarray(mnfe)[outliers])
	print(np.asarray(x)[outliers])

	#np.savetxt('data/gc_checks/7089_parallaxes.csv', name, fmt='%s')

	'''
	check = np.where((np.asarray(mnfe) < -1.1) & (np.asarray(mnfe) > -2.5))[0]
	x = np.asarray(x)[check]
	mnfe = np.asarray(mnfe)[check]
	mnfeerr = np.asarray(mnfeerr)[check]
	name = np.asarray(name)[check]
	'''

	# Compute sigma_sys
	if sigmasys:
		avg = np.mean(mnfe)

		test = np.linspace(0,1,1000)

		check = []
		for i in range(len(test)):
			disp = np.std( (mnfe-avg)/np.sqrt(np.power(mnfeerr,2.) + np.power(test[i],2.)) ) - 1.
			check.append(disp)
			#print(test[i], disp)

		#print(np.min(np.abs(np.asarray(check))))
		sigma_sys = test[np.argmin(np.abs(np.asarray(check)))]
		print('Sigma_sys: ', sigma_sys)
		print('Sigma_stat: ', np.average(mnfeerr))

		'''
		# Make plot
		fig, ax = plt.subplots(1, 1, tight_layout=True)
		ax.hist((mnfe-avg)/np.sqrt(np.power(mnfeerr,2.) + np.power(sigma_sys,2.)) )
		plt.savefig(outfile[:-4]+'hist.png', bbox_inches='tight')
		plt.show()
		'''

		# Double check and compute reduced chisq!
		rchisq = np.sum( np.power((mnfe - avg),2.)/np.power(mnfeerr,2.) ) /len(mnfe)
		print(rchisq)

	# Plot stuff
	# Scatter plot
	fig, ax = plt.subplots(figsize=(8,6))
	#area = 2*np.reciprocal(np.power(mnfeerr,2.))
	ax.errorbar(x, mnfe, yerr=mnfeerr, marker='o', linestyle='None')

	if weighted:
		stats = DescrStatsW(mnfe, weights=np.reciprocal(np.asarray(mnfeerr)**2.), ddof=0)

		mean = stats.mean
		std  = stats.std

	else:
		mean = np.average(mnfe)
		std  = np.std(mnfe)

	ax.axhline(mean, color='r', linestyle='--')
	ax.axhspan(mean - std, mean + std, color='r', alpha=0.25)

	string = 'N = '+str(len(x))+'\n'+'Mean: '+"{:.2f}".format(mean)+'\n'+r'$\sigma$: '+"{:.2f}".format(std)
	ax.text(0.05, 0.8, string, transform=ax.transAxes, fontsize=14)

	# Format plot
	ax.set_title(title, fontsize=18)
	ax.set_xlabel(xlabel, fontsize=16)
	ax.set_ylabel('[Mn/Fe]', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)
	#ax.set_ylim([-3.3,-1.1])

	# Print labels
	#for i in range(len(x)):
	#	if mnfe[i] > -1.75:
	#		ax.text(x[i], mnfe[i], name[i])

	# Output file
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

	return

def main():
	# Globular cluster checks
	# Linelist check using globular cluster
	#comparison_plot(['data/newlinelist_data/n2419b_blue_final.csv','data/oldlinelist_data/n2419b_blue_final.csv'],['New linelist [Mn/H]', 'Old linelist [Mn/H]'],'figures/gc_checks/n2419b_linelistcheck.png','NGC 2419', membercheck='NGC 2419', memberlist='data/gc_checks/table_catalog.dat', maxerror=1) #, weighted=False)
	#plot_mn_fe(['data/newlinelist_data/n2419b_blue_final.csv',],'figures/gc_checks/mnfe_n2419total.png','NGC 2419') #,snr=[3,5])
	#plot_mn_vs_something('data/newlinelist_data/n2419b_blue_final.csv', 'feh', 'figures/gc_checks/n2419b_mnh_temp.png','NGC 2419', membercheck='NGC 2419', memberlist='data/gc_checks/table_catalog.dat', maxerror=1, weighted=True)
	#plot_mn_vs_something('data/n2419b_blue_final.csv', 'temp', 'figures/gc_checks/n2419b_mnh_temp.png','NGC 2419', membercheck='NGC 2419', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True)

	# Intrinsic dispersion checks for GCs
	#plot_mn_vs_something('data/7078l1_1200B_final.csv', 'temp', 'figures/gc_checks/n7078l1_mnfe_temp_nooutliers.png', 'M15', membercheck='M15', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True, sigmasys=True)
	#plot_mn_vs_something('data/7078l1_1200B_final.csv', 'logg', 'figures/gc_checks/n7078l1_mnfe_logg.png', 'M15', membercheck='M15', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True)
	#plot_mn_vs_something('../data/7078l1_1200B_final.csv', 'feh', '../figures/gc_checks/n7078l1_mnfe_feh_nooutliers.png', 'M15', membercheck='M15', memberlist='../data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True, sigmasys=True)
	#plot_mn_vs_something('data/7078l1_1200B_final.csv', 'temp', 'figures/gc_checks/n7078l1_feh_temp.png', 'M15', membercheck='M15', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True, plotfeh=True)

	#plot_mn_vs_something('data/7089l1_1200B_final.csv', 'temp', 'figures/gc_checks/n7089_mnfe_temp_nooutliers.png', 'M2', membercheck='M2', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True, sigmasys=True)
	#plot_mn_vs_something('data/7089_1200B_final.csv', 'logg', 'figures/gc_checks/n7089_mnfe_logg.png', 'M2', membercheck='M2', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True)
	#plot_mn_vs_something('../data/7089l1_1200B_final.csv', 'feh', '../figures/gc_checks/n7089_mnfe_feh.png', 'M2', membercheck='M2', memberlist='../data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True, sigmasys=True)
	#plot_mn_vs_something('data/7089_1200B_final.csv', 'temp', 'figures/gc_checks/n7089_feh_temp.png', 'M2', membercheck='M2', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True, plotfeh=True)

	#plot_mn_vs_something('data/ng1904_1200B_final.csv', 'temp', 'figures/gc_checks/n1904_mnfe_temp.png', 'M79', membercheck='M79', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True, sigmasys=True)
	#plot_mn_vs_something('data/ng1904_1200B_final.csv', 'logg', 'figures/gc_checks/n1904_mnfe_logg.png', 'M79', membercheck='M79', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True)
	#plot_mn_vs_something('data/ng1904_1200B_final.csv', 'feh', 'figures/gc_checks/n1904_mnfe_feh.png', 'M79', membercheck='M79', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True)
	#plot_mn_vs_something('data/ng1904_1200B_final.csv', 'temp', 'figures/gc_checks/n1904_feh_temp.png', 'M79', membercheck='M79', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True, plotfeh=True)

	#plot_mn_vs_something('data/n5024b_1200B_final.csv', 'temp', 'figures/gc_checks/n5024b_mnfe_temp_nooutliers.png', 'M53', membercheck='M53', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=False, sigmasys=True)
	#plot_mn_vs_something('data/n5024b_1200B_final.csv', 'logg', 'figures/gc_checks/n5024b_mnfe_logg.png', 'M53', membercheck='M53', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True)
	#plot_mn_vs_something('../data/n5024b_1200B_final.csv', 'feh', '../figures/gc_checks/n5024b_mnfe_feh_nooutliers.png', 'M53', membercheck='M53', memberlist='../data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True, sigmasys=True)
	#plot_mn_vs_something('data/n5024b_1200B_final.csv', 'temp', 'figures/gc_checks/n5024b_feh_temp.png', 'M53', membercheck='M53', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True, plotfeh=True)

	#plot_hist(['data/7078l1_1200B_final.csv'], ['M15'], 'error', 'figures/gc_checks/errorhist', membercheck=['M15'], memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, sigmasys=[0.13])
	#plot_hist(['data/7089l1_1200B_final.csv'], ['M2'], 'error', 'figures/gc_checks/errorhist', membercheck=['M2'], memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, sigmasys=[0.21])
	#plot_hist(['data/n5024b_1200B_final.csv'], ['M53'], 'error', 'figures/gc_checks/errorhist', membercheck=['M53'], memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, sigmasys=[0.03])

	# Check if adding smoothing parameter does anything
	#comparison_plot(['data/no_dlam/scl5_1200B_final.csv','data/scl5_1200B.csv'],['Don\'t fit smoothing [Mn/H]', 'Fit smoothing [Mn/H]'],'figures/scl5_1200B_smoothcheck.png','Sculptor', maxerror=1) #, weighted=False)

if __name__ == "__main__":
	main()