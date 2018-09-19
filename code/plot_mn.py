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

def plot_mn_fe(filenames, outfile, title, gratings=None, maxerror=None, snr=None, solar=False, typeii=True, typei=True):
	"""Plot [Mn/Fe] vs [Fe/H] for all the stars in a set of files.

	Inputs:
	filename 	-- list of input filenames
	outfile 	-- name of output file
	title 		-- title of graph

	Keywords:
	gratings 	-- if not None, must be list of gratings used for input filenames.
					Plot points from different gratings in different colors.
	maxerror 	-- if not None, points with error > maxerror will not be plotted
	solar 		-- if 'True', plot line marking solar abundance
	typeii 		-- if 'True', plot theoretical Type II yield
	typei 		-- if 'True', plot theoretical Type II yields
	"""

	# Get data from files
	name 	= []
	feh 	= []
	feherr 	= []
	mnh 	= []
	mnherr	= []
	colors  = []
	redchisq = []

	for i in range(len(filenames)):

		file = filenames[i]
		current_name 	= np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')

		data = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=[5,6,8,9,10])
		current_feh 	= data[:,0]
		current_feherr 	= data[:,1]
		current_mnh 	= data[:,2]
		current_mnherr 	= data[:,3]
		current_redchisq = data[:,4]

		# Append to total data arrays
		name.append(current_name)
		feh.append(current_feh)
		feherr.append(current_feherr)
		mnh.append(current_mnh)
		mnherr.append(current_mnherr)
		redchisq.append(current_redchisq)

		if gratings is not None:
			current_grating = len(current_feh) * [gratings[i]]
			colors.append(current_grating)

	# Convert back to numpy arrays
	name 	= np.hstack(name)
	feh 	= np.hstack(feh)
	feherr 	= np.hstack(feherr)
	mnh 	= np.hstack(mnh)
	mnherr  = np.hstack(mnherr)
	colors  = np.hstack(colors)
	redchisq = np.hstack(redchisq)

	# Compute [Mn/Fe]
	mnfe = mnh - feh
	mnfeerr = np.sqrt(np.power(feherr,2.)+np.power(mnherr,2.))

	# Remove points with error > maxerror
	if maxerror is not None:
		mask 	= np.where((mnfeerr < maxerror)) # & (redchisq < 3.0))
		name 	= name[mask]
		feh 	= feh[mask]
		mnfe 	= mnfe[mask]
		mnfeerr = mnfeerr[mask]
		colors  = colors[mask]

	# Testing: Label some stuff
	outlier = np.where(mnfe > 1)[0]
	print(name[outlier])
	notoutlier = np.where(mnfe > 0.5)[0]

	#for i in range(len(outlier)):
	#	idx = outlier[i]
	#	plt.text(feh[idx], mnfe[idx], name[idx])

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
	plt.legend(loc='best')
	'''

	# Create figure
	fig, ax = plt.subplots(figsize=(10,5))

	# Plot other lines
	if typeii:
		typeii_Z  = np.array([0.000000000001, 0.001,0.004,0.02])			# Metallicities from Nomoto+2006 (in total Z)
		typeii_mn = np.array([8.72e-7, 9.72e-7, 1.16e-6, 1.54e-6])	# Mn yields from Nomoto+2006 (in units of Msun)
		typeii_fe54 = np.array([7.32e-6, 8.34e-6, 9.03e-6, 1.13e-5]) # Fe-54 yields from Nomoto+2006 (in units of Msun)
		typeii_fe56 = np.array([3.17e-4, 3.38e-4, 3.22e-4, 3.48e-4]) # Fe-56 yields from Nomoto+2006 (in units of Msun)
		typeii_fe57 = np.array([4.66e-6, 6.03e-6, 6.79e-6, 9.57e-6]) # Fe-57 yields from Nomoto+2006 (in units of Msun)
		typeii_fe58 = np.array([6.15e-12, 1.81e-7, 5.26e-7, 2.15e-6]) # Fe-58 yields from Nomoto+2006 (in units of Msun)

		solar_mn = 5.43
		solar_fe = 7.50

		typeii_mnh = np.log10((typeii_mn/55.) / ((typeii_fe54/54.) + (typeii_fe56/56.) + (typeii_fe57/57.) + (typeii_fe58/58.))) - (solar_mn - solar_fe)
		typeii_feh = np.log10(typeii_Z/0.0134) # Solar metallicity from Asplund+2009

		ax.plot(typeii_feh, typeii_mnh, 'b-', label='CCSNe yield')

	if typei:
		#xmin=(-2.34+3)/2.5, 
		ax.axhline(0.18, color='b', linestyle='dashed', label='DDT(T16)')
		ax.axhspan(0.01, 0.53, color='g', hatch='\\', alpha=0.2, label='DDT(S13)')
		ax.axhspan(0.36, 0.52, color='darkorange', hatch='//', alpha=0.3, label='Def(F14)')
		ax.axhspan(-1.69, -1.21, color='r', alpha=0.2, label='Sub(B)')
		ax.axhspan(-1.52, -0.68, color='b', hatch='//', alpha=0.2, label='Sub(S18)')

	if solar:
		ax.axhline(0, color='gold', linestyle='solid')

	#if fit == 'scl':

	# Scatter plot
	area = 2*np.reciprocal(np.power(mnfeerr,2.))
	ax.scatter(feh, mnfe, s=area, c=colors, alpha=0.5, zorder=100) #, label='N = '+str(len(name)))
	ax.text(0.025, 0.9, 'N = '+str(len(name)), transform=ax.transAxes, fontsize=14)

	# Format plot
	ax.set_title(title, fontsize=18)
	ax.set_xlabel('[Fe/H]', fontsize=16)
	ax.set_ylabel('[Mn/Fe]', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)

	ax.set_xlim([-3,-0.75])
	ax.set_ylim([-2,2])
	plt.legend(loc='best')

	# Print labels
	#for i in range(len(feh)):
	#	if feh[i] < -3.5:
	#		ax.text(feh[i], mnfe[i], name[i])

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

	textx_left = -2.85
	textx_right = 1
	texty_up = 5.
	texty_down = -1.2
	texty_down_adjscatter = 0.1

	axScatter.text(textx_left, texty_down + texty_down_adjscatter, 'N = '+str(len(x)), fontsize=13)

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

	# The scatter plot
	axScatter.errorbar(x, y, xerr=xerr, yerr=yerr, marker='o', linestyle='none')
	axScatter.plot(axScatter.get_xlim(), axScatter.get_xlim(), 'k-')

	# The histograms
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

def plot_mn_vs_something(filename, quantity, outfile, title, membercheck=None, memberlist=None, maxerror=None, weighted=True):
	"""Compare [Mn/H] vs another quantity for two different files.

	Inputs:
	filename 	-- list of input filename
	quantity 	-- quantity against which to plot [Mn/H]; options: 'temp', 'feh'
	outfile 	-- name of output file
	title 		-- title of graph

	Keywords:
	membercheck -- do membership check for this object
	memberlist	-- member list
	maxerror	-- if not 'None', throw out any objects with measurement error > maxerror
	weighted 	-- if 'True', compute weighted mean/std; else, compute unweighted mean/std
	"""

	# Get data
	name 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=0, dtype='str')
	mnh 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=8)
	mnherr 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=9)

	if quantity=='temp':
		x 	= np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=3)
		xlabel = r'$T_{eff}$' + ' (K)'

	if quantity=='feh':
		x = np.genfromtxt(filename, delimiter='\t', skip_header=1, usecols=5)
		xlabel = '[Fe/H]'

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
	fig, ax = plt.subplots(figsize=(8,6))
	#area = 2*np.reciprocal(np.power(mnfeerr,2.))
	ax.errorbar(x, mnh, yerr=mnherr, marker='o', linestyle='None')

	if weighted:
		stats = DescrStatsW(mnh, weights=np.reciprocal(np.asarray(mnherr)**2.), ddof=0)

		mean = stats.mean
		std  = stats.std

	else:
		mean = np.average(mnh)
		std  = np.std(mnh)

	ax.axhline(mean, color='r', linestyle='--')
	ax.axhspan(mean - std, mean + std, color='r', alpha=0.25)

	string = 'N = '+str(len(x))+'\n'+'Mean: '+"{:.2f}".format(mean)+'\n'+r'$\sigma$: '+"{:.2f}".format(std)
	ax.text(0.05, 0.8, string, transform=ax.transAxes, fontsize=14)

	# Format plot
	ax.set_title(title, fontsize=18)
	ax.set_xlabel(xlabel, fontsize=16)
	ax.set_ylabel('[Mn/H]', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)
	ax.set_ylim([-3.3,-1.1])

	# Print labels
	#for i in range(len(x)):
	#	if mnh[i] > -1.75:
	#		ax.text(x[i], mnh[i], name[i])

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
	plot_mn_fe(['data/scl1_final.csv','data/scl2_final.csv','data/scl6_final.csv','data/scl5_1200B_final.csv'],
			'figures/mnfe_scltotal_newlinelist.png','Sculptor',gratings=['k','k','k','r'],maxerror=0.3,solar=False,typei=True,typeii=False) #,snr=[3,5])

	# Ursa Minor
	#plot_mn_fe(['data/umi1_final.csv','data/umi2_final.csv','data/umi3_final.csv'],'figures/mnfe_umitotal.png','Ursa Minor',snr=[3,5])

	# Draco
	#plot_mn_fe(['data/dra1_final.csv','data/dra2_final.csv','data/dra3_final.csv'],'figures/mnfe_dratotal.png','Draco',snr=[3,5])

	# Linelist check using globular cluster
	#comparison_plot(['data/newlinelist_data/n2419b_blue_final.csv','data/oldlinelist_data/n2419b_blue_final.csv'],['New linelist [Mn/H]', 'Old linelist [Mn/H]'],'figures/gc_checks/n2419b_linelistcheck.png','NGC 2419', membercheck='NGC 2419', memberlist='data/gc_checks/table_catalog.dat', maxerror=1) #, weighted=False)
	#plot_mn_fe(['data/newlinelist_data/n2419b_blue_final.csv',],'figures/gc_checks/mnfe_n2419total.png','NGC 2419') #,snr=[3,5])
	#plot_mn_vs_something('data/newlinelist_data/n2419b_blue_final.csv', 'feh', 'figures/gc_checks/n2419b_mnh_temp.png','NGC 2419', membercheck='NGC 2419', memberlist='data/gc_checks/table_catalog.dat', maxerror=1, weighted=True)
	#plot_mn_vs_something('data/n2419b_blue_final.csv', 'temp', 'figures/gc_checks/n2419b_mnh_temp.png','NGC 2419', membercheck='NGC 2419', memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, weighted=True)

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