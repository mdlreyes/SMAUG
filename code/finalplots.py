# finalplots.py
# Make plots for paper
#
# Created 27 Feb 19
# Updated 27 Feb 19
###################################################################

import matplotlib as mpl
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
from matplotlib.lines import Line2D
import cycler

def gc_mnfe_feh(filenames, outfile, title=None, gratings=None, gratingnames=None, maxerror=None, membercheck=False, solar=False):
	"""Plot [Mn/Fe] vs [Fe/H] for all the stars in a set of files (designed for globular clusters).

	Inputs:
	filename 	-- list of input filenames
	outfile 	-- name of output file

	Keywords:
	title 		-- title of graph
	gratings 	-- if not None, lists of colors and markers to use for plotting different grating colors
	gratingnames -- if not None, list of names to use for different grating colors in legend
	maxerror 	-- if not None, points with error > maxerror will not be plotted
	membercheck -- if 'True', check GCs for membership
	solar 		-- if 'True', plot line marking solar abundance
	"""

	# Create figure
	fig, ax = plt.subplots(figsize=(8,7.8))

	# Define colors
	colors = ['C0','C1','None']
	edges = ['C0','C1','C2']
	markers = ['^','o','s']

	for i in range(len(filenames)):

		# Get data from file
		file = filenames[i]
		name 	= np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')
		data = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=[5,6,8,9,10])
		feh 	= data[:,0]
		feherr 	= data[:,1]
		#feherr 	= np.ones(len(feh)) * 0.1
		mnh 	= data[:,2]
		mnherr 	= data[:,3]
		redchisq = data[:,4]

		# Compute [Mn/Fe]
		mnfe = mnh - feh
		mnfeerr = mnherr #np.sqrt(np.power(feherr,2.)+np.power(mnherr,2.))

		# Remove points with error > maxerror
		if maxerror is not None:
			mask 	= np.where((mnfeerr < maxerror)) # & (redchisq < 3.0))
			name 	= name[mask]
			feh 	= feh[mask]
			mnfe 	= mnfe[mask]
			mnfeerr = mnfeerr[mask]

		# Do membership check
		if membercheck:
			name_new 	= []
			feh_new 	= []
			feherr_new 	= []
			mnfe_new 	= []
			mnfeerr_new = []

			table = ascii.read('data/gc_checks/table_catalog.dat')
			memberindex = np.where(table.columns[0] == gratingnames[i])
			membernames = table.columns[1][memberindex]

			for j in range(len(name)):
				if name[j] in membernames:
					feh_new.append(feh[j])
					feherr_new.append(feherr[j])
					mnfe_new.append(mnfe[j])
					mnfeerr_new.append(mnfeerr[j])
					name_new.append(name[j])

			name = name_new
			feh = feh_new
			feherr = feherr_new
			mnfe = mnfe_new
			mnfeerr = mnfeerr_new

		# Plot solar abundance
		#if solar:
		#	ax.axhline(0, color='r', linestyle=':')

		# Make plot
		ax.errorbar(feh, mnfe, yerr=mnfeerr, xerr=feherr, markerfacecolor=colors[i], markeredgecolor=edges[i], ecolor=edges[i], marker=markers[i], markersize=8, linestyle='', capsize=3, label=gratingnames[i]+' (N='+str(len(feh))+')')
		#ax.text(0.025, 0.9, 'N = '+str(len(name)), transform=ax.transAxes, fontsize=14)

	# Format plot
	if title is not None:
		ax.set_title(title, fontsize=20)
	ax.set_xlabel('[Fe/H]', fontsize=20)
	ax.set_ylabel('[Mn/Fe]', fontsize=20)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(18)

	#ax.set_xlim([-3,-0.75])
	ax.set_ylim([-1,0.4])
	plt.legend(loc='upper left', fontsize=16)

	# Output file
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

	return

def plot_hist(files, labels, quantity, outfile, membercheck=None, memberlist=None, maxerror=0.3, sigmasys=None):
	"""Compare [Mn/Fe] vs another quantity for two different files.

	Inputs:
	filename 	-- list of input filename
	names 		-- label for each input filename
	quantity 	-- quantity to plot; options: 'error', 'mnfe', 'temp', 'feh', 'logg', 'radius'
	outfile 	-- name of output file

	Keywords:
	membercheck -- list of objects to do membership check for
	memberlist	-- member list
	maxerror	-- if not 'None', throw out any objects with measurement error > maxerror
	sigmasys 	-- list of sigma_sys values for each file
	"""

	# Open histogram
	fig, ax = plt.subplots(figsize=(8,7.2))
	if quantity == 'error':
		bins = np.linspace(-3, 3, 15)
	else: bins = 10

	# Open memberlist table
	if membercheck is not None:
		table = ascii.read(memberlist)

	colors = ['C0','C1','None']
	hatches = [None, '/', None]
	edges = ['C0','C1','C2']
	styles = ['-','--',':']

	# Loop over each file
	for i in range(len(files)):
		print('Opening '+files[i])

		# Get data
		data 	= pd.read_csv(files[i], delimiter='\t')

		# Do check for max errors
		if maxerror is not None:
			data = data[data["error([Mn/H])"] < maxerror]

		# Do membership check
		if membercheck is not None:
			N = len(data['Name'])
			checkarray = np.zeros(N, dtype='bool')

			memberindex = np.where(table.columns[0] == membercheck[i])
			membernames = table.columns[1][memberindex]

			for j in range(N):
				if str(np.asarray(data['Name'])[j]) in np.asarray(membernames):
					checkarray[j] = True

			data = data[checkarray]

		# Define data columns
		name 	= data['Name']
		mnh 	= data['[Mn/H]']
		mnherr 	= data['error([Mn/H])']
		feh 	= data['[Fe/H]']
		feherr 	= data['error([Fe/H])']

		# Get quantity
		if quantity=='error':
			mnfe 	= mnh - feh
			mnfeerr = mnherr

			avg = np.mean(mnfe)
			print('Average: ',avg)
			'''
			stats = DescrStatsW(mnfe, weights=np.reciprocal(np.asarray(mnfeerr)**2.), ddof=0)
			avg = stats.mean
			std  = stats.std
			'''

			x = (mnfe - avg)/np.sqrt(np.power(mnfeerr,2.) + np.power(sigmasys[i],2.))
			xlabel = r'$(\mathrm{[Mn/Fe]}-\langle\mathrm{[Mn/Fe]}\rangle)/\sqrt{\sigma_{\mathrm{stat}}^{2}+\sigma_{\mathrm{sys}}^{2}}$'

		elif quantity=='mnfe':
			x = mnh - feh
			xlabel = '[Mn/Fe]'

		elif quantity=='temp':
			x = data['Temp']
			xlabel = r'$T_{eff}$' + ' (K)'

		elif quantity=='feh':
			x = data['[Fe/H]']
			xlabel = '[Fe/H]'

		elif quantity=='logg':
			x = data['log(g)']
			xlabel = 'Log(g) [cm/s'+r'$^{2}$]'

		elif quantity=='radius':
			RA = data['RA']
			Dec = data['Dec']
			starcoords = SkyCoord(RA, Dec, frame='icrs', unit=u.deg)
			centralcoords = SkyCoord(ra='21h33m27.01s', dec='-00d49m23.9s', frame='icrs', unit=(u.hourangle,u.deg)) # central coordinates for M2

			x = starcoords.separation(centralcoords)
			x = x.degree
			xlabel = 'Radius [deg]'

			for i in range(len(RA)):
				if x[i] > 0.05:
					print(data['Name'][i])

		# Plot histogram
		#if i != 2:
		n, bins, _ = ax.hist(x, bins, alpha=0.5, label=labels[i], facecolor=colors[i], hatch=hatches[i], edgecolor=edges[i], fill=True, linewidth=1.5)
		#else: # no label
		#	n, bins, _ = ax.hist(x, bins, alpha=0.5, facecolor=colors[i], hatch=hatches[i], edgecolor=edges[i], fill=True)


		if quantity == 'error':
			# Overplot edges of histogram
			#if i == 2:
			#	ax.hist(x, bins, label=labels[i], facecolor='None', edgecolor=edges[i], linewidth=1.5, fill=True)

			#Get bin width from this
			binwidth = bins[1] - bins[0]

			# Overplot best-fit Gaussian
			xbins = np.linspace(-3,3,1000)
			sigma = 1.
			mu = np.mean(x)
			y = ((1. / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1. / sigma * (xbins - mu))**2))*len(x)*binwidth

			plt.plot(xbins, y, linestyle=styles[i], color=edges[i], linewidth=3)

	# Format plot
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(18)
	plt.xlabel(xlabel, fontsize=20)
	plt.ylabel('N', fontsize=20)
	plt.legend(loc='best', fontsize=16)
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

	return

def plot_hires_comparison(filenames, objnames, glob, offset, sigmasys=True):
	"""Plot hi-res comparisons.

	Inputs:
	filenames -- list of matched files
	objnames  -- list of object names for each file (for plot legend)
	glob 	  -- list of booleans for each file; if 'True', object data are plotted as GC points
	offset 	  -- list of offsets to solar Mn abundance

	Keywords:
	sigmasys  -- if 'True' (default), compute sigma_sys
	"""

	# Set up figure
	fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=False, figsize=(8,12), gridspec_kw={'height_ratios':[4,2]})

	# Format labels
	ax0.set_ylabel(r'$\mathrm{[Mn/Fe]}_{\mathrm{MRS}}$', fontsize=24)
	ax0.set_xlabel(r'$\mathrm{[Mn/Fe]}_{\mathrm{HRS}}$', fontsize=24)
	ax1.set_xlabel(r'$(\mathrm{[Mn/Fe]}_{\mathrm{MRS}}+\mathrm{[Mn/Fe]}_{\mathrm{HRS}})/2$', fontsize=24)
	ax1.set_ylabel(r'$\mathrm{[Mn/Fe]}_{\mathrm{MRS}}-\mathrm{[Mn/Fe]}_{\mathrm{HRS}}$', fontsize=24)

	# Containers for labels (to make different legends)
	globplots = []
	dsphplots = []

	# Containers for all MRS and HRS measurements
	totalHRS = []
	totalHRSerr = []
	totalMRS = []
	totalMRSerr = []

	for filenum in range(len(filenames)):

		file = filenames[filenum]

		data = np.genfromtxt(file, skip_header=1, delimiter='\t', usecols=[0,1,2,3], dtype='float')
		if hasattr(data[0], "__iter__"):
			sources = np.genfromtxt(file, skip_header=1, delimiter='\t', usecols=5, dtype='str')[0]
		else:
			data = np.asarray([data])
			sources = np.genfromtxt(file, skip_header=1, delimiter='\t', usecols=5, dtype='str')			

		# Figure out which marker to use
		if glob[filenum]:
			marker = 'o'
		else:
			marker = 's'

		# Plot direct comparison
		container = ax0.errorbar(x=data[:,2]+offset[filenum], y=data[:,0], xerr=data[:,3], yerr=data[:,1], marker=marker, linestyle='None', markersize=12, elinewidth=1.5)
		plot = Line2D([0],[0], marker=marker, label=sources, color=(container[0].get_color()), linestyle='None', markersize=12)

		if glob[filenum]:
			globplots.append(plot)
		else:
			dsphplots.append(plot)

		# Plot residuals
		avg = (data[:,0] + data[:,2])/2.
		resids = data[:,0] - data[:,2]
		residserr = np.sqrt(np.power(data[:,1],2.) + np.power(data[:,3],2.))
		ax1.errorbar(x=avg, y=resids, yerr=residserr, marker=marker, linestyle='None', markersize=12)

		# Add data to containers
		totalMRS.append(data[:,2]+offset[filenum])
		totalHRS.append(data[:,0])
		totalMRSerr.append(data[:,3])
		totalHRSerr.append(data[:,1])

	# Set ranges
	#Main plot
	xmin = min(ax0.get_xlim()[0],ax0.get_ylim()[0])
	xmax = max(ax0.get_xlim()[1],ax0.get_ylim()[1])
	ax0.set_xlim([xmin,xmax])
	ax0.set_ylim([xmin,xmax])
	#Residual plot
	residlim = max(abs(ax1.get_ylim()[0]),ax1.get_ylim()[1])
	ax1.set_xlim([xmin,xmax])
	ax1.set_ylim([-1.*residlim,residlim])

	#Plot 1-1 lines
	ax0.plot([xmin,xmax],[xmin,xmax],'k:', linewidth=2)
	ax1.plot([xmin,xmax],[0,0],'k:', linewidth=2)

	# Compute systematic error
	if sigmasys:

		totalHRS = np.hstack(totalHRS)
		totalMRS = np.hstack(totalMRS)
		totalHRSerr = np.hstack(totalHRSerr)
		totalMRSerr = np.hstack(totalMRSerr)

		test = np.linspace(0,1,1000)

		check = []
		for i in range(len(test)):
			disp = np.std( (totalMRS-totalHRS)/np.sqrt(np.power(totalMRSerr,2.) + np.power(totalHRSerr,2.) + np.power(test[i],2.)) ) - 1.
			check.append(disp)
			#print(test[i], disp)

		#print(np.min(np.abs(np.asarray(check))))
		sigma_sys = test[np.argmin(np.abs(np.asarray(check)))]
		print('Average: ', np.average(totalMRS-totalHRS))
		print('Sigma_sys: ', sigma_sys)
		ax1.fill_between([xmin, xmax], [sigma_sys, sigma_sys], [-sigma_sys, -sigma_sys], alpha=0.25, zorder=0, color='k')

	# Make legends
	l1 = ax0.legend(handles=dsphplots, bbox_to_anchor=(1.04,0.8), borderaxespad=0, fontsize=18, frameon=False, title='dSphs')
	plt.setp(l1.get_title(), fontsize=20)
	l1._legend_box.align="left"

	l2 = ax0.legend(handles=globplots, bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=18, frameon=False, title='Globular Clusters')
	plt.setp(l2.get_title(), fontsize=20)
	l2._legend_box.align="left"

	ax0.add_artist(l1)

	# Other formatting
	for label in (ax0.get_xticklabels() + ax0.get_yticklabels() + ax1.get_xticklabels() + ax1.get_yticklabels()):
		label.set_fontsize(18)

	plt.savefig('figures/hires_comparison.pdf', bbox_inches='tight')
	plt.show()

	return

def plot_mn_fe(filenames, outfile, title, gratings=None, maxerror=None, snr=None, solar=False, typeii=True, typei=True, averages=False, n=4):
	"""Plot [Mn/Fe] vs [Fe/H] for all the stars in a set of files.

	Inputs:
	filename 	-- list of input filenames
	outfile 	-- name of output file
	title 		-- title of graph

	Keywords:
	gratings 	-- if not None, list of names for different files
	maxerror 	-- if not None, points with error > maxerror will not be plotted
	solar 		-- if 'True', plot line marking solar abundance
	typeii 		-- if 'True', plot theoretical Type II yield
	typei 		-- if 'True', plot theoretical Type II yields
	averages 	-- if 'True', plot binned averages
	n 			-- number of colors for colormap
	"""

	# Get data from files
	name 	= []
	feh 	= []
	feherr 	= []
	mnh 	= []
	mnherr	= []
	redchisq = []

	# Get color wheel
	color = plt.cm.viridis(np.linspace(0,1,n))
	mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
	cwheel = [np.array(mpl.rcParams['axes.prop_cycle'])[x]['color'] for x in range(n)]

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
		ax.axhline(0.18, color='b', linestyle='dashed', label='DDT(T16)')
		ax.axhspan(0.01, 0.53, color='g', hatch='\\', alpha=0.2, label='DDT(S13)')
		ax.axhspan(0.36, 0.52, color='darkorange', hatch='//', alpha=0.3, label='Def(F14)')
		ax.axhspan(-1.69, -1.21, color='r', alpha=0.2, label='Sub(B)')
		ax.axhspan(-1.52, -0.68, color='b', hatch='//', alpha=0.2, label='Sub(S18)')

	if solar:
		ax.axhline(0, color='gold', linestyle='solid')

	# Now loop over all data files
	for i in range(len(filenames)):

		file = filenames[i]
		name 	= np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')
		data = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=[5,6,8,9,10])
		feh 	= data[:,0]
		feherr 	= data[:,1]
		mnh 	= data[:,2]
		mnherr 	= data[:,3]
		redchisq = data[:,4]

		# Compute [Mn/Fe]
		mnfe = mnh - feh
		mnfeerr = np.sqrt(np.power(feherr,2.)+np.power(mnherr,2.))

		# Remove points with error > maxerror
		if maxerror is not None:
			mask 	= np.where((mnfeerr < maxerror))[0] # & (redchisq < 3.0))
			name 	= name[mask]
			feh 	= feh[mask]
			feherr  = feherr[mask]
			mnfe 	= mnfe[mask]
			mnfeerr = mnfeerr[mask]

		# Testing: Label some stuff
		outlier = np.where(mnfe > 1)[0]
		#print(name[outlier])

		if averages:
			# Make data points transparent
			ax.errorbar(feh, mnfe, color=cwheel[i], marker='.', alpha=0.9, markersize=8, linestyle='', capsize=0, zorder=99)

			# Compute binned averages
			binwidth = 0.2
			bins = np.arange(-2.7, -0.4, binwidth)
			bin_indices = np.digitize(feh, bins+(0.5*binwidth))

			# Boolean to check if legend handle needs to be created
			newlegend = True

			# Loop over all bins
			for idx in range(len(bins)):
				bin_match = np.where(bin_indices == idx)[0]
				mnfe_matched = mnfe[bin_match]
				mnfeerr_matched = mnfeerr[bin_match]

				# Check if there's anything in bin
				if len(bin_match) > 0:
					weights = 1./np.power(mnfeerr_matched,2.)
					binerror = np.sqrt(np.sum(np.power(mnfeerr_matched,2.)))/len(mnfeerr_matched)

					# Create new legend handle if it hasn't already been created
					if newlegend:
						ax.errorbar(bins[idx], np.average(mnfe_matched, weights=weights), color=cwheel[i], markersize=8, xerr=0.5*binwidth, yerr=binerror, marker='o', linestyle='', capsize=0, zorder=100, label=gratings[i]+': N='+str(len(feh)))
						newlegend = False
					else:
						ax.errorbar(bins[idx], np.average(mnfe_matched, weights=weights), color=cwheel[i], markersize=8, xerr=0.5*binwidth, yerr=binerror, marker='o', linestyle='', capsize=0, zorder=100)

		else:
			ax.errorbar(feh, mnfe, yerr=mnfeerr, xerr=feherr, alpha=0.9, marker='o', markersize=8, elinewidth=1, mew=0.1, linestyle='', capsize=0, zorder=99, label=gratings[i]+': N='+str(len(feh)))

	# Format plot
	ax.set_title(title, fontsize=18)
	ax.set_xlabel('[Fe/H]', fontsize=16)
	ax.set_ylabel('[Mn/Fe]', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)

	#ax.set_xlim([-3,-0.75])
	#ax.set_ylim([-2,2])
	ax.legend(loc='best', fontsize=12)

	# Output file
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

	return

def main():
	# GC checks
	#gc_mnfe_feh(['data/7089l1_1200B_final.csv','data/7078l1_1200B_final.csv','data/n5024b_1200B_final.csv'], 'figures/gc_checks/GCs_mnfe_feh.pdf', gratingnames=['M2', 'M15', 'M53'], maxerror=0.3, membercheck=True, solar=True)
	#plot_hist(['data/7089l1_1200B_final.csv','data/7078l1_1200B_final.csv','data/n5024b_1200B_final.csv'], ['M2: '+r'$\,\,\,\sigma_{\mathrm{sys}}=0.19$','M15: '+r'$\sigma_{\mathrm{sys}}=0.06$','M53: '+r'$\sigma_{\mathrm{sys}}=0.05$'], 'error', 'figures/gc_checks/errorhist.pdf', membercheck=['M2','M15','M53'], memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, sigmasys=[0.20,0.06,0.05])
	#plot_hist(['data/7089l1_1200B_final.csv','data/7078l1_1200B_final.csv','data/n5024b_1200B_final.csv'], ['M2','M15','M53'], 'error', 'figures/errorhist.png', membercheck=['M2','M15','M53'], memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, sigmasys=[0.20,0.06,0.05])

	# Test: radius check of M2 member stars
	#plot_hist(['data/7089l1_1200B_final.csv'], ['M2'], 'radius', 'figures/gc_checks/radiushist.pdf', membercheck=['M2'], memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3)

	# Hi-res comparison
	#plot_hires_comparison(['data/hires_data_final/GCs/M2_matched.csv', 'data/hires_data_final/GCs/M15_matched.csv', 'data/hires_data_final/north12_matched_total.csv', 'data/hires_data_final/shetrone03_matched_total.csv', 'data/hires_data_final/frebel10_matched.csv'], objnames=['M2', 'M15', 'Scl/For', 'For/LeoI', 'UMaII'], glob=[True, True, False, False, False], offset=[0.,5.43-5.39,5.43-5.39,(5.43-5.39)-(7.50-7.52), 0.])

	# dSph plots without chem evolution model
	#plot_mn_fe(['data/bscl5_1200B_final3.csv','data/LeoIb_1200B_final3.csv','data/bfor7_1200B_final3.csv','data/CVnIa_1200B_final3.csv','data/UMaIIb_1200B_final3.csv','data/bumia_1200B_final3.csv'], 'figures/other_dsphs.pdf', None, gratings=['Sculptor','Leo I','Fornax','Canes Venatici I','Ursa Major II','Ursa Minor'], maxerror=0.3, snr=None, solar=False, typeii=False, typei=False, n=6)
	#plot_mn_fe(['data/bscl5_1200B_final3.csv','data/CVnIa_1200B_final3.csv','data/UMaIIb_1200B_final3.csv','data/bumia_1200B_final3.csv'], 'figures/other_dsphs_test.pdf', None, gratings=['Sculptor','Canes Venatici I','Ursa Major II','Ursa Minor'], maxerror=0.3, snr=None, solar=False, typeii=False, typei=False)
	#plot_mn_fe(['data/bscl5_1200B_final3.csv','data/LeoIb_1200B_final3.csv','data/bfor7_1200B_final3.csv'], 'figures/other_dsphs_zoom.pdf', None, gratings=['Sculptor','Leo I','Fornax'], maxerror=0.2, snr=None, solar=False, typeii=False, typei=False, averages=True)


if __name__ == "__main__":
	main()