# finalplots.py
# Make plots for paper
#
# Created 27 Feb 19
# Updated 27 Feb 19
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
		if i != 2:
			n, bins, _ = ax.hist(x, bins, alpha=0.5, label=labels[i], facecolor=colors[i], hatch=hatches[i], edgecolor=edges[i], fill=True)
		else: # no label
			n, bins, _ = ax.hist(x, bins, alpha=0.5, facecolor=colors[i], hatch=hatches[i], edgecolor=edges[i], fill=True)


		if quantity == 'error':
			# Overplot edges of histogram
			if i == 2:
				ax.hist(x, bins, label=labels[i], facecolor='None', edgecolor=edges[i], linewidth=1.5, fill=True)

			#Get bin width from this
			binwidth = bins[1] - bins[0]

			# Overplot best-fit Gaussian
			xbins = np.linspace(-3,3,1000)
			sigma = 1.
			mu = np.mean(x)
			y = ((1. / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1. / sigma * (xbins - mu))**2))*len(x)*binwidth

			plt.plot(xbins, y, linestyle=styles[i], color=edges[i], linewidth=2)

	# Format plot
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(18)
	plt.xlabel(xlabel, fontsize=20)
	plt.ylabel('N', fontsize=20)
	plt.legend(loc='best', fontsize=16)
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

	return

def main():
	#gc_mnfe_feh(['data/7089l1_1200B_final.csv','data/7078l1_1200B_final.csv','data/n5024b_1200B_final.csv'], 'figures/gc_checks/GCs_mnfe_feh.pdf', gratingnames=['M2', 'M15', 'M53'], maxerror=0.3, membercheck=True, solar=True)
	#plot_hist(['data/7089l1_1200B_final.csv','data/7078l1_1200B_final.csv','data/n5024b_1200B_final.csv'], ['M2: '+r'$\,\,\,\sigma_{\mathrm{sys}}=0.19$','M15: '+r'$\sigma_{\mathrm{sys}}=0.06$','M53: '+r'$\sigma_{\mathrm{sys}}=0.05$'], 'error', 'figures/gc_checks/errorhist.pdf', membercheck=['M2','M15','M53'], memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, sigmasys=[0.20,0.06,0.05])
	plot_hist(['data/7089l1_1200B_final.csv','data/7078l1_1200B_final.csv','data/n5024b_1200B_final.csv'], ['M2','M15','M53'], 'error', 'figures/gc_checks/errorhist.pdf', membercheck=['M2','M15','M53'], memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3, sigmasys=[0.20,0.06,0.05])

	#plot_hist(['data/7089l1_1200B_final.csv'], ['M2'], 'radius', 'figures/gc_checks/radiushist.pdf', membercheck=['M2'], memberlist='data/gc_checks/table_catalog.dat', maxerror=0.3)

if __name__ == "__main__":
	main()