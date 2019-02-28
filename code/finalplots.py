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
import pandas
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
	fig, ax = plt.subplots(figsize=(10,5))

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

		if gratings is not None:
			color = gratings[0][i]
			marker = gratings[1][i]

		print('markers: ', marker)

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
		if solar:
			ax.axhline(0, color='k', linestyle=':')

		# Make plot
		ax.errorbar(feh, mnfe, yerr=mnfeerr, xerr=feherr, color=color, marker=marker, linestyle='', capsize=3, label=gratingnames[i]+' (N='+str(len(feh))+')')
		#ax.text(0.025, 0.9, 'N = '+str(len(name)), transform=ax.transAxes, fontsize=14)

	# Format plot
	if title is not None:
		ax.set_title(title, fontsize=18)
	ax.set_xlabel('[Fe/H]', fontsize=16)
	ax.set_ylabel('[Mn/Fe]', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)

	#ax.set_xlim([-3,-0.75])
	#ax.set_ylim([-2,2])
	plt.legend(loc='best', fontsize=14)

	# Output file
	plt.savefig(outfile, bbox_inches='tight')
	plt.show()

	return

def main():
	gc_mnfe_feh(['data/7089_1200B_final.csv','data/7078l1_1200B_final.csv'], 'figures/GCs_mnfe_feh.png', gratings=[['orange','b'],['^','o']], gratingnames=['M2', 'M15'], maxerror=0.3, membercheck=True, solar=True)

if __name__ == "__main__":
	main()