# misc_functions.py
# Miscellaneous functions:
# 	- put_feherr: function to put [Fe/H] errors into data files that don't have them
# 	- plot_gaia: plot Gaia data from a file
#
# Created 15 July 2019
###################################################################

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import math
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from matplotlib.ticker import NullFormatter
from statsmodels.stats.weightstats import DescrStatsW

def put_feherr(filelist, feh_filelist):
	""" Put [Fe/H] errors into data files that don't already have them. """

	for num in range(len(filelist)):

		# Open file
		data = pd.read_csv(filelist[num], delimiter='\t')
		names = np.asarray(data['Name'], dtype='str')

		#for i in range(len(names)):
		#	names[i] = names[i][1:]

		# Get file where [Fe/H] error is stored
		f = fits.open(feh_filelist[num])
		fehids = f[1].data['OBJNAME'].astype('str')
		feherr = f[1].data['FEHERR']

		# Match IDs to corret [Fe/H] error
		idx = np.array([item in names for item in fehids])
		print(fehids[idx])
		#for i in range(len(names)):
		#	print(names[i], fehids[idx][i])

		# Update dataframe
		data['error([Fe/H])'] = np.sqrt(np.power(feherr[idx],2.) + 0.10103081**2.)

		data.to_csv(filelist[num], sep='\t', index=False)

	return

def put_vt(filelist, feh_filelist):
	""" Put microturbulent velocities into data files that don't already have them. """

	for num in range(len(filelist)):

		# Open file
		data = pd.read_csv(filelist[num], delimiter='\t')
		names = np.asarray(data['Name'], dtype='str')

		# Get file where [Fe/H] error is stored
		f = fits.open(feh_filelist[num])
		fehids = f[1].data['OBJNAME'].astype('str')
		feherr = f[1].data['VT']

		# Match IDs to corret [Fe/H] error
		idx = np.array([item in names for item in fehids])
		print(fehids[idx])
		#for i in range(len(names)):
		#	print(names[i], fehids[idx][i])

		# Update dataframe
		data['error([Fe/H])'] = np.sqrt(np.power(feherr[idx],2.) + 0.10103081**2.)

		data.to_csv(filelist[num], sep='\t', index=False)

	return

def plot_gaia(filename, outputname):
	""" Plot Gaia PMs from a tab-delimited file. """

	# Open file
	data = pd.read_csv(filename, delimiter='\t')

	# Plot data
	plt.errorbar(x=data['PMra(mas/y)'],y=data['PMdec(mas/y)'],xerr=data['err_PMra'],yerr=data['err_PMdec'], linestyle='None', marker='o', color='k')
	plt.ylabel('PM_Dec (mas/y)')
	plt.xlabel('PM_RA (mas/y)')
	plt.savefig(outputname, bbox_inches='tight')
	plt.show()

	# Find outliers
	outliers = np.where((data['PMra(mas/y)'] > 4.0) | (data['PMdec(mas/y)'] < -3.0) | (data['PMdec(mas/y)'] > -1.75))[0]
	print(data['Name'][outliers])

	return

def main():
	#put_feherr(filelist = ['data/7078l1_1200B_final.csv', 'data/7089l1_1200B_final.csv', 'data/n5024b_1200B_final.csv'], feh_filelist = ['data/glob_data/feherr_data/7078l1_flexteff.fits.gz','data/glob_data/feherr_data/7089l1_flexteff.fits.gz','data/glob_data/feherr_data/n5024_flexteff.fits.gz'])
	#plot_gaia('data/gc_checks/7089l1_parallaxes.csv', 'figures/gc_checks/7089l1_pmcheck.png')
	#put_feherr(filelist = ['data/bscl5_1200B_final3.csv', 'data/bfor7_1200B_final3.csv', 'data/LeoIb_1200B_final3.csv','data/CVnIa_1200B_final3.csv'], feh_filelist = ['data/dsph_data/feherr_data/bscl5_flexteff.fits.gz','data/dsph_data/feherr_data/bfor7_flexteff.fits.gz','data/dsph_data/feherr_data/LeoIb_flexteff.fits.gz','data/dsph_data/feherr_data/CVnIa_flexteff.fits.gz'])
	put_feherr(filelist = ['data/LeoIb_1200B_final3.csv'], feh_filelist = ['data/dsph_data/feherr_data/LeoIb_moogify.fits.gz'])

if __name__ == "__main__":
	main()