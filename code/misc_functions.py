# misc_functions.py
# Miscellaneous functions:
# 	- put_feherr: function to put [Fe/H] errors into data files that don't have them
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

		# Get file where [Fe/H] error is stored
		f = fits.open(feh_filelist[num])
		fehids = f[1].data['OBJNAME'].astype('str')
		feherr = f[1].data['FEHERR']

		# Match IDs to corret [Fe/H] error
		idx = np.array([item in names for item in fehids])

		# Update dataframe
		data['error([Fe/H])'] = np.sqrt(np.power(feherr[idx],2.) + 0.10103081**2.)

		data.to_csv(filelist[num], sep='\t', index=False)

	return

def main():
	put_feherr(filelist = ['data/7078l1_1200B_final.csv', 'data/7089l1_1200B_final.csv', 'data/n5024b_1200B_final.csv'], feh_filelist = ['data/glob_data/feherr_data/7078l1_flexteff.fits.gz','data/glob_data/feherr_data/7089l1_flexteff.fits.gz','data/glob_data/feherr_data/n5024_flexteff.fits.gz'])

if __name__ == "__main__":
	main()