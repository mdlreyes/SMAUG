# get_hires_data.py
# Format high-resolution data into files that can be read by other
# codes (plot_mn.py, fit_mnfe_feh.py)
#
# Created 29 Nov 18
# Updated 29 Nov 18
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

def north_etal_12(coordfile, datafile1, datafile2, outfile):
	"""Get data from North+12 paper

	Inputs:
	coordfile 	-- table of coordinates
	datafile1 	-- table with abundances for lines 5407, 5420
	datafile1 	-- table with abundances for lines 5432, 5516
	outfile 	-- name of output file

	Keywords:
	"""

	# Open files
	coorddata = pd.read_fwf(coordfile, colspecs=[(0,6),(7,17),(18,30)])
	N = len(coorddata['Name'])

	data1 	= pd.read_csv(datafile1, delimiter='\t', na_values = ' ')

	data2 	= pd.read_csv(datafile2, delimiter='\t', na_values = ' ')

	# Prep the output data file
	outputname = 'data/Sculptor_hires_data/'+outfile
	with open(outputname, 'w+') as f:
		f.write('Name\tRA\tDec\tTemp\tlog(g)\t[Fe/H]\terror([Fe/H])\t[alpha/Fe]\t[Mn/H]\terror([Mn/H])\tchisq(reduced)\n')

	# Loop over all stars listed in coordinate file
	for i in range(N):

		starname = coorddata['Name'][i][:2] + coorddata['Name'][i][3:]
		name1 = data1['Star']
		name2 = data2['Star']

		# Check to see if star has values in datafiles
		if (starname in name1.values) and (starname in name2.values):

			idx1 = np.where((name1.values == starname))[0]
			idx2 = np.where((name2.values == starname))[0]

			# Check to see if star has [Mn/Fe] for all the requisite lines
			if (pd.isnull(data1['[Mn/Fe]5407'].values[idx1][0]) == False) and (pd.isnull(data1['[Mn/Fe]5420'].values[idx1][0]) == False) and (pd.isnull(data2['[Mn/Fe]5516'].values[idx2][0]) == False):
			
				# Get [Mn/H], [Mn/H]error for each line, [Fe/H]
				mnfe5407 = pd.to_numeric(data1['[Mn/Fe]5407'])[idx1].values[0]
				mnfe5420 = pd.to_numeric(data1['[Mn/Fe]5420'])[idx1].values[0]
				mnfe5516 = pd.to_numeric(data2['[Mn/Fe]5516'])[idx2].values[0]

				mnh5407 = pd.to_numeric(data1['[Mn/H]5407'])[idx1].values[0]
				mnh5420 = pd.to_numeric(data1['[Mn/H]5420'])[idx1].values[0]
				mnh5516 = pd.to_numeric(data2['[Mn/H]5516'])[idx2].values[0]

				mnherr5407 = pd.to_numeric(data1['[Mn/Fe]error5407'])[idx1].values[0]
				mnherr5420 = pd.to_numeric(data1['[Mn/Fe]error5420'])[idx1].values[0]
				mnherr5516 = pd.to_numeric(data2['[Mn/Fe]error5516'])[idx2].values[0]

				feh = mnh5407 - mnfe5407

				# Average all of the [Mn/H] to get a final abundance
				mnh = np.average([mnh5407, mnh5420, mnh5516], weights=[1./(mnherr5407**2.),1./(mnherr5420**2.),1./(mnherr5516**2.)])
				mnherr = np.sqrt(mnherr5407**2. + mnherr5420**2. + mnherr5516**2.)

				# Get coordinates
				starra = coorddata['RA'][i]
				stardec = coorddata['Dec'][i]
				coord = SkyCoord(starra+' '+stardec, frame='icrs', unit=(u.hourangle, u.deg))

				# Put all other info together
				RA 		= coord.ra.degree
				Dec 	= coord.dec.degree
				temp 	= 0.
				logg	= 0.
				feherr 	= 0.
				alphafe = 0.
				chisq 	= 0.

				# Write star data to file
				with open(outputname, 'a') as f:
					f.write(starname+'\t'+str(RA)+'\t'+str(Dec)+'\t'+str(temp)+'\t'+str(logg)+'\t'+str(feh)+'\t'+str(feherr)+'\t'+str(alphafe)+'\t'+str(mnh)+'\t'+str(mnherr)+'\t'+str(chisq)+'\n')


		else:
			continue

	return

def main():
	north_etal_12('data/Sculptor_hires_data/scl_sample.coord','data/Sculptor_hires_data/Sculptor_north_tab1.tsv','data/Sculptor_hires_data/Sculptor_north_tab2.tsv','north12_final.csv')

if __name__ == "__main__":
	main()