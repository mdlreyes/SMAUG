# output_mn.py
# Gets Mn abundances for a list of stars
# 
# Created 5 June 18
# Updated 5 June 18
###################################################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import math
from run_moog import runMoog
from match_spectrum import open_obs_file
from continuum_div import get_synth, mask_obs_for_division, divide_spec, mask_obs_for_abundance
import subprocess
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas
import scipy.optimize
import chi_sq

def run_chisq(filename):

	# Get number of stars in file
	Nstars = open_obs_file(filename)

	# Run chi-sq fitting for all stars in file
	#for i in range(Nstars):

	pass


def match_hires(hiresfile, obsfile):

	# Output filename
	outputname = hiresfile[:-4]+'_matched.csv'
	
	# Get RA and Dec of stars from (lit) hi-res datafile
	hiresRA = np.genfromtxt(hiresfile, skip_header=1, delimiter='\t', usecols=2, dtype='str')
	hiresDec = np.genfromtxt(hiresfile, skip_header=1, delimiter='\t', usecols=3, dtype='str')

	hiresabundance = np.genfromtxt(hiresfile, skip_header=1, delimiter='\t', usecols=8, dtype='str')

	# Get RA and Dec of stars from (our) med-res file
	medresRA, medresDec = open_obs_file(obsfile, coords=True)
	medrescatalog = SkyCoord(ra=medresRA*u.degree, dec=medresDec*u.degree) 

	# Initialize arrays to hold stuff
	medresMn 	  = np.zeros(len(hiresRA))
	medresMnerror = np.zeros(len(hiresRA))
	hiresMn 	  = np.zeros(len(hiresRA))
	hiresMnerror  = np.zeros(len(hiresRA))

	# Loop over each star from hires list
	medres_starnum = np.zeros(len(hiresRA), dtype='int')
	for i in range(len(hiresRA)):

		# Convert coordinates to decimal form
		coord = SkyCoord(hiresRA[i], hiresDec[i], unit=(u.hourangle, u.deg))

		# Search for matching star in our catalog 
		idx, sep, _ = coord.match_to_catalog_sky(medrescatalog) 
		print('Separation: ', sep.arcsecond)
		print('ID: ', idx)

		if sep.arcsec < 10:
			print('Got one!')
		
			# Get metallicity of this matching star to use for initial guess
			temp, logg, fe, alpha = open_obs_file(obsfile, retrievespec=idx, specparams=True)

			# Measure med-res Mn abundance of the star
			medresMn[i], medresMnerror[i] = chi_sq.obsSpectrum(obsfile, idx).minimize_scipy(fe)

			# Get hi-res Mn abundance
			hiresMn[i] = float(hiresabundance[i][:5])
			hiresMnerror[i] = float(hiresabundance[i][-4:])

			np.savetxt(outputname, np.asarray((medresMn, medresMnerror, hiresMn, hiresMnerror, sep.arcsec)).T, delimiter='\t', header='our [Mn/H]\terror(our [Mn/H])\thires [Mn/H]\terror(hires [Mn/H])\tseparation (arcsec)')

	return medresMn, medresMnerror, hiresMn, hiresMnerror, sep.arcsec

def main():
	medresMn, medresMnerror, hiresMn, hiresMnerror, sep.arcsec = match_hires('Sculptor_hires.tsv','/raid/caltech/moogify/bscl1/moogify.fits.gz')

if __name__ == "__main__":
	main()