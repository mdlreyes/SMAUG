# continuum_div.py
# - masks parts of observed spectra (mask_obs);
# - obtains synthetic spectrum from Ivanna's grid (get_synth); 
# - divides obs/synth, fits spline, and divides obs/spline (divide_spec)
# 
# Created 22 Feb 18
# Updated 2 Apr 18
###################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import gzip
from astropy.io import fits
from smooth_gauss import smooth_gauss
from interp_atmosphere import interpolateAtm
from match_spectrum import open_obs_file, smooth_gauss_wrapper

def get_synth(obsfilename, starnum, temp, logg, fe, alpha):
	"""Get synthetic spectrum from Ivanna's grid, and smooth to match observed spectrum.

    Inputs:
    For observed spectrum --
    obsfilename -- filename of observed spectrum
    starnum     -- nth star from the file (where n = starnum)

	For synthetic spectrum -- 
    temp  -- effective temperature (K)
    logg  -- surface gravity
    fe 	  -- [Fe/H]
    alpha -- [alpha/Fe]

    Keywords:

    Outputs:
    synthflux -- synthetic spectrum (array of fluxes) for continuum division
    obsflux   -- observed spectrum (array of fluxes) for continuum division
    obswvl    -- array of wavelengths
    """

	# Use modified version of interpolateAtm to get synthetic spectrum
	synthflux = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/raid/gridie/bin/')
	wvl_range = np.arange(4100., 6300.+0.14, 0.14)
	synthwvl  = 0.5*(wvl_range[1:] + wvl_range[:-1])

	# For testing purposes
	print(len(synthflux), len(synthwvl))
	#plt.plot(synthwvl, synthflux)
	#plt.show()

	# Open observed spectrum
	obswvl, obsflux = open_obs_file(obsfilename, retrievespec=starnum)

	# Interpolate and smooth the synthetic spectrum onto the observed wavelength array
	synthflux = smooth_gauss_wrapper(synthwvl, synthflux, obswvl, 1.1)

	return synthflux, obsflux, obswvl

get_synth('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, temp=3500, logg=3.0, fe=-3.3, alpha=1.2)
#synthflux, obsflux, obswvl = get_synth('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, temp=3500, logg=3.0, fe=-3.3, alpha=1.2)
#print(len(synthflux), len(obsflux), len(obswvl))
#print(synthflux[0],synthflux[1000])
#print(obsflux[0],obsflux[1000])
#print(obswvl[0],obswvl[1000])