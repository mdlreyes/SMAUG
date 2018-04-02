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
import numpy.ma as ma
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
    synthflux -- synthetic flux array
    obsflux   -- observed flux array
    obswvl    -- wavelength array
    ivar 	  -- inverse variance array
    """

	# Use modified version of interpolateAtm to get synthetic spectrum
	synthflux = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/raid/gridie/bin/')
	wvl_range = np.arange(4100., 6300.+0.14, 0.14)
	synthwvl  = 0.5*(wvl_range[1:] + wvl_range[:-1])

	# For testing purposes
	#print(len(synthflux), len(synthwvl))
	#plt.plot(synthwvl, synthflux)
	#plt.show()

	# Open observed spectrum
	obswvl, obsflux, ivar = open_obs_file(obsfilename, retrievespec=starnum)

	# Interpolate and smooth the synthetic spectrum onto the observed wavelength array
	synthflux = smooth_gauss_wrapper(synthwvl, synthflux, obswvl, 1.1)

	return synthflux, obsflux, obswvl, ivar

def mask_obs(obsfilename, starnum, temp, logg, fe, alpha):
	"""Make a mask for synthetic and observed spectra.

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
    synthfluxmask -- (masked!) synthetic flux array
    obsfluxmask   -- (masked!) observed flux array
    obswvlmask    -- (masked!) wavelength array
    mask 	  -- mask to avoid bad shit (chip gaps, bad pixels, Na D lines)
    """

    synthflux, obsflux, obswvl, ivar = get_synth(obsfilename, starnum, temp, logg, fe, alpha)

	mask = np.zeros(len(synthflux), dtype=bool)

	# Mask out first and last five pixels
	mask[:5]  = True
	mask[-5:] = True

	# Mask out pixels near chip gap
	chipgap = int(len(mask)/2 - 1)
	mask[(chipgap - 5): (chipgap + 5)] = True

	# Mask out any bad pixels
	mask[np.where(synthflux <= 0.)] = True
	mask[np.where(ivar < 0.)] = True

	# Mask out pixels around Na D doublet (5890, 5896 A)
	mask[np.where((obswvl > 5884.) & (obswvl < 5904.))] = True

	# Mask out pixels in regions around Mn lines (+/- 5A) 
	mask[np.where((obswvl > 4749.) & (obswvl < 4759.))] = True
	mask[np.where((obswvl > 4778.) & (obswvl < 4788.))] = True
	mask[np.where((obswvl > 4818.) & (obswvl < 4828.))] = True
	mask[np.where((obswvl > 5389.) & (obswvl < 5399.))] = True
	mask[np.where((obswvl > 5532.) & (obswvl < 5542.))] = True
	mask[np.where((obswvl > 6008.) & (obswvl < 6018.))] = True
	mask[np.where((obswvl > 6016.) & (obswvl < 6026.))] = True

	synthfluxmask = ma.masked_array(synthflux, mask)
	obsfluxmask   = ma.masked_array(obsflux, mask)
	obswvlmask	  = ma.masked_array(obswvl, mask)

	return synthfluxmask, obsfluxmask, obswvlmask, mask

def divide_spec(obsfilename, starnum, temp, logg, fe, alpha):
	"""Do the actual continuum fitting:
	Divide obs/synth, fit spline to quotient, and divide obs/spline.

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

    """

    synthfluxmask, obsfluxmask, obswvlmask, mask = mask_obs(obsfilename, starnum, temp, logg, fe, alpha)

    quotient = np.divide(obsfluxmask, synthfluxmask)

    return

#get_synth('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, temp=3500, logg=3.0, fe=-3.3, alpha=1.2)
#synthflux, obsflux, obswvl = get_synth('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, temp=3500, logg=3.0, fe=-3.3, alpha=1.2)
#testarray = np.array([synthflux, obsflux, obswvl])
#np.savetxt('test.txt.gz',testarray)