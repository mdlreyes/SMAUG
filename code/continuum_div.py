# continuum_div.py
# - masks parts of observed spectra (mask_obs);
# - obtains synthetic spectrum from Ivanna's grid (get_synth); 
# - divides obs/synth, fits spline, and divides obs/spline (divide_spec)
# 
# Created 22 Feb 18
# Updated 10 Apr 18
###################################################################

import os
import sys
import numpy as np
import numpy.ma as ma
import matplotlib

np.set_printoptions(threshold=np.inf)

#Backend for python3 on mahler
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import math
import gzip
from astropy.io import fits
from smooth_gauss import smooth_gauss
from interp_atmosphere import interpolateAtm
from match_spectrum import open_obs_file, smooth_gauss_wrapper
from scipy.interpolate import splrep, splev

def get_synth(obsfilename, starnum, synth=None, temp=None, logg=None, fe=None, alpha=None):
	"""Get synthetic spectrum and smooth it to match observed spectrum.

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
    synth -- if None, get synthetic spectrum from Ivanna's grid;
    		 else, use synth (should be a list of arrays [synthflux, synthwvl])

    Outputs:
    synthflux -- synthetic flux array
    obsflux   -- observed flux array
    obswvl    -- wavelength array
    ivar 	  -- inverse variance array
    """

	# Get synthetic spectrum
	if synth is None:

		# Use modified version of interpolateAtm to get synthetic spectrum from Ivanna's grid
		synthflux = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/raid/gridie/bin/')
		wvl_range = np.arange(4100., 6300.+0.14, 0.14)
		synthwvl  = 0.5*(wvl_range[1:] + wvl_range[:-1])

	else:
		synthflux = synth[0]
		synthwvl  = synth[1]

	# Open observed spectrum
	obswvl, obsflux, ivar = open_obs_file(obsfilename, retrievespec=starnum)

	# Interpolate and smooth the synthetic spectrum onto the observed wavelength array
	synthfluxnew = smooth_gauss_wrapper(synthwvl, synthflux, obswvl, 1.1)

	# For testing purposes
	'''
	plt.figure()
	plt.subplot(211)
	plt.plot(synthwvl, synthflux, 'r-', label='Original')
	plt.plot(obswvl, synthfluxnew, 'b-', label='Smoothed')
	plt.legend()
	plt.title('Synthetic spectrum')
	
	plt.subplot(212)
	plt.plot(obswvl, obsflux)
	plt.title('Observed spectrum')

	plt.show()
	'''

	return synthfluxnew, obsflux, obswvl, ivar

def mask_obs(obsfilename, starnum, synth=None, mnlines=False, temp=None, logg=None, fe=None, alpha=None):
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
    synth   -- if None, get synthetic spectrum from Ivanna's grid;
    		 else, use synth (should be a list of arrays [synthflux, synthwvl])
	mnlines -- if True, mask out everything BUT Mn lines (for abundance measurement);
			   else, mask out Mn lines (for continuum division) -- default

    Outputs:
    synthfluxmask -- (masked!) synthetic flux array
    obsfluxmask   -- (masked!) observed flux array
    obswvlmask    -- (masked!) wavelength array
    mask 	  -- mask to avoid bad shit (chip gaps, bad pixels, Na D lines)
    """

    # Mask out bad stuff + Mn lines (for divide_spec)
	if not mnlines:

		# Get smoothed synthetic spectrum and (NOT continuum-normalized) observed spectrum
		synthflux, obsflux, obswvl, ivar = get_synth(obsfilename, starnum, synth=None, temp=temp, logg=logg, fe=fe, alpha=alpha)

		# Make a mask
		mask = np.zeros(len(synthflux), dtype=bool)

		# Mask out first and last five pixels
		mask[:5]  = True
		mask[-5:] = True

		# Mask out pixels near chip gap
		chipgap = int(len(mask)/2 - 1)
		mask[(chipgap - 5): (chipgap + 5)] = True

		# Mask out any bad pixels
		mask[np.where(synthflux <= 0.)] = True
		mask[np.where(ivar <= 0.)] = True

		# Mask out pixels around Na D doublet (5890, 5896 A)
		mask[np.where((obswvl > 5884.) & (obswvl < 5904.))] = True

		# Mask out pixels in regions around Mn lines (+/- 5A) 
		mnmask = np.zeros(len(synthflux), dtype=bool)
		mnmask[np.where((obswvl > 4749.) & (obswvl < 4759.))] = True
		mnmask[np.where((obswvl > 4778.) & (obswvl < 4788.))] = True
		mnmask[np.where((obswvl > 4818.) & (obswvl < 4828.))] = True
		mnmask[np.where((obswvl > 5389.) & (obswvl < 5399.))] = True
		mnmask[np.where((obswvl > 5532.) & (obswvl < 5542.))] = True
		mnmask[np.where((obswvl > 6008.) & (obswvl < 6018.))] = True
		mnmask[np.where((obswvl > 6016.) & (obswvl < 6026.))] = True
		mask[mnmask] = True

		# Create masked arrays
		synthfluxmask = ma.masked_array(synthflux, mask)
		obsfluxmask   = ma.masked_array(obsflux, mask)
		obswvlmask	  = ma.masked_array(obswvl, mask)
		ivarmask	  = ma.masked_array(ivar, mask)

		return synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask

	# Mask out bad stuff + EVERYTHING BUT Mn lines (for actual abundance measurements)
	else:

		# Get smoothed synthetic spectrum and continuum-normalized observed spectrum
		synthflux, obsflux_norm, obswvl, ivar_norm = divide_spec(obsfilename, starnum, synth=synth, temp=temp, logg=logg, fe=fe, alpha=alpha)

		# Make a mask
		mask = np.zeros(len(synthflux), dtype=bool)

		# Mask out first and last five pixels
		mask[:5]  = True
		mask[-5:] = True

		# Mask out pixels near chip gap
		chipgap = int(len(mask)/2 - 1)
		mask[(chipgap - 5): (chipgap + 5)] = True

		# Mask out any bad pixels
		mask[np.where(synthflux <= 0.)] = True
		mask[np.where(ivar_norm <= 0.)] = True

		# Mask out pixels around Na D doublet (5890, 5896 A)
		mask[np.where((obswvl > 5884.) & (obswvl < 5904.))] = True

		# Mask out everything EXCEPT Mn lines
		mnmask = np.zeros(len(synthflux), dtype=bool) # Mask with all Mn lines masked out
		mnmask[np.where((obswvl > 4749.) & (obswvl < 4759.))] = True
		mnmask[np.where((obswvl > 4778.) & (obswvl < 4788.))] = True
		mnmask[np.where((obswvl > 4818.) & (obswvl < 4828.))] = True
		mnmask[np.where((obswvl > 5389.) & (obswvl < 5399.))] = True
		mnmask[np.where((obswvl > 5532.) & (obswvl < 5542.))] = True
		mnmask[np.where((obswvl > 6008.) & (obswvl < 6018.))] = True
		mnmask[np.where((obswvl > 6016.) & (obswvl < 6026.))] = True

		mask[~mnmask] = True

		# Create masked arrays
		synthfluxmask = ma.masked_array(synthflux, mask)
		obsfluxmask   = ma.masked_array(obsflux_norm, mask)
		obswvlmask	  = ma.masked_array(obswvl, mask)
		ivarmask	  = ma.masked_array(ivar_norm, mask)

		return synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask

def divide_spec(obsfilename, starnum, synth=None, temp=None, logg=None, fe=None, alpha=None):
	"""Do the actual continuum fitting:
	- Divide obs/synth.
	- Fit spline to quotient. 
		- Use cubic B-spline representation with breakpoints spaced every 150 Angstroms (Kirby+09)
		- Do iteratively, so that pixels that deviate from fit by more than 5sigma are removed for next iteration
		- Don't worry about telluric absorption corrections?
	- Divide obs/spline.

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
    synth -- if None, get synthetic spectrum from Ivanna's grid;
    		 else, use synth (should be a list of arrays [synthflux, synthwvl])

    Outputs:
    synthflux    -- synthetic flux
    obsflux_norm -- continuum-normalized observed flux
    obswvl 		 -- observed wavelength
    ivar_norm    -- continuum-normalized inverse variance

    """
	synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs(obsfilename, starnum, synth=synth, temp=temp, logg=logg, fe=fe, alpha=alpha)

	# Mask out everything EXCEPT Mn lines
	mnmask = np.zeros(len(synthfluxmask.data), dtype=bool) # Mask with all Mn lines masked out
	mnmask[np.where((obswvlmask.data > 4749.) & (obswvlmask.data < 4759.))] = True
	mnmask[np.where((obswvlmask.data > 4778.) & (obswvlmask.data < 4788.))] = True
	mnmask[np.where((obswvlmask.data > 4818.) & (obswvlmask.data < 4828.))] = True
	mnmask[np.where((obswvlmask.data > 5389.) & (obswvlmask.data < 5399.))] = True
	mnmask[np.where((obswvlmask.data > 5532.) & (obswvlmask.data < 5542.))] = True
	mnmask[np.where((obswvlmask.data > 6008.) & (obswvlmask.data < 6018.))] = True
	mnmask[np.where((obswvlmask.data > 6016.) & (obswvlmask.data < 6026.))] = True

	# Convert inverse variance to inverse standard dev
	ivarmask = ma.masked_array(np.sqrt(ivarmask.data), mask)

	# Divide obs/synth
	quotient = obsfluxmask/synthfluxmask

	# First check if there are enough points to compute continuum
	if len(synthfluxmask.compressed()) < 300:
		print('Insufficient number of pixels to determine the continuum!')
		return

	# Compute breakpoints for B-spline 
	def calc_breakpoints(array, interval):
		"""
		Helper function for use with a B-spline.
		Computes breakpoints for an array given an interval.
		"""

		breakpoints = []
		counter = 0
		for i in range(len(array)):

			if (array[i] - array[counter]) >= interval:
				counter = i
				breakpoints.append(array[i])

		return breakpoints
	
	# Determine initial spline fit, before sigma-clipping
	breakpoints_old	= calc_breakpoints(obswvlmask.compressed(), 150.) # Use 150 A spacing
	print('breakpoints: ', breakpoints_old)
	#print(obswvlmask.compressed(), quotient.compressed(), ivarmask.compressed())
	splinerep_old 	= splrep(obswvlmask.compressed(), quotient.compressed(), w=ivarmask.compressed(), t=breakpoints_old)
	continuum_old	= splev(obswvlmask.compressed(), splinerep_old)

	# Iterate the fit, sigma-clipping until it converges or max number of iterations is reached
	iternum  = 0
	maxiter  = 10
	clipmask = np.ones(len(obswvlmask.compressed()), dtype=bool)

	while iternum < maxiter:

		# Compute residual between quotient and spline
		resid = quotient.compressed() - continuum_old
		sigma = np.std(resid)

		# Sigma-clipping
		clipmask[np.where((resid < -0.3*sigma) | (resid > 5*sigma))] = False

		# Recalculate the fit after sigma-clipping
		breakpoints_new = calc_breakpoints((obswvlmask.compressed())[clipmask], 150.)
		splinerep_new 	= splrep((obswvlmask.compressed())[clipmask], (quotient.compressed())[clipmask], w=(ivarmask.compressed())[clipmask], t=breakpoints_new)
		continuum_new 	= splev(obswvlmask.compressed(), splinerep_new)

		# For testing purposes
		'''
		print('Iteration ', iternum)
		print((obswvlmask.compressed()[clipmask]).size)

		plt.figure()

		plt.subplot(211)
		plt.title('Iteration '+str(iternum))
		plt.plot(obswvlmask, quotient, 'b.')
		plt.plot(obswvlmask[~clipmask], quotient[~clipmask], 'ko')
		plt.plot(obswvlmask.compressed(), continuum_new, 'r-')

		plt.subplot(212)
		plt.plot(obswvlmask.compressed(), resid)
		plt.show()
		'''

		# Check for convergence (if all points have been clipped)
		if (obswvlmask.compressed()[clipmask]).size == 0:
			print('Continuum fit converged at iteration ', iternum)
			break 

		else:
			continuum_old = continuum_new
			iternum += 1

	# Compute final spline
	continuum_final = splev(obswvlmask.data, splinerep_new)

	# Now divide obs/spline
	obswvl 		 = obswvlmask.data
	obsflux_norm = obsfluxmask.data/continuum_final

	ivar_norm 	 = ivarmask.data * np.power(continuum_final, 2.)

	# Get synthetic spectrum too
	synthflux = synthfluxmask.data

	return synthflux, obsflux_norm, obswvl, ivar_norm

#synthflux, obsflux, _, ivar = get_synth('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, temp=3500, logg=3.0, fe=-3.3, alpha=1.2)
#synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, temp=3500, logg=3.0, fe=-3.3, alpha=1.2)
#synthflux, obsflux_norm, obswvl, ivar_norm = divide_spec('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, synth=None, temp=3500, logg=3.0, fe=-3.3, alpha=1.2)

#print(obswvl, obswvl[1]-obswvl[0], obswvl[2]-obswvl[1])
#np.set_printoptions(threshold=np.inf)
#print(synthflux, synthfluxmask)
#testarray = np.array([synthflux, obsflux, obswvl, obsflux_norm, ivar_norm])
#np.savetxt('test.txt.gz',testarray)