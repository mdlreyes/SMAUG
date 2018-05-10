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

np.set_printoptions(threshold=np.inf)

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import math
import gzip
from astropy.io import fits
from smooth_gauss import smooth_gauss
from interp_atmosphere import interpolateAtm
from match_spectrum import open_obs_file, smooth_gauss_wrapper
from scipy.interpolate import splrep, splev

def get_synth(obswvl, obsflux, ivar, dlam, synth=None, temp=None, logg=None, fe=None, alpha=None):
	"""Get synthetic spectrum and smooth it to match observed spectrum.

    Inputs:
    For observed spectrum --
    obswvl  -- wavelength array of observed spectrum
    obsflux -- flux array of observed spectrum
    ivar 	-- inverse variance array of observed spectrum
    dlam 	-- FWHM of observed spectrum

    Keywords:
    synth -- if None, get synthetic spectrum from Ivanna's grid;
    		 else, use synth (should be a list of arrays [synthflux, synthwvl])

		    For synthetic spectrum from Ivanna's grid -- 
		    temp  -- effective temperature (K)
		    logg  -- surface gravity
		    fe 	  -- [Fe/H]
		    alpha -- [alpha/Fe]

    Outputs:
    synthflux -- synthetic flux array
    obsflux   -- observed flux array
    obswvl    -- wavelength array
    ivar 	  -- inverse variance array
    """

	# Get synthetic spectrum from grid
	if synth is None:

		# Use modified version of interpolateAtm to get synthetic spectrum from Ivanna's grid
		synthflux = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/raid/gridie/bin/')
		wvl_range = np.arange(4100., 6300.+0.14, 0.14)
		synthwvl  = 0.5*(wvl_range[1:] + wvl_range[:-1])

	# Else, use input synthetic spectrum
	else:
		synthflux = synth[0]
		synthwvl  = synth[1]

	# Clip synthetic spectrum so it's within range of obs spectrum
	synthwvl = synthwvl[(np.where((synthwvl > obswvl[0]) & (synthwvl < obswvl[-1])))]
	synthflux = synthflux[(np.where((synthwvl > obswvl[0]) & (synthwvl < obswvl[-1])))]

	# Interpolate and smooth the synthetic spectrum onto the observed wavelength array
	synthfluxnew = smooth_gauss_wrapper(synthwvl, synthflux, obswvl, dlam)

	# For testing purposes
	'''
	plt.figure()
	plt.plot(synthwvl, synthflux, 'k-', label='Synthetic')
	plt.plot(obswvl, synthfluxnew, 'r-', label='Synthetic (smoothed)')
	#plt.xlim(5000, 5250)
	#plt.ylim(0.9,1.0)
	plt.legend(loc='best')
	plt.savefig('synth_cont.png')
	#plt.show()
	'''

	return synthfluxnew

def mask_obs_for_division(obswvl, obsflux, ivar, temp=None, logg=None, fe=None, alpha=None, dlam=None):
	"""Make a mask for synthetic and observed spectra.
	Mask out Mn lines for continuum division.
	Split spectra into red and blue parts.

    Inputs:
    For observed spectrum --
    obswvl  -- wavelength array of observed spectrum
    obsflux -- flux array of observed spectrum
    ivar 	-- inverse variance array of observed spectrum

    Keywords:
    For synthetic spectrum -- 
    temp  -- effective temperature (K)
    logg  -- surface gravity
    fe 	  -- [Fe/H]
    alpha -- [alpha/Fe]

    dlam -- FWHM of observed spectrum

    Outputs:
    synthfluxmask -- (masked!) synthetic flux array
    obsfluxmask   -- (masked!) observed flux array
    obswvlmask    -- (masked!) wavelength array
    ivarmask 	  -- (masked!) inverse variance array
    mask 	  -- mask to avoid bad shit (chip gaps, bad pixels, Na D lines)
    """

	# Get smoothed synthetic spectrum and (NOT continuum-normalized) observed spectrum
	synthflux = get_synth(obswvl, obsflux, ivar, dlam, synth=None, temp=temp, logg=logg, fe=fe, alpha=alpha)

	# Make a mask
	mask = np.zeros(len(synthflux), dtype=bool)

	# Mask out first and last five pixels
	mask[:5]  = True
	mask[-5:] = True

	# Mask out pixels near chip gap
	chipgap = int(len(mask)/2 - 1)
	print('wavelength of chip gap: ', obswvl[chipgap])
	mask[(chipgap - 5): (chipgap + 5)] = True

	# Mask out any bad pixels
	mask[np.where(synthflux <= 0.)] = True
	#print('Where synthflux < 0: ', synthflux[np.where(synthflux <=0.)])

	mask[np.where(ivar <= 0.)] = True
	#print('Where ivar < 0: ', obswvl[np.where(ivar <=0.)])

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
	synthfluxmask 	= ma.masked_array(synthflux, mask)
	obsfluxmask   	= ma.masked_array(obsflux, mask)
	obswvlmask	  	= ma.masked_array(obswvl, mask)
	ivarmask	  	= ma.masked_array(ivar, mask)

	# Split spectra into blue (index 0) and red (index 1) parts
	synthfluxmask 	= [synthfluxmask[:chipgap], synthfluxmask[chipgap:]]
	obsfluxmask		= [obsfluxmask[:chipgap], obsfluxmask[chipgap:]]
	obswvlmask 		= [obswvlmask[:chipgap], obswvlmask[chipgap:]]
	ivarmask 		= [ivarmask[:chipgap], ivarmask[chipgap:]]
	mask 			= [mask[:chipgap], mask[chipgap:]]

	return synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask

def divide_spec(synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, sigmaclip=False):
	"""Do the actual continuum fitting:
	- Divide obs/synth.
	- Fit spline to quotient. 
		- Use cubic B-spline representation with breakpoints spaced every 150 Angstroms (Kirby+09)
		- Do iteratively, so that pixels that deviate from fit by more than 5sigma are removed for next iteration
		- Don't worry about telluric absorption corrections?
	- Divide obs/spline.

	Do this for blue and red parts of spectra separately, then splice back together.

    Inputs:
    synthfluxmask 	-- smoothed synth spectrum (Mn lines masked out)
    obsfluxmask		-- obs spectrum (Mn lines masked out)
    obswvlmask		-- wavelength (Mn lines masked out)
    ivarmask		-- inverse variance array (Mn lines masked out)
    mask 			-- mask used to mask stuff (Mn lines, bad pixels) out

    Keywords:
    sigmaclip 		-- if 'True', do sigma clipping while spline-fitting

    Outputs:
    obsflux_norm_final -- continuum-normalized observed flux (blue and red parts spliced together)
    ivar_norm_final    -- continuum-normalized inverse variance (blue and red parts spliced together)
    """

	# Prep outputs of continuum division
	obswvl 		 = []
	obsflux_norm = []
	ivar_norm 	 = []

	# Do continuum division for blue and red parts separately
	for ipart in [0,1]:

		# Convert inverse variance to inverse standard deviation
		newivarmask = ma.masked_array(np.sqrt(ivarmask[ipart].data), mask[ipart])

		# Divide obs/synth
		quotient = obsfluxmask[ipart]/synthfluxmask[ipart]

		# First check if there are enough points to compute continuum
		if len(synthfluxmask[ipart].compressed()) < 300:
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
		breakpoints_old	= calc_breakpoints(obswvlmask[ipart].compressed(), 150.) # Use 150 A spacing
		#print('breakpoints: ', breakpoints_old)
		splinerep_old 	= splrep(obswvlmask[ipart].compressed(), quotient.compressed(), w=newivarmask.compressed(), t=breakpoints_old)
		continuum_old	= splev(obswvlmask[ipart].compressed(), splinerep_old)

		# Iterate the fit, sigma-clipping until it converges or max number of iterations is reached
		if sigmaclip:
			iternum  = 0
			maxiter  = 10
			clipmask = np.ones(len(obswvlmask[ipart].compressed()), dtype=bool)

			while iternum < maxiter:

				# Compute residual between quotient and spline
				resid = quotient.compressed() - continuum_old
				sigma = np.std(resid)

				# Sigma-clipping
				clipmask[np.where((resid < -5*sigma) | (resid > 5*sigma))] = False

				# Recalculate the fit after sigma-clipping
				breakpoints_new = calc_breakpoints((obswvlmask[ipart].compressed())[clipmask], 150.)
				splinerep_new 	= splrep((obswvlmask[ipart].compressed())[clipmask], (quotient.compressed())[clipmask], w=(newivarmask.compressed())[clipmask], t=breakpoints_new)
				continuum_new 	= splev(obswvlmask[ipart].compressed(), splinerep_new)

				# For testing purposes
				'''
				print('Iteration ', iternum)
				print((obswvlmask[ipart].compressed()[clipmask]).size)

				plt.figure()

				plt.subplot(211)
				plt.title('Iteration '+str(iternum))
				plt.plot(obswvlmask[ipart], quotient, 'b.')
				plt.plot(obswvlmask[ipart][~clipmask], quotient[~clipmask], 'ko')
				plt.plot(obswvlmask[ipart].compressed(), continuum_new, 'r-')

				plt.subplot(212)
				plt.plot(obswvlmask[ipart].compressed(), resid)
				plt.show()
				'''

				# Check for convergence (if all points have been clipped)
				if (obswvlmask[ipart].compressed()[clipmask]).size == 0:
					print('Continuum fit converged at iteration ', iternum)
					break 

				else:
					continuum_old = continuum_new
					iternum += 1

			# Compute final spline
			continuum_final = splev(obswvlmask[ipart].data, splinerep_new)

		# If no sigma clipping, just compute final spline from initial spline fit
		else:
			continuum_final = splev(obswvlmask[ipart].data, splinerep_old)

		# Now divide obs/spline
		obswvl.append(obswvlmask[ipart].data)
		obsflux_norm.append(obsfluxmask[ipart].data/continuum_final)

		# Compute final inverse variance
		ivar_norm.append(ivarmask[ipart].data * np.power(continuum_final, 2.))

		# Plots for testing
		'''
		plt.figure()
		plt.subplot(211)
		plt.plot(obswvlmask[ipart].data, obsfluxmask[ipart].data, 'r-', label='Masked')
		plt.plot(obswvlmask[ipart].compressed(), obsfluxmask[ipart].compressed(), 'k-', label='Observed')
		plt.axvspan(4749, 4759, alpha=0.5, color='blue')
		plt.axvspan(4778, 4788, alpha=0.5, color='blue')
		plt.axvspan(4818, 4828, alpha=0.5, color='blue')
		plt.axvspan(5389, 5399, alpha=0.5, color='blue')
		plt.axvspan(5532, 5542, alpha=0.5, color='blue')
		plt.axvspan(6008, 6018, alpha=0.5, color='blue')
		plt.axvspan(6016, 6026, alpha=0.5, color='blue')
		#plt.plot(obswvlmask[ipart][mask], obsfluxmask[ipart][mask], 'r-', label='Mask')
		plt.legend(loc='best')
		plt.subplot(212)
		plt.plot(obswvlmask[ipart].data, synthfluxmask[ipart].data, 'r-', label='Masked')
		plt.plot(obswvlmask[ipart].compressed(), synthfluxmask[ipart].compressed(), 'k-', label='Synthetic')
		plt.axvspan(4749, 4759, alpha=0.5, color='blue')
		plt.axvspan(4778, 4788, alpha=0.5, color='blue')
		plt.axvspan(4818, 4828, alpha=0.5, color='blue')
		plt.axvspan(5389, 5399, alpha=0.5, color='blue')
		plt.axvspan(5532, 5542, alpha=0.5, color='blue')
		plt.axvspan(6008, 6018, alpha=0.5, color='blue')
		plt.axvspan(6016, 6026, alpha=0.5, color='blue')
		plt.legend(loc='best')
		plt.savefig('maskedMnlines'+str(ipart)+'.png')

		print(len(obswvlmask[ipart].compressed()), len(continuum_final[~mask[ipart]]))

		plt.figure()
		plt.plot(obswvlmask[ipart].compressed(), quotient.compressed(), 'k-', label='Quotient (masked)')
		plt.plot(obswvlmask[ipart].compressed(), continuum_final[~mask[ipart]], 'r-', label='Final spline (masked)')
		plt.axvspan(4749, 4759, alpha=0.5, color='blue')
		plt.axvspan(4778, 4788, alpha=0.5, color='blue')
		plt.axvspan(4818, 4828, alpha=0.5, color='blue')
		plt.axvspan(5389, 5399, alpha=0.5, color='blue')
		plt.axvspan(5532, 5542, alpha=0.5, color='blue')
		plt.axvspan(6008, 6018, alpha=0.5, color='blue')
		plt.axvspan(6016, 6026, alpha=0.5, color='blue')
		plt.legend(loc='best')
		plt.savefig('quotient_maskedMnlines'+str(ipart)+'.png')
		'''

	# Now splice blue and red parts together
	obswvl_final	 	= np.hstack((obswvl[0], obswvl[1]))
	obsflux_norm_final 	= np.hstack((obsflux_norm[0], obsflux_norm[1]))
	ivar_norm_final 	= np.hstack((ivar_norm[0], ivar_norm[1]))

	return obsflux_norm_final, ivar_norm_final

def mask_obs_for_abundance(obswvl, obsflux_norm, ivar_norm, dlam):
	"""Make a mask for synthetic and observed spectra.
	Mask out bad stuff + EVERYTHING BUT Mn lines (for actual abundance measurements)

    Inputs:
    Observed spectrum --
    obswvl  	 -- wavelength array of (continuum-normalized!) observed spectrum
    obsflux_norm -- flux array of (continuum-normalized!) observed spectrum
    ivar_norm	 -- inverse variance array of (continuum-normalized!) observed spectrum
    dlam 		 -- FWHM array of (continuum-normalized!) observed spectrum

	Synthetic spectrum --
    synthwvl  -- wavelength array of synthetic spectrum
    synthflux -- flux array of synthetic spectrum

    Keywords:

    Outputs:
    obsfluxmask   -- (masked!) observed flux array
    obswvlmask    -- (masked!) wavelength array
    ivarmask 	  -- (masked!) inverse variance array
    dlammask	  -- (masked!) FWHM array
    """

	# Make a mask
	mask = np.zeros(len(obswvl), dtype=bool)

	# Mask out first and last five pixels
	mask[:5]  = True
	mask[-5:] = True

	# Mask out pixels near chip gap
	chipgap = int(len(mask)/2 - 1)
	mask[(chipgap - 5): (chipgap + 5)] = True

	# Mask out any bad pixels
	mask[np.where(ivar_norm <= 0.)] = True

	# Mask out pixels around Na D doublet (5890, 5896 A)
	mask[np.where((obswvl > 5884.) & (obswvl < 5904.))] = True

	# Mask out everything EXCEPT Mn lines
	obsfluxmask = []
	obswvlmask  = []
	ivarmask 	= []
	dlammask	= []

	masklist 	= [obsfluxmask, obswvlmask, ivarmask, dlammask]
	arraylist 	= [obsflux_norm, obswvl, ivar_norm, dlam]

	for i in range(len(masklist)):
		masklist[i].append( arraylist[i][np.where(((obswvl > 4749.) & (obswvl < 4759.) & (~mask)))] )
		masklist[i].append( arraylist[i][np.where(((obswvl > 4778.) & (obswvl < 4788.) & (~mask)))] )
		masklist[i].append( arraylist[i][np.where(((obswvl > 4818.) & (obswvl < 4828.) & (~mask)))] )
		masklist[i].append( arraylist[i][np.where(((obswvl > 5389.) & (obswvl < 5399.) & (~mask)))] )
		masklist[i].append( arraylist[i][np.where(((obswvl > 5532.) & (obswvl < 5542.) & (~mask)))] )
		masklist[i].append( arraylist[i][np.where(((obswvl > 6008.) & (obswvl < 6026.) & (~mask)))] )

	#mnmask[np.where((obswvl > 4749.) & (obswvl < 4759.))] = True
	#mnmask[np.where((obswvl > 4778.) & (obswvl < 4788.))] = True
	#mnmask[np.where((obswvl > 4818.) & (obswvl < 4828.))] = True
	#mnmask[np.where((obswvl > 5389.) & (obswvl < 5399.))] = True
	#mnmask[np.where((obswvl > 5532.) & (obswvl < 5542.))] = True
	#mnmask[np.where((obswvl > 6008.) & (obswvl < 6018.))] = True
	#mnmask[np.where((obswvl > 6016.) & (obswvl < 6026.))] = True
	#mask[~mnmask] = True

	# Create masked arrays
	#synthfluxmask = ma.masked_array(synthflux, mask)
	#obsfluxmask   = ma.masked_array(obsflux_norm, mask)
	#obswvlmask	  = ma.masked_array(obswvl, mask)
	#ivarmask	  = ma.masked_array(ivar_norm, mask)

	#obsfluxmask   = obsflux_norm[~mask]
	#obswvlmask	  = obswvl[~mask]
	#ivarmask	  = ivar_norm[~mask]

	return obsfluxmask, obswvlmask, ivarmask, dlammask

#synthflux, obsflux, _, ivar = get_synth('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, temp=3500, logg=3.0, fe=-3.3, alpha=1.2)
#synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, temp=3500, logg=3.0, fe=-3.3, alpha=1.2)
#synthflux, obsflux_norm, obswvl, ivar_norm = divide_spec('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, synth=None, temp=3500, logg=3.0, fe=-3.3, alpha=1.2)

#print(obswvl, obswvl[1]-obswvl[0], obswvl[2]-obswvl[1])
#np.set_printoptions(threshold=np.inf)
#print(synthflux, synthfluxmask)
#testarray = np.array([synthflux, obsflux, obswvl, obsflux_norm, ivar_norm])
#np.savetxt('test.txt.gz',testarray)