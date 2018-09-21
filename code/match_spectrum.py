# match_spectrum.py
# Opens observed spectra files; smooths and interpolates synthetic 
# spectrum to match observed spectrum
#
# - open_obs_file: opens observed spectrum
# - smooth_gauss_wrapper: Python wrapper for Fortran code that 
#	    smooths and interpolates synth spectrum to match observed
# 
# Created 9 Feb 18
# Updated 22 June 18
###################################################################

import os
import sys
import numpy as np
import math
from astropy.io import fits
from smooth_gauss import smooth_gauss
import matplotlib.pyplot as plt

def open_obs_file(filename, retrievespec=None, specparams=False, objname=None, coords=False):
	"""Open .fits.gz files with observed spectra.

    Inputs:
    filename - name of file to open

    Keywords:
    retrievespec - if None (default), output number of stars in file; 
                    else, retrieve spectrum of nth star from the file (where n = retrievespec)

    specparams	 - if True, output temp, logg, fe, alpha of nth star in file 
    				(only works when retrievespec is not None)

    objname 	 - if not None, finds parameters for spectrum of objname
    				(only works when specparams=True)

    coords 		 - if True, output the coordinates of the stars in file

    Outputs:
    wvl  - rest wavelength array for nth star
    flux - flux array for nth star
    ivar - inverse variance array for nth star
    """

	print('Opening ', filename)
	hdu1 = fits.open(filename)
	data = hdu1[1].data

	if retrievespec is not None:

		# If don't need to return other parameters, just return spectrum
		if not specparams:

			namearray = data['OBJNAME']
			wavearray = data['LAMBDA']
			fluxarray = data['SPEC']
			ivararray = data['IVAR']
			dlamarray = data['DLAM']

			# Get spectrum of a single star
			name = namearray[retrievespec]
			wvl  = wavearray[retrievespec]
			flux = fluxarray[retrievespec] 
			ivar = ivararray[retrievespec]
			dlam = dlamarray[retrievespec]

			# Correct for wavelength
			zrest = data['ZREST'][retrievespec]
			wvl = wvl / (1. + zrest)
			print('Redshift: ', zrest)

			return name, wvl, flux, ivar, dlam, zrest

		# Else, return other parameters
		else:

			# If necessary, match object name from other output file
			if objname is not None:
				# Get index of entry that matches object name of spectrum
				namearray = data['OBJNAME']
				index 	  = np.where(namearray==objname)

				# Check that such an entry exists
				if len(index[0]) > 0:
					print(index[0])

					temp 	= int(data['TEFF'][index[0]])
					logg 	= data['LOGG'][index[0]]
					fe 		= data['FEH'][index[0]]
					alpha 	= data['ALPHAFE'][index[0]]

					fe_err  = data['FEHERR'][index[0]]
					#zrest = data['ZREST'][index[0]]

					print('Parameters: ', temp, logg, fe, alpha)
					#print('Redshift: ', zrest)

				# If not, then missing best-fit parameters; just end the program
				else:
					raise ValueError('Spectrum not properly reduced! No best-fit parameters')

			else:
				temp 	= int(data['TEFF'][retrievespec])
				logg 	= data['LOGG'][retrievespec]
				fe 		= data['FEH'][retrievespec]
				alpha 	= data['ALPHAFE'][retrievespec]

				fe_err 	= data['FEHERR'][retrievespec]
				#zrest = data['ZREST'][retrievespec]

				print('Parameters: ', temp, logg, fe, alpha)

			return temp, logg, fe, alpha, fe_err #, zrest

	# Else, return number of stars in file (if coords=False) or coordinates of stars in file (if coords = True)
	else:

		if coords:
			RA = data['RA']
			dec = data['DEC']

			return RA, dec

		else:
			wavearray = data['LAMBDA']
			return len(wavearray)

def smooth_gauss_wrapper(lambda1, spec1, lambda2, dlam_in):
	"""
	Written by I. Escala

	A wrapper around the Fortran routine smooth_gauss.f, which
	interpolates the synthetic spectrum onto the wavelength array of the
	observed spectrum, while smoothing it to the specified resolution of the
	observed spectrum.
	Adapted into Python from IDL (E. Kirby)

	Parameters
	----------
	lambda1: array-like: synthetic spectrum wavelength array
	spec1: array-like: synthetic spectrum normalized flux values
	lambda2: array-like: observed wavelength array
	dlam_in: float, or array-like: full-width half max resolution in Angstroms
	 		to smooth the synthetic spectrum to, or the FWHM as a function of wavelength
	 	 	  
	Returns
	-------
	spec2: array-like: smoothed and interpolated synthetic spectrum, matching observations
	"""

	if not isinstance(lambda1, np.ndarray): lambda1 = np.array(lambda1)
	if not isinstance(lambda2, np.ndarray): lambda2 = np.array(lambda2)
	if not isinstance(spec1, np.ndarray): spec1 = np.array(spec1)
	
	#Make sure the synthetic spectrum is within the range specified by the
	#observed wavelength array!!!

	n2 = lambda2.size; n1 = lambda1.size
	
	# def findex(u, v):
	# 	"""
	# 	Return the index, for each point in the synthetic wavelength array, that corresponds
	#     to the bin it belongs to in the observed spectrum
	# 	e.g., lambda1[i-1] <= lambda2 < lambda1[i] if lambda1 is monotonically increasing
	#     The minus one changes it such that lambda[i] <= lambda2 < lambda[i+1] for i = 0,n2-2
	# 	in accordance with IDL
	# 	"""
	# 	result = np.digitize(u, v)-1 # Gives bin i for each lambda1[i] such that lambda2[i] <= lambda1 < lambda2[i+1]
	# 	w = [int((v[i] - u[result[i]])/(u[result[i]+1] - u[result[i]]) + result[i]) for i in range(n2)]
	# 	return np.array(w)
	
	#f = findex(lambda1, lambda2)
	f = np.digitize(lambda2, lambda1)
	f[-1] -= 1

	#print(f[0:20])
	#print(lambda1[0:20])
	#print(lambda2[0:20])

	#print(f)
	
	#Make it such that smooth_gauss.f takes an array corresponding to the resolution
	#each point of the synthetic spectrum will be smoothed to
	if isinstance(dlam_in, list) or isinstance(dlam_in, np.ndarray): dlam = dlam_in
	else: dlam = np.full(n2, dlam_in)
	dlam = np.array(dlam)
	
	dlambda1 = np.diff(lambda1)
	dlambda1 = dlambda1[dlambda1 > 0.]
	halfwindow = int(np.ceil(1.1*5.*dlam.max()/dlambda1.min()))
	
	#Python wrapped fortran implementation of smooth gauss
	spec2 = smooth_gauss(lambda1, spec1, lambda2, dlam, f, halfwindow)
	#temp = np.zeros(500); gauss = np.zeros(500)

	return spec2