# chi_sq.py
# Computes synthetic spectrum, then compares to observed spectrum
# and finds parameters that minimze chisq measure
#
# Uses one of the following packages:
# 1) lmfit package
#   - residual_lmfit: computes residual (obs-synth) spectrum
#   - minimize_lmfit: minimizes residual using Levenberg-Marquardt (default)
# 2) scipy.optimize
#   - residual_scipy: computes residual (obs-synth) spectrum
#   - minimize_scipy: minimizes residual using Levenberg-Marquardt
# 
# Created 5 Feb 18
# Updated 10 Apr 18
###################################################################

import os
import numpy as np
import math
from run_moog import runMoog
from match_spectrum import open_obs_file
from continuum_div import get_synth, divide_spec, mask_obs_for_abundance
import subprocess
from astropy.io import fits
import pandas
#import lmfit
from scipy.optimize import least_squares

def residual_lmfit(obsfilename, starnum, params):
	"""Compute residual for lmfit.

    Inputs:
    For observed spectrum --
    obsfilename -- filename of observed spectrum
    starnum     -- nth star from the file (where n = starnum)

	For synthetic spectrum --
    params   -- input parameters [temp, logg, fe, alpha, mn]

    Outputs:
    residual -- residual (weighted by measurement uncertainties)
    """

	# Parameters
	obsfilename = params['obsfilename']
	starnum 	= params['starnum']
	temp	= params['temp']
	logg	= params['logg']
	fe 		= params['fe']
	alpha	= params['alpha']
	mn 		= params['mn']

	# Compute synthetic spectrum
	synth = runMoog(temp=temp, logg=logg, fe=fe, alpha=alpha, elements=[25], abunds=[mn], solar=[5.43])

	# Get observed spectrum and smooth the synthetic spectrum to match it
	synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs(obsfilename, starnum, synth=synth, mnlines=True)

	# Calculate residual
	if ivar is None:
		return (obsfluxmask.compressed() - synthfluxmask.compressed())
	else:
		return (obsfluxmask.compressed() - synthfluxmask.compressed())/np.sqrt(ivarmask.compressed())

def minimize_lmfit(obsfilename, starnum, temp, logg, fe, alpha, mn, method='leastsq'):
	"""Minimize residual using lmfit Levenberg-Marquardt.

    Inputs:
    For observed spectrum --
    obsfilename -- filename of observed spectrum
    starnum     -- nth star from the file (where n = starnum)

    For synthetic spectrum --
    temp 	 -- effective temperature (K)
    logg 	 -- surface gravity
    fe 		 -- [Fe/H]
    alpha 	 -- [alpha/Fe]
    mn 		 -- [Mn/H] abundance

	Keywords:
	method 	 -- method for minimization (default = 'leastsq'; see lmfit documentation for more options)

    Outputs:
    fitparams -- best fit parameters
    rchisq	  -- reduced chi-squared
    """

    # Get measured parameters from observed spectrum
	temp, temperr, logg, loggerr, fe, feerr, alpha, alphaerr = open_obs_file(obsfilename, starnum, specparams=True)

	# Define parameters
	params = lmfit.Parameters()
	params.add('obsfilename', value = obsfilename, vary=False)
	params.add('starnum', value = starnum, vary=False)
	params.add('temp', value = temp, vary=False)
	params.add('logg', value = logg, vary=False)
	params.add('fe', value = fe, vary=False)
	params.add('alpha', value = alpha, vary=False)
	params.add('mn', value = mn, vary=True)

	# Do minimization
	mini = lmfit.Minimizer(residual_lmfit, params, method)
	out  = mini.minimize()

	# Outputs
	fitparams = out.params 	# best-fit parameters
	rchisq  = out.redchi 	# reduced chi square
	cints 	= lmfit.conf_interval(mini,out) 	# confidence intervals

	return fitparams, rchisq

# Observed spectrum
class obsSpectrum(filename, starnum):

	def __init__(self):

		# Observed star
		self.obsfilename = filename # File with observed spectra
		self.starnum = starnum	# Star number

		# Open observed spectrum
		self.obswvl, self.obsflux, self.ivar = open_obs_file(self.obsfilename, retrievespec=self.starnum)

		# Get measured parameters from observed spectrum
		self.temp, self.temperr, self.logg, self.loggerr, self.fe, self.feerr, self.alpha, self.alphaerr = open_obs_file(obsfilename, starnum, specparams=True)

		# Compute continuum_normalized observed spectrum
		self.obsflux_norm, self.ivar_norm = divide_spec(self.obswvl, self.obsflux, self.ivar, temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha)

		# Mask out parts of the observed spectrum
		self.obsflux_fit, self.obswvl_fit, self.ivar_fit = mask_obs_for_abundance(self.obswvl, self.obsflux_norm, self.ivar_norm)

	def synthetic(self, mn):
		"""Get synthetic spectrum for fitting.

		Inputs:
		mn -- argument to vary (Mn abundance)

	    Outputs:
	    synthflux -- array-like, 
	    """

		# Compute synthetic spectrum
		print('Computing synthetic spectrum...')
		synth = runMoog(temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, elements=[25], abunds=[mn], solar=[5.43])

		# Smooth synthetic spectrum to match continuum-normalized observed spectrum
		print('Smoothing to match observed spectrum...')
		synthflux = get_synth(self.obswvl_fit, self.obsflux_fit, self.ivar_fit, synth=synth)

		return synthflux

	def minimize_scipy(self, mn0):
		"""Minimize residual using scipy.optimize Levenberg-Marquardt.

	    Inputs:
	    mn0 -- initial guess for Mn abundance

	    Outputs:
	    fitparams -- best fit parameters
	    rchisq	  -- reduced chi-squared
	    """

		# Do minimization
		print('Starting minimization!')
		best_mn, covar = curve_fit(self.synthetic, self.obswvl_fit, self.obsflux_fit, p0=mn0, sigma=self.ivar_fit**(-0.5))

		print('Made it here')

		print('Answer: ', best_mn)

		# Compute reduced chi-squared
		#finalresid = residual_scipy(result.x, obsfilename, starnum, temp, logg, fe, alpha)
		#rchisq = np.sum(np.power(finalresid,2.))/(len(finalresid) - 1.)
		
		# Compute standard error
		#error = []
		#for i in range(len(params)):
		#	try:
		#		error.append( np.absolute((cov[i][i] * rchisq)**2.) )
		#	except:
		#		error.append( 0.0 )

		return best_mn

filename = '/raid/caltech/moogify/bscl1/moogify.fits.gz'
test = obsSpectrum(filename, 3).minimize_scipy(0.)