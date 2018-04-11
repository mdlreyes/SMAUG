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
from continuum_div import get_synth, mask_obs, divide_spec
import subprocess
from astropy.io import fits
import pandas
import lmfit
from scipy.optimize import leastsq

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

def residual_scipy(obsfilename, starnum, mn, temp, logg, fe, alpha):
	"""Compute residual for scipy.optimize.

    Inputs:
    For observed spectrum --
    obsfilename -- filename of observed spectrum
    starnum     -- nth star from the file (where n = starnum)

	For synthetic spectrum --
    [mn, temp, logg, fe, alpha] -- the usual parameters

    Outputs:
    residual -- residual (weighted by measurement uncertainties)
    obsflux  -- observed flux
    ivar 	 -- inverse variance
    """

	# Compute synthetic spectrum
	synth = runMoog(temp=temp, logg=logg, fe=fe, alpha=alpha, elements=[25], abunds=[mn], solar=[5.43])

	# Get observed spectrum and smooth the synthetic spectrum to match it
	synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs(obsfilename, starnum, synth=synth, mnlines=True)

	# Calculate residual
	if ivar is None:
		return (obsfluxmask.compressed() - synthfluxmask.compressed())
	else:
		return (obsfluxmask.compressed() - synthfluxmask.compressed())/np.sqrt(ivarmask.compressed())

def minimize_scipy(mn, obsfilename, starnum, temp, logg, fe, alpha):
	"""Minimize residual using scipy.optimize Levenberg-Marquardt.

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

    Outputs:
    fitparams -- best fit parameters
    rchisq	  -- reduced chi-squared
    """

    # Define parameters to vary
    params = [mn]

	# Do minimization
	fitparams, cov = leastsq(residual_scipy, params, args=(obsfilename, starnum, temp, logg, fe, alpha))

	# Compute reduced chi-squared
	rchisq = np.sum(np.power(residual(mn, data, eps_data, temp, logg, fe, alpha),2.))/(len(data) - 1.)
	
	# Compute standard error
	#error = []
	#for i in range(len(params)):
	#	try:
	#		error.append( np.absolute((cov[i][i] * rchisq)**2.) )
	#	except:
	#		error.append( 0.0 )

	return fitparams, rchisq

fitparams_lmfit, rchisq_lmfit = minimize_lmfit('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, temp=3500, logg=3.0, fe=-3.3, alpha=1.2, mn=0, method='leastsq')
fitparams_scipy, rchisq_scipy = minimize_scipy('/raid/caltech/moogify/bscl1/moogify.fits.gz', starnum=0, temp=3500, logg=3.0, fe=-3.3, alpha=1.2, mn=0)