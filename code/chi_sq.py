# chi_sq.py
# Computes synthetic spectrum, then compares to observed spectrum
# and finds parameters that minimze chisq measure
#
# - residual: computes residual (obs-synth) spectrum
# - minimize: minimizes residual using Levenberg-Marquardt (default)
# 
# Created 5 Feb 18
# Updated 5 Feb 18
###################################################################

import os
import numpy as np
import math
from interp_atmosphere import *
from run_moog import *
import subprocess
import pandas
import lmfit

def residual(params, data, eps_data):
	"""Compute residual.

    Inputs:
    params   -- input parameters [temp, logg, fe, alpha, mn]
    data 	 -- observed spectrum
    eps_data -- measurement uncertainty on observed spectrum

    Outputs:
    residual -- residual (weighted by measurement uncertainties)
    """

    # Parameters
    temp	= params['temp']
    logg	= params['logg']
    fe 		= params['fe']
    alpha	= params['alpha']
    mn 		= params['mn']

	# Compute synthetic spectrum
	synth = runMoog(temp=temp, logg=logg, fe=fe, alpha=alpha, elements=[25], abunds=[mn], solar=[5.43])

	# Calculate residual
	if eps_data is None:
		return (data - spectrum)
	else:
		return (data - spectrum)/eps_data

def minimize(data, eps_data=None, temp, logg, fe, alpha, mn, vary=[False, False, False, False, True], method='leastsq'):
	"""Minimize residual using Levenberg-Marquardt.

    Inputs:
    data 	 -- observed spectrum
    eps_data -- measurement uncertainty on observed spectrum
    temp 	 -- effective temperature (K)
    logg 	 -- surface gravity
    fe 		 -- [Fe/H]
    alpha 	 -- [alpha/Fe]
    mn 		 -- [Mn/H] abundance

	Keywords:
	vary 	 -- Boolean array describing whether to vary each parameter (True) or hold fixed (False)
	method 	 -- method for minimization (default = 'leastsq'; see lmfit documentation for more options)

    Outputs:
    params -- chi-squared (weighted by measurement uncertainties)
    """

	# Define parameters
	params = lmfit.Parameters()
	params.add('temp', value = temp, vary=vary[0])
	params.add('logg', value = logg, vary=vary[1])
	params.add('fe', value = fe, vary=vary[2])
	params.add('alpha', value = alpha, vary=vary[3])
	params.add('mn', value = mn, vary=vary[4])

	# Do minimization
	mini = lmfit.Minimizer(residual, params, method)
	out  = mini.minimize()

	# Outputs
	fitparams = out.params 	# best-fit parameters
	rchisq  = out.redchi 	# reduced chi square
	cints 	= lmfit.conf_interval(mini,out) 	# confidence intervals

	return fitparams, rchisq