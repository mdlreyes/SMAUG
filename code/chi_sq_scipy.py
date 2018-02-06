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
from scipy.optimize import leastsq

def residual(mn, data, eps_data, temp, logg, fe, alpha):
	"""Compute residual.

    Inputs:
    params   -- input parameters [temp, logg, fe, alpha, mn]
    data 	 -- observed spectrum
    eps_data -- measurement uncertainty on observed spectrum

    Outputs:
    residual -- residual (weighted by measurement uncertainties)
    """

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

    # Define parameters to vary
    params = [mn]

	# Do minimization
	fitparams, cov = leastsq(residual, params, args=(data, eps_data, temp, logg, fe, alpha))

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