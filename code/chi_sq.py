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
import pandas
#import lmfit
from scipy.optimize import curve_fit

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
class obsSpectrum():

	def __init__(self, filename, starnum):

		# Observed star
		self.obsfilename = filename # File with observed spectra
		self.starnum = starnum	# Star number

		# Open observed spectrum
		self.specname, self.obswvl, self.obsflux, self.ivar, self.dlam = open_obs_file(self.obsfilename, retrievespec=self.starnum)

		'''
		# Plot first Mn line region of observed spectrum
		testmask = np.where((self.obswvl > 4749) & (self.obswvl < 4759))
		plt.figure()
		plt.plot(self.obswvl[testmask], self.obsflux[testmask], 'k-') #, yerr=np.power(self.ivar[testmask], 0.5))
		plt.savefig('obs_singleline.png')
		'''

		# Get measured parameters from observed spectrum
		self.temp, self.logg, self.fe, self.alpha = open_obs_file('/raid/m31/dsph/scl/scl1/moogify7_flexteff.fits.gz', self.starnum, specparams=True, objname=self.specname)

		# Plot observed spectrum
		plt.figure()
		plt.plot(self.obswvl, self.obsflux, 'k-')
		plt.axvspan(4749, 4759, alpha=0.5, color='blue')
		plt.axvspan(4778, 4788, alpha=0.5, color='blue')
		plt.axvspan(4818, 4828, alpha=0.5, color='blue')
		plt.axvspan(5389, 5399, alpha=0.5, color='blue')
		plt.axvspan(5532, 5542, alpha=0.5, color='blue')
		plt.axvspan(6008, 6018, alpha=0.5, color='blue')
		plt.axvspan(6016, 6026, alpha=0.5, color='blue')
		plt.xlim((4856, 4866))
		plt.savefig('obs.png')

		# Get synthetic spectrum, split both obs and synth spectra into red and blue parts
		synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs_for_division(self.obswvl, self.obsflux, self.ivar, temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, dlam=self.dlam)

		# Compute continuum-normalized observed spectrum
		self.obsflux_norm, self.ivar_norm = divide_spec(synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask)

		# Plot continuum-normalized observed spectrum
		plt.figure()
		plt.plot(self.obswvl, self.obsflux_norm, 'k-')
		plt.axvspan(4749, 4759, alpha=0.5, color='blue')
		plt.axvspan(4778, 4788, alpha=0.5, color='blue')
		plt.axvspan(4818, 4828, alpha=0.5, color='blue')
		plt.axvspan(5389, 5399, alpha=0.5, color='blue')
		plt.axvspan(5532, 5542, alpha=0.5, color='blue')
		plt.axvspan(6008, 6018, alpha=0.5, color='blue')
		plt.axvspan(6016, 6026, alpha=0.5, color='blue')
		#plt.xlim((4856, 4866))
		plt.xlim((4300, 6100))
		plt.ylim((0,1.5))
		plt.savefig('obs_normalized.png')

		# Crop observed spectrum into regions around Mn lines
		self.obsflux_fit, self.obswvl_fit, self.ivar_fit, self.dlam_fit = mask_obs_for_abundance(self.obswvl, self.obsflux_norm, self.ivar_norm, self.dlam)

		# Splice together Mn line regions of observed spectra
		self.obsflux_final = np.hstack((self.obsflux_fit[:]))
		self.obswvl_final = np.hstack((self.obswvl_fit[:]))
		self.ivar_final = np.hstack((self.ivar_fit[:]))

	def synthetic(self, obswvl, mn):
		"""Get synthetic spectrum for fitting.

		Inputs:
		obswvl -- independent variable (wavelength)
		mn -- parameter to fit (Mn abundance)

	    Outputs:
	    synthflux -- array-like, 
	    """

		# Compute synthetic spectrum
		print('Computing synthetic spectrum...')
		synth = runMoog(temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, elements=[25], abunds=[mn], solar=[5.43])

		# Smooth synthetic spectrum to match continuum-normalized observed spectrum
		#print('Smoothing to match observed spectrum...')

		# Loop over all lines
		for i in range(len(synth)):

			# For testing purposes

			# Smooth each region of synthetic spectrum to match each region of observed spectrum
			synthflux = get_synth(self.obswvl_fit[i], self.obsflux_fit[i], self.ivar_fit[i], self.dlam_fit[i], synth=synth[i])
			#print(i, synthflux)

			plt.figure()
			plt.plot(self.obswvl_fit[i], self.obsflux_fit[i], 'k-', label='Observed')
			plt.plot(self.obswvl_fit[i], synthflux, 'r--', label='Synthetic')
			plt.legend(loc='best')
			plt.savefig('final_obs_'+str(i)+'.png')
			plt.close()

		sys.exit()

		'''
		# Smooth each region of synthetic spectrum to match each region of observed spectrum
		if i == 0:
			synthflux = get_synth(self.obswvl_fit[i], self.obsflux_fit[i], self.ivar_fit[i], self.dlam_fit[i], synth=synth[i])
		else:
			# Splice synthflux together
			synthflux = np.hstack((synthflux, get_synth(self.obswvl_fit[i], self.obsflux_fit[i], self.ivar_fit[i], self.dlam_fit[i], synth=synth[i])))
		'''

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
		params = [mn0]
		best_mn, covar = curve_fit(self.synthetic, self.obswvl_final, self.obsflux_final, p0=params, sigma=np.power(self.ivar_final,-0.5), absolute_sigma=True, method='lm')

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

def main():
	filename = '/raid/caltech/moogify/bscl1/moogify.fits.gz'
	#test = obsSpectrum(filename, 65).minimize_scipy(0.)
	#test = obsSpectrum(filename, 28).minimize_scipy(0.)
	#test = obsSpectrum(filename, 30).minimize_scipy(0.)
	#test = obsSpectrum(filename, 32).minimize_scipy(0.)
	test = obsSpectrum(filename, 39).minimize_scipy(-0.5)

if __name__ == "__main__":
	main()