# chi_sq.py
# Computes synthetic spectrum, then compares to observed spectrum
# and finds parameters that minimze chisq measure
# 
# Created 5 Feb 18
# Updated 20 June 18
###################################################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
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
import scipy.optimize
from wvl_corr import fit_wvl

# Observed spectrum
class obsSpectrum:

	def __init__(self, filename, starnum, wvlcorr, plot=False):

		# Observed star
		self.obsfilename = filename # File with observed spectra
		self.starnum = starnum	# Star number

		# Open observed spectrum
		self.specname, self.obswvl, self.obsflux, self.ivar, self.dlam, self.zrest = open_obs_file(self.obsfilename, retrievespec=self.starnum)

		# Get measured parameters from observed spectrum
		self.temp, self.logg, self.fe, self.alpha = open_obs_file('/raid/m31/dsph/scl/scl1/moogify7_flexteff.fits.gz', self.starnum, specparams=True, objname=self.specname)
		#self.temp, self.logg, self.fe, self.alpha, self.zrest = open_obs_file('/raid/m31/dsph/scl/scl1/moogify7_flexteff.fits.gz', self.starnum, specparams=True, objname=self.specname)

		# Correct observed spectrum for redshift
		#self.obswvl = self.obswvl/(1. + self.zrest)

		if plot:
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
			plt.axvspan(4335, 4345, alpha=0.5, color='red')
			plt.axvspan(4856, 4866, alpha=0.5, color='red')
			plt.axvspan(6558, 6568, alpha=0.5, color='red')
			plt.savefig('/raid/madlr/dsph/scl/moogify/'+self.specname+'_obs.png')
			plt.close()

		# Get synthetic spectrum, split both obs and synth spectra into red and blue parts
		synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs_for_division(self.obswvl, self.obsflux, self.ivar, temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, dlam=self.dlam)

		# Compute continuum-normalized observed spectrum
		self.obsflux_norm, self.ivar_norm = divide_spec(synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask)

		if plot:
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
			plt.axvspan(4335, 4345, alpha=0.5, color='red')
			plt.axvspan(4856, 4866, alpha=0.5, color='red')
			plt.axvspan(6558, 6568, alpha=0.5, color='red')
			plt.ylim((0,5))
			plt.xlim((6553,6573))
			plt.savefig('/raid/madlr/dsph/scl/moogify/'+self.specname+'_obsnormalized.png')
			plt.close()

		if wvlcorr:
			# Compute standard deviation
			contdivstd = np.zeros(len(self.ivar_norm))+np.inf
			contdivstd[self.ivar_norm > 0] = np.sqrt(np.reciprocal(self.ivar_norm[self.ivar_norm > 0]))

			# Wavelength correction
			self.obswvl_corr = fit_wvl(self.obswvl, self.obsflux_norm, contdivstd, self.dlam, 
				self.temp, self.logg, self.fe, self.alpha, self.specname, '/raid/madlr/dsph/scl/moogify/')

		# Crop observed spectrum into regions around Mn lines
		self.obsflux_fit, self.obswvl_fit, self.ivar_fit, self.dlam_fit, self.skip = mask_obs_for_abundance(self.obswvl, self.obsflux_norm, self.ivar_norm, self.dlam)

		# Splice together Mn line regions of observed spectra
		self.obsflux_final = np.hstack((self.obsflux_fit[self.skip]))
		self.obswvl_final = np.hstack((self.obswvl_fit[self.skip]))
		self.ivar_final = np.hstack((self.ivar_fit[self.skip]))
		self.dlam_final = np.hstack((self.dlam_fit[self.skip]))

		#print('Skip: ', self.skip)
		#print(len(self.obswvl_final))

	# Define function to minimize
	def synthetic(self, obswvl, mn, full=True):
		"""Get synthetic spectrum for fitting.

		Inputs:
		obswvl  -- independent variable (wavelength)
		mn 		-- parameter to fit (Mn abundance)

		Keywords:
		full 	-- if True, splice together all Mn line regions; else, keep as array

		Outputs:
		synthflux -- array-like, output synthetic spectrum
		"""

		# Compute synthetic spectrum
		print('Computing synthetic spectrum...')
		synth = runMoog(temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, elements=[25], abunds=[mn], solar=[5.43])

		# Loop over each line
		synthflux = []
		#a = [a0, a1, a2, a3, a4, a5]
		for i in self.skip:

			synthregion = synth[i]

			# Smooth each region of synthetic spectrum to match each region of continuum-normalized observed spectrum
			newsynth = get_synth(self.obswvl_fit[i], self.obsflux_fit[i], self.ivar_fit[i], self.dlam_fit[i], synth=synthregion)
			synthflux.append(newsynth)

		# If necessary, splice everything together
		if full:
			synthflux = np.hstack(synthflux[:])

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
		best_mn, covar = scipy.optimize.curve_fit(self.synthetic, self.obswvl_final, self.obsflux_final, p0=[mn0], sigma=np.sqrt(np.reciprocal(self.ivar_final)), epsfcn=0.01)
		error = np.sqrt(np.diag(covar))

		print('Answer: ', best_mn)
		print('Error: ', error)

		# Do some checks
		finalsynth = self.synthetic(self.obswvl_final, best_mn, full=False)
		plt.figure(figsize=(12,8))
		for i in range(len(finalsynth)):

			# Index for self arrays
			idx = self.skip[i]

			# Compute reduced chi-squared
			chisq = np.sum(np.power(self.obsflux_fit[idx] - finalsynth[i], 2.) * self.ivar_fit[idx]) / (len(self.obsflux_fit[idx]) - 1.)
			#print('Reduced chisq = ', chisq)

			# Plot for testing
			plotindex = 230+idx+1
			plt.subplot(plotindex)
			plt.errorbar(self.obswvl_fit[idx], self.obsflux_fit[idx], yerr=np.power(self.ivar_fit[idx],-0.5), color='k', fmt='o', label='Observed')
			plt.plot(self.obswvl_fit[idx], finalsynth[i], 'r--', label='Synthetic')
			#plt.legend(loc='best')

		plt.savefig('/raid/madlr/dsph/scl/moogify/'+self.specname+'_finalfits.png')
		plt.show()

		return best_mn, error

	def plot_chisq(self, mn0):
		"""Plot chi-sq as a function of [Mn/H].

		Inputs:
		mn0 -- initial guess for Mn abundance

		Outputs:
		fitparams -- best fit parameters
		rchisq	  -- reduced chi-squared
		"""

		mn_result, mn_error = self.minimize_scipy(mn0)

		mn_list = np.array([-3,-2,-1.5,-1,-0.5,-0.1,0,0.1,0.5,1,1.5,2,3])*mn_error + mn_result
		chisq_list = np.zeros(len(mn_list))
		for i in range(len(mn_list)):
			finalsynth = self.synthetic(self.obswvl_final, mn_list[i])
			chisq = np.sum(np.power(self.obsflux_final - finalsynth, 2.) * self.ivar_final) / (len(self.obsflux_final) - 1.)
			chisq_list[i] = chisq

		plt.figure()
		#plt.title(r'T = 4345K, log(g) = 0.83, [Fe/H] = -1.59, [$\alpha$/Fe] = 0.06')
		plt.plot(mn_list, chisq_list, '-o')
		plt.ylabel(r'$\chi^{2}_{red}$', fontsize=24)
		plt.xlabel('[Mn/H]', fontsize=24)
		plt.savefig('redchisq.png')
		plt.close()

		return

def main():
	filename = '/raid/caltech/moogify/bscl1/moogify.fits.gz'
	test = obsSpectrum(filename, 57, True, plot=False).minimize_scipy(-2.1661300692266998)
	#print('we done')
	#test = obsSpectrum(filename, 57).plot_chisq(-2.1661300692266998)

if __name__ == "__main__":
	main()