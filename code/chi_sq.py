# chi_sq.py
# Computes synthetic spectrum, then compares to observed spectrum
# and finds parameters that minimze chisq measure
# 
# Created 5 Feb 18
# Updated 15 June 18
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
import scipy.optimize

# Observed spectrum
class obsSpectrum:

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
		#self.temp, self.logg, self.fe, self.alpha, self.zrest = open_obs_file('/raid/m31/dsph/scl/scl1/moogify7_flexteff.fits.gz', self.starnum, specparams=True, objname=self.specname)

		# Correct observed spectrum for redshift
		#self.obswvl = self.obswvl/(1. + self.zrest)

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
		plt.axvspan(4856, 4866, alpha=0.5, color='red')
		plt.xlim((4500, 5000))
		plt.savefig('obs.png')

		# Get synthetic spectrum, split both obs and synth spectra into red and blue parts
		synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs_for_division(self.obswvl, self.obsflux, self.ivar, temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, dlam=self.dlam)

		# Compute continuum-normalized observed spectrum
		self.obsflux_norm, self.ivar_norm = divide_spec(synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask)

		#print('Test', self.obsflux_norm, self.ivar_norm)

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
		plt.axvspan(4859, 4863, alpha=0.5, color='red')
		plt.xlim((4800, 4900))
		plt.ylim((0,1.5))
		plt.savefig('obs_normalized.png')

		# Crop observed spectrum into regions around Mn lines
		self.obsflux_fit, self.obswvl_fit, self.ivar_fit, self.dlam_fit, self.skip = mask_obs_for_abundance(self.obswvl, self.obsflux_norm, self.ivar_norm, self.dlam)

		# Splice together Mn line regions of observed spectra
		self.obsflux_final = np.hstack((self.obsflux_fit[self.skip]))
		self.obswvl_final = np.hstack((self.obswvl_fit[self.skip]))
		self.ivar_final = np.hstack((self.ivar_fit[self.skip]))
		self.dlam_final = np.hstack((self.dlam_fit[self.skip]))

		print('Skip: ', self.skip)
		print(len(self.obswvl_final))

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

		'''
		# Testing: smooth synthetic spectrum to match continuum-normalized observed spectrum
		i = 0
		synthflux = get_synth(self.obswvl_fit[i], self.obsflux_fit[i], self.ivar_fit[i], self.dlam_fit[i], synth=synth[i])

		chisq = np.sum(np.power(self.obsflux_fit[i] - synthflux, 2.) * self.ivar_fit[i]) / (len(self.obsflux_fit[i]) - 1.)
		print('Chisq = ', chisq)

		'''

		# Loop over each line
		synthflux = []
		for i in self.skip:

			# Smooth each region of synthetic spectrum to match each region of continuum-normalized observed spectrum
			newsynth = get_synth(self.obswvl_fit[i], self.obsflux_fit[i], self.ivar_fit[i], self.dlam_fit[i], synth=synth[i])
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

		# Testing
		'''		
		#synth0 = synthetic(self.obswvl_final, 0.0)
		synth1 = synthetic(self.obswvl_final, -1.91)
		synth2 = synthetic(self.obswvl_final, -2.04)
		synth3 = synthetic(self.obswvl_final, -10.0)
		synth4 = synthetic(self.obswvl_final, -2.17)

		i = 0
		plt.figure()
		#plt.title(r'T = 4345K, log(g) = 0.83, [Fe/H] = -1.59, [$\alpha$/Fe] = 0.06')
		plt.errorbar(self.obswvl_fit[i], self.obsflux_fit[i], yerr=np.power(self.ivar_fit[i],-0.5), color='k', fmt='o')
		#plt.plot(self.obswvl_fit[0], synth0, color='red', linestyle='-', label='[Mn/H]=0.0')
		plt.fill_between(self.obswvl_fit[i], synth1, synth4, color='red', alpha=0.25)
		plt.plot(self.obswvl_fit[i], synth2, color='red', linestyle='-', label=r'[Mn/H] = -2.04$\pm$0.13')
		plt.plot(self.obswvl_fit[i], synth3, color='blue', linestyle='--', label='No Mn')
		plt.axvspan(4753, 4755, facecolor='g', alpha=0.25)
		plt.axvspan(4761, 4764, facecolor='g', alpha=0.25)
		plt.axvspan(4765, 4768, facecolor='g', alpha=0.25)
		plt.ylabel('Relative flux', fontsize=18)
		plt.xlabel(r'$\lambda (\AA)$', fontsize=18)
		plt.xlim((4744,4772))
		plt.ylim((0.75,1.05))
		plt.legend(loc='best', fontsize=16)
		plt.savefig('final_obs.png', bbox_inches='tight')
		plt.close()

		chisq = np.sum(np.power(self.obsflux_fit[0] - synth2, 2.) * self.ivar_fit[0]) / (len(self.obsflux_fit[0]) - 1.)
		print('Chisq = ', chisq)
		'''

		# Do minimization
		print('Starting minimization!')
		best_mn, covar = scipy.optimize.curve_fit(self.synthetic, self.obswvl_final, self.obsflux_final, p0=[mn0], sigma=np.sqrt(np.reciprocal(self.ivar_final)), epsfcn=0.001)

		error = np.sqrt(np.diag(covar))

		print('Answer: ', best_mn)
		print('Error: ', error)

		# Compute reduced chi-squared
		'''
		finalsynth = self.synthetic(self.obswvl_final, best_mn, full=False)
		for i in range(len(finalsynth)):
			chisq = np.sum(np.power(self.obsflux_fit[i] - finalsynth[i], 2.) * self.ivar_fit[i]) / (len(self.obsflux_fit[i]) - 1.)
			print('Reduced chisq = ', chisq)

			# Plot for testing
			plt.figure()
			plt.errorbar(self.obswvl_fit[i], self.obsflux_fit[i], yerr=np.power(self.ivar_fit[i],-0.5), color='k', fmt='o', label='Observed')
			plt.plot(self.obswvl_fit[i], finalsynth[i], 'r--', label='Synthetic')
			plt.legend(loc='best')
			plt.savefig('final_obs_'+str(i)+'.png')
			plt.close()
		'''

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
	#test = obsSpectrum(filename, 57).minimize_scipy(-2.1661300692266998)
	test = obsSpectrum(filename, 57).plot_chisq(-2.1661300692266998)

if __name__ == "__main__":
	main()