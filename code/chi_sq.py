# chi_sq.py
# Computes synthetic spectrum, then compares to observed spectrum
# and finds parameters that minimze chisq measure
# 
# Created 5 Feb 18
# Updated 2 Nov 18
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
import csv

# Code to make plots
def make_plots(lines, specname, obswvl, obsflux, synthflux, outputname, ivar=None, title=None, synthfluxup=None, synthfluxdown=None):
	"""Make plots.

	Inputs:
	lines -- which linelist to use? Options: 'new', 'old'
	specname -- name of star
	obswvl 	-- observed wavelength array
	obsflux -- observed flux
	synthflux 	-- synthetic flux
	outputname 	-- where to output file

	Keywords:
	ivar 	-- inverse variance; if 'None' (default), don't print errorbars
	title 	-- plot title; if 'None' (default), then plot title = "Star + ID"
	synthfluxup & synthfluxdown -- if not 'None' (default), then plot synthetic spectrum as region between [Mn/H]_best +/- 0.3dex

	Outputs:
	"""

	# Define lines to plot
	if lines == 'new':
		linelist = np.array([4739.,4754.,4761.5,4765.5,4783.,4823.,5394.,5399.,
							 5407.,5420.,5432.,5516.,5537.,6013.,6016.,6021.,6384.,6491.])
		linewidth = np.array([1.,1.,1.5,1.5,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])

		nrows = 3
		ncols = 6
		figsize = (24,12)

	elif lines == 'old':
		linelist = np.array([4739.,4783.,4823.,5394.,5432.,5516.,5537.,6013.,6021.,6384.,6491.])
		linewidth = np.ones(len(linelist))

		nrows = 3
		ncols = 4
		figsize = (16,12)

	# Define title
	if title is None:
		title = 'Star'+specname

	# Plot showing fits
	plt.figure(num=1, figsize=figsize)
	plt.title(title)

	# Plot showing residuals
	plt.figure(num=2, figsize=figsize)
	plt.title(title)

	for i in range(len(linelist)):

		# Range over which to plot
		lolim = linelist[i] - 5
		uplim = linelist[i] + 5

		# Make mask for wavelength
		try:
			mask = np.where((obswvl > lolim) & (obswvl < uplim))

			if len(mask[0]) > 0:

				if ivar is not None:
					yerr=np.power(ivar[mask],-0.5)
				else:
					yerr=None

				# Plot fits
				plt.figure(1)
				plt.subplot(nrows,ncols,i+1)
				plt.axvspan(linelist[i] - linewidth[i], linelist[i] + linewidth[i], color='green', alpha=0.25)
				if (synthfluxup is not None) and (synthfluxdown is not None):
					plt.fill_between(obswvl[mask], synthfluxup[mask], synthfluxup[mask], facecolor='red', alpha=0.25, label='Synthetic')
				else:
					plt.plot(obswvl[mask], synthflux[mask], 'r-', label='Synthetic')
				plt.errorbar(obswvl[mask], obsflux[mask], yerr=yerr, color='k', fmt='o', label='Observed')

				# Only plot residuals if synth spectrum has been smoothed to match obswvl
				plt.figure(2)
				plt.subplot(nrows,ncols,i+1)
				plt.axvspan(linelist[i] - linewidth[i], linelist[i] + linewidth[i], color='green', alpha=0.25)
				plt.errorbar(obswvl[mask], obsflux[mask] - synthflux[mask], yerr=yerr, color='k', fmt='o', label='Residuals')
				plt.axhline(0, color='r', linestyle='solid', label='Zero')

		except:
			continue

	# Legend for plot showing fits
	fig = plt.figure(1)
	fig.text(0.5, 0.04, 'Wavelength (A)', fontsize=18, ha='center', va='center')
	fig.text(0.06, 0.5, 'Relative flux', fontsize=18, ha='center', va='center', rotation='vertical')
	#plt.ylabel('Relative flux')
	#plt.xlabel('Wavelength (A)')
	plt.legend(loc='best')
	plt.savefig(outputname+'/'+specname+'_finalfits.png',bbox_inches='tight')
	plt.close(1)

	fig2 = plt.figure(2)
	fig2.text(0.5, 0.04, 'Wavelength (A)', fontsize=18, ha='center', va='center')
	fig2.text(0.06, 0.5, 'Residuals', fontsize=18, ha='center', va='center', rotation='vertical')
	plt.ylabel('Residuals')
	plt.xlabel('Wavelength (A)')
	plt.legend(loc='best')
	plt.savefig(outputname+'/'+specname+'_resids.png',bbox_inches='tight')
	plt.close(2)

# Observed spectrum
class obsSpectrum:

	def __init__(self, obsfilename, paramfilename, starnum, wvlcorr, galaxyname, slitmaskname, globular, lines, obsspecial=None, plot=False):

		# Observed star
		self.obsfilename 	= obsfilename 	# File with observed spectra
		self.paramfilename  = paramfilename # File with parameters of observed spectra
		self.starnum 		= starnum		# Star number
		self.galaxyname 	= galaxyname 	# Name of galaxy
		self.slitmaskname 	= slitmaskname 	# Name of slitmask
		self.globular 		= globular		# Parameter marking if globular cluster
		self.lines 			= lines 		# Parameter marking whether or not to use revised or original linelist

		# If observed spectrum comes from moogify file (default), open observed file and continuum normalize as usual
		if obsspecial is None:

			# Output filename
			if self.globular:
				self.outputname = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname
			else:
				self.outputname = '/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname

			# Open observed spectrum
			self.specname, self.obswvl, self.obsflux, self.ivar, self.dlam, self.zrest = open_obs_file(self.obsfilename, retrievespec=self.starnum)

			# Get measured parameters from observed spectrum
			self.temp, self.logg, self.fe, self.alpha, self.fe_err = open_obs_file(self.paramfilename, self.starnum, specparams=True, objname=self.specname)

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
				plt.savefig(self.outputname+'/'+self.specname+'_obs.png')
				plt.close()

			# Get synthetic spectrum, split both obs and synth spectra into red and blue parts
			synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs_for_division(self.obswvl, self.obsflux, self.ivar, temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, dlam=self.dlam, lines=self.lines)

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
				#plt.xlim((6553,6573))
				plt.savefig(self.outputname+'/'+self.specname+'_obsnormalized.png')
				plt.close()

			if wvlcorr:
				print('Doing wavelength correction...')

				try:
					# Compute standard deviation
					contdivstd = np.zeros(len(self.ivar_norm))+np.inf
					contdivstd[self.ivar_norm > 0] = np.sqrt(np.reciprocal(self.ivar_norm[self.ivar_norm > 0]))

					# Wavelength correction
					self.obswvl = fit_wvl(self.obswvl, self.obsflux_norm, contdivstd, self.dlam, 
						self.temp, self.logg, self.fe, self.alpha, self.specname, self.outputname+'/')

					print('Done with wavelength correction!')

				except:
					print('Couldn\'t complete wavelength correction for some reason.')

			# Crop observed spectrum into regions around Mn lines
			self.obsflux_fit, self.obswvl_fit, self.ivar_fit, self.dlam_fit, self.skip = mask_obs_for_abundance(self.obswvl, self.obsflux_norm, self.ivar_norm, self.dlam, lines=self.lines)

		# Else, take spectrum and observed parameters from obsspecial keyword
		else:

			# Output filename
			self.outputname = '/raid/madlr/test/'+slitmaskname

			self.obsflux_fit = obsspecial[0]
			self.obswvl_fit = obsspecial[1]
			self.ivar_fit 	= obsspecial[2]
			self.dlam_fit 	= obsspecial[3]
			self.skip 		= obsspecial[4]
			self.temp		= obsspecial[5]
			self.logg		= obsspecial[6]
			self.fe 		= obsspecial[7]
			self.alpha 		= obsspecial[8]
			self.fe_err 	= obsspecial[9]

			self.specname 	= self.slitmaskname

		# Splice together Mn line regions of observed spectra
		print('Skip: ', self.skip)

		self.obsflux_final = np.hstack((self.obsflux_fit[self.skip]))
		self.obswvl_final = np.hstack((self.obswvl_fit[self.skip]))
		self.ivar_final = np.hstack((self.ivar_fit[self.skip]))
		#self.dlam_final = np.hstack((self.dlam_fit[self.skip]))
		#print(len(self.obswvl_final))

	# Define function to minimize
	def synthetic(self, obswvl, mn, dlam, full=True):
		"""Get synthetic spectrum for fitting.

		Inputs:
		obswvl  -- independent variable (wavelength)
		parameters to fit:
			mn 		-- Mn abundance
			dlam    -- FWHM to be used for smoothing

		Keywords:
		full 	-- if True, splice together all Mn line regions; else, keep as array

		Outputs:
		synthflux -- array-like, output synthetic spectrum
		"""

		# Compute synthetic spectrum
		print('Computing synthetic spectrum with parameters: ', mn, dlam)
		synth = runMoog(temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, elements=[25], abunds=[mn], solar=[5.43], lines=self.lines)

		#print('Ran synthetic spectrum.')

		# Loop over each line
		synthflux = []
		for i in self.skip:

			synthregion = synth[i]
			#print(len(synthregion), len(self.obswvl_fit[i]),len(self.obsflux_fit[i]),len(self.ivar_fit[i]), len(self.dlam_fit[i]))

			# Smooth each region of synthetic spectrum to match each region of continuum-normalized observed spectrum

			# uncomment this if dlam is not a fitting parameter
			# newsynth = get_synth(self.obswvl_fit[i], self.obsflux_fit[i], self.ivar_fit[i], self.dlam_fit[i], synth=synthregion)

			# uncomment this if dlam is a fitting parameter
			newsynth = get_synth(self.obswvl_fit[i], self.obsflux_fit[i], self.ivar_fit, dlam, synth=synthregion)
			synthflux.append(newsynth)

		print('Finished smoothing synthetic spectrum!')

		# If necessary, splice everything together
		if full:
			synthflux = np.hstack(synthflux[:])

		return synthflux

	def minimize_scipy(self, params0, plots=False, output=False):
		"""Minimize residual using scipy.optimize Levenberg-Marquardt.

		Inputs:
		params0 -- initial guesses for parameters:
			mn0 -- Mn abundance
			smoothivar0 -- if applicable, inverse variance to use for smoothing

		Keywords:
		plots  -- if 'True', also plot final fit & residual
		output -- if 'True', also output a file (default='False')

		Outputs:
		fitparams -- best fit parameters
		rchisq	  -- reduced chi-squared
		"""

		# Do minimization
		print('Starting minimization! Initial guesses: ', params0)
		best_mn, covar = scipy.optimize.curve_fit(self.synthetic, self.obswvl_final, self.obsflux_final, p0=[params0], sigma=np.sqrt(np.reciprocal(self.ivar_final)), epsfcn=0.01)
		error = np.sqrt(np.diag(covar))

		print('Answer: ', best_mn)
		print('Error: ', error)

		# Do some checks
		if len(np.atleast_1d(best_mn)) == 1:
			finalsynth = self.synthetic(self.obswvl_final, best_mn, full=True)
		else:
			finalsynth = self.synthetic(self.obswvl_final, best_mn[0], best_mn[1], full=True)

		# Make plots
		if plots:
			make_plots(self.lines, self.specname, self.obswvl_final, self.obsflux_final, finalsynth, self.outputname, ivar=self.ivar_final)

		# Output the final data
		if output:

			if len(np.atleast_1d(best_mn)) == 1:
				finalsynthup 	= self.synthetic(self.obswvl_final, best_mn + error, full=True)
				finalsynthdown 	= self.synthetic(self.obswvl_final, best_mn - error, full=True)
			else:
				finalsynthup = self.synthetic(self.obswvl_final, best_mn[0] + error[0], best_mn[1], full=True)
				finalsynthdown = self.synthetic(self.obswvl_final, best_mn[0] - error[0], best_mn[1], full=True)

			# Create file
			filename = self.outputname+'/'+self.specname+'_data.csv'

			# Define columns
			columnstr = ['wvl','obsflux','synthflux','synthflux_up','synthflux_down','ivar']
			columns = np.asarray([self.obswvl_final, self.obsflux_final, finalsynth, finalsynthup, finalsynthdown, self.ivar_final])

			with open(filename, 'w') as csvfile:
				datawriter = csv.writer(csvfile, delimiter=',')

				# Write header
				datawriter.writerow(['[Mn/H]', best_mn[0]])
				if len(np.atleast_1d(best_mn)) > 1:
					datawriter.writerow(['dlam', best_mn[1]])
				datawriter.writerow(columnstr)

				# Write data
				for i in range(len(finalsynth)):
					datawriter.writerow(columns[:,i])

		return best_mn, error

	def plot_chisq(self, params0, minimize=True):
		"""Plot chi-sq as a function of [Mn/H].

		Inputs:
		params0 -- initial guesses for parameters:
			mn0 -- Mn abundance
			smoothivar0 -- if applicable, inverse variance to use for smoothing

		Keywords:
		minimize -- if 'True' (default), mn0 is an initial guess, and code will minimize;
					else, mn0 must be a list containing the best-fit Mn and the error [mn_result, mn_error]

		Outputs:
		fitparams -- best fit parameters
		rchisq	  -- reduced chi-squared
		"""

		if minimize:
			mn_result, mn_error = self.minimize_scipy(params0, output=True)
		else:
			mn_result = [params0[0]]
			mn_error  = [params0[1]]

		mn_list = np.array([-3,-2,-1.5,-1,-0.5,-0.1,0,0.1,0.5,1,1.5,2,3])*mn_error[0] + mn_result[0]
		dlam 	= mn_result[1]
		chisq_list = np.zeros(len(mn_list))
		for i in range(len(mn_list)):
			finalsynth = self.synthetic(self.obswvl_final, mn_list[i], dlam)
			chisq = np.sum(np.power(self.obsflux_final - finalsynth, 2.) * self.ivar_final) / (len(self.obsflux_final) - 1.)
			chisq_list[i] = chisq

		if mn_error < 1.0:
			plt.figure()
			plt.title('Star '+self.specname, fontsize=18)
			plt.plot(mn_list, chisq_list, '-o')
			plt.ylabel(r'$\chi^{2}_{red}$', fontsize=16)
			plt.xlabel('[Mn/H]', fontsize=16)
			plt.savefig(self.outputname+'/'+self.specname+'_redchisq.png')
			plt.close()

		return mn_result, mn_error, chisq_list[6]

def main():
	filename = '/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz'
	#paramfilename = '/raid/m31/dsph/scl/scl1/moogify7_flexteff.fits.gz'
	paramfilename = '/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz'
	galaxyname = 'scl'
	slitmaskname = 'scl5_1200B'

	# Code for Evan for Keck 2019A proposal
	#test1 = obsSpectrum(filename, paramfilename, 16, True, galaxyname, slitmaskname, False, 'new', plot=True).minimize_scipy(-2.68, output=True)
	#test2 = obsSpectrum(filename, paramfilename, 30, True, galaxyname, slitmaskname, False, 'new', plot=True).minimize_scipy(-1.29, output=True)
	test2 = obsSpectrum(filename, paramfilename, 26, True, galaxyname, slitmaskname, False, 'new', plot=True).minimize_scipy([-1.50, 0.48], output=True)

	#print('we done')
	#test = obsSpectrum(filename, 57).plot_chisq(-2.1661300692266998)

if __name__ == "__main__":
	main()