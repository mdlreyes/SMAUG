# fit_mnfe_feh.py
# Use model described in Kirby (in prep.) to determine Type Ia SNe
# yields for [Mn/H]
#
# Created 20 Nov 18
# Updated 29 Nov 18
###################################################################

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import emcee
import corner
import scipy.optimize as op
import math
from astropy.io import fits, ascii
import pandas
import matplotlib.ticker as ticker
from statsmodels.stats.weightstats import DescrStatsW

def fit_mnfe_feh(filenames, mnfe_check, outfile, title, fehia, maxerror=None, gratings=None):
	"""Use model described in Kirby (in prep.) to determine Type Ia supernova yields for [Mn/H]

	Inputs:
	filename 	-- list of input filenames
	mnfe_check 	-- list of bools indicating whether input file includes [Mn/Fe] (True) or [Mn/H] (False)
	outfile 	-- name of output file
	title 		-- title of graph
	fehia 		-- [Fe/H] when Type Ia SNe turn on

	Keywords:
	maxerror 	-- if not None, points with error > maxerror will not be used in computation
	gratings 	-- if not None, must be list of colors for different input filenames.
					Plot points from different filenames in different colors.
	"""

	#############
	# Prep data #
	#############

	# Get data from files
	name 	= []
	feh 	= []
	feherr 	= []
	mnh 	= []
	mnherr	= []
	colors  = []
	redchisq = []
	mnfeflag = []

	for i in range(len(filenames)):

		file = filenames[i]
		current_name 	= np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')

		data = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=[5,6,8,9,10])
		current_feh 	= data[:,0]
		current_feherr 	= data[:,1]
		current_mnh 	= data[:,2]
		current_mnherr 	= data[:,3]
		current_redchisq = data[:,4]
		current_mnfeflag = len(current_feh) * [mnfe_check[i]]

		# Append to total data arrays
		name.append(current_name)
		feh.append(current_feh)
		feherr.append(current_feherr)
		mnh.append(current_mnh)
		mnherr.append(current_mnherr)
		redchisq.append(current_redchisq)
		mnfeflag.append(current_mnfeflag)

		if gratings is not None:
			current_grating = len(current_feh) * [gratings[i]]
			colors.append(current_grating)

	# Convert back to numpy arrays
	name 	= np.hstack(name)
	feh 	= np.hstack(feh)
	feherr 	= np.hstack(feherr)
	mnh 	= np.hstack(mnh)
	mnherr  = np.hstack(mnherr)
	colors  = np.hstack(colors)
	redchisq = np.hstack(redchisq)
	mnfeflag = np.hstack(mnfeflag)

	# Compute [Mn/Fe]
	mnfe = np.zeros(len(mnh))
	mnfeerr = np.zeros(len(mnherr))

	for i in range(len(feh)):
		if mnfeflag[i]:
			mnfe[i] = mnh[i]
			mnfeerr[i] = mnherr[i]
		else:
			mnfe[i] = mnh[i] - feh[i]
			mnfeerr[i] = np.sqrt(np.power(feherr[i],2.)+np.power(mnherr[i],2.))

	# Remove points with error > maxerror
	if maxerror is not None:
		goodmask 	= np.where((mnfeerr < maxerror)) # & notoutlier) # & (redchisq < 3.0))
		name 	= name[goodmask]
		feh 	= feh[goodmask]
		mnfe 	= mnfe[goodmask]
		mnfeerr = mnfeerr[goodmask]
		colors  = colors[goodmask]

	# Define outliers if necessary
	outlier = np.where((mnfe > 0.5))[0]
	print(name[outlier])
	'''
	notoutlier = np.ones(len(mnfe), dtype='bool')
	notoutlier[outlier] = False
	notoutlier = np.ones(len(mnfe), dtype='bool')
	'''

	#######################
	# Fit a simple model! #
	#######################

	# Start by defining log likelihood function
	def lnlike(params, x, y, xerr, yerr):
		theta, bperp = params

		L = 0

		for i in range(len(x)):

			if x[i] <= fehia:
				delta = y[i] - (bperp + fehia*np.sin(theta))/(np.cos(theta))
				sigma = yerr[i]
				test = 1

			else:
				delta = y[i]*np.cos(theta) - x[i]*np.sin(theta) - bperp
				sigma = np.sqrt(yerr[i]**2. * np.cos(theta)**2. + xerr[i]**2. * np.sin(theta)**2.)
				test = 2

			L = L + np.log(1/(2.*sigma)) - (delta**2.)/(2*sigma**2.)

		return L

	# Define the priors
	def lnprior(params):
		theta, bperp = params
		if -np.pi/2. < theta < np.pi/2.:
			return 0.0
		return -np.inf

	# Define the full log-probability function
	def lnprob(params, x, y, xerr, yerr):
		lp = lnprior(params)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(params, x, y, xerr, yerr)

	# Start by doing basic max-likelihood fit to get initial values
	nll = lambda *args: -lnlike(*args)
	maskinit = np.where((feh > -2.5))
	result = op.minimize(nll, [0., 0.], args=(feh[maskinit], mnfe[maskinit], feherr[maskinit], mnfeerr[maskinit]))
	theta_init, b_init = result["x"]

	# Sample the log-probability function using emcee - first, initialize the walkers
	ndim = 2
	nwalkers = 100
	pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

	# Run MCMC for 11000 steps
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(feh,mnfe,feherr,mnfeerr))
	sampler.run_mcmc(pos,1100)

	# Plot walkers
	fig, ax = plt.subplots(2,1,figsize=(8,8), sharex=True)
	ax = ax.ravel()   

	names = [r"$\theta$", r"$b_{\bot}$"]
	for i in range(ndim):
		for j in range(nwalkers):
			chain = sampler.chain[j,:,i]
			ax[i].plot(range(len(chain)), chain, 'k-', alpha=0.5)
			ax[i].set_ylabel(names[i], fontsize=16)
			for label in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
				label.set_fontsize(14)

	ax[0].set_title(title, fontsize=18)
	ax[1].set_xlabel('Steps', fontsize=16)

	plt.savefig(outfile+'_walkers.png', bbox_inches='tight')
	plt.close()

	# Make corner plots
	samples = sampler.chain[:,100:, :].reshape((-1, ndim))
	cornerfig = corner.corner(samples, labels=[r"$\theta$", r"$b_{\bot}$"],
								quantiles=[0.16, 0.5, 0.84],
								show_titles=True, title_kwargs={"fontsize": 12})
	cornerfig.savefig(outfile+'_cornerfig.png')
	plt.close(cornerfig)

	# Compute 16th, 50th, and 84th percentiles of parameters
	theta = np.array([np.percentile(samples[:,0],16), np.percentile(samples[:,0],50), np.percentile(samples[:,0],84)])
	bperp = np.array([np.percentile(samples[:,1],16), np.percentile(samples[:,1],50), np.percentile(samples[:,1],84)])

	##################
	# Compute yields #
	##################

	# Define axis for plot
	fig, ax = plt.subplots(figsize=(10,5))

	# Compute core-collapse yield
	all_thetas = samples[:,0] # all values in the chain
	all_bperps = samples[:,1]
	all_mnfe_cc = (all_bperps + fehia*np.sin(all_thetas))/np.cos(all_thetas)

	# Fiducial core-collapse yield
	mnfe_cc = np.array([np.percentile(all_mnfe_cc,16), np.percentile(all_mnfe_cc,50), np.percentile(all_mnfe_cc,84)])
	print(mnfe_cc)

	# Plot best fit
	xfit = np.linspace(np.min(feh), ax.get_xlim()[1], 100)

	# Loop over every single x value and compute an array of every possible value of best-fit y
	yfit = np.zeros((3, len(xfit)))
	yvalues = []
	for i in range(len(xfit)):

		if xfit[i] <= fehia:
			yvalues.append(0.)
			yfit[:,i] = mnfe_cc
		else:
			y = xfit[i]*np.tan(all_thetas) + all_bperps/(np.cos(all_thetas))
			yvalues.append(y)
			yfit[:,i] = np.array([np.percentile(y,16), np.percentile(y,50), np.percentile(y,84)])

	# Now, determine the Type Ia yield of Mn
	# Start by determining the amount of Mg/Fe predicted from model
	theta_mg = -0.50 # XXX turn this into upper/median/lower sequence
	bperp_mg = -0.64 # XXX turn this into upper/median/lower sequence
	mgfe = xfit*np.tan(theta_mg) + bperp_mg/(np.cos(theta_mg))

	# Compute f_Ia
	mgfe_cc = 0.55
	mgfe_ia = -1.5
	frac_ia = (np.power(10.,mgfe_cc) - np.power(10.,mgfe)) / (np.power(10.,mgfe) - np.power(10.,mgfe_ia))

	# Compute Mn yield based on best-fit model!
	mnfe_ia = np.zeros((3, len(xfit)))
	for i in range(len(xfit)):
		if xfit[i] > fehia:
			all_mnfe_ia = np.log10( (frac_ia[i] + 1.)/(frac_ia[i]) * np.power(10.,yvalues[i]) - (1./frac_ia[i] * np.power(10.,all_mnfe_cc)) )
			mnfe_ia[:,i] = np.array([np.percentile(all_mnfe_ia,16), np.percentile(all_mnfe_ia,50), np.percentile(all_mnfe_ia,84)])

	#################
	# Create figure #
	#################

	# Scatter plot
	if len(gratings) > 1:
		plotcolors = np.zeros(len(gratings), dtype='bool')
		for i in range(len(feh)):

			if plotcolors[0] == False and colors[i] == '#B0B0B0':
				ax.errorbar(feh[i], mnfe[i], yerr=mnfeerr[i], color=colors[i], marker='o', linestyle='', capsize=3, zorder=100, label='North+12')
				plotcolors[0] = True

			elif plotcolors[1] == False and colors[i] == '#594F4F':
				ax.errorbar(feh[i], mnfe[i], yerr=mnfeerr[i], color=colors[i], marker='o', linestyle='', capsize=3, zorder=100, label='This work')
				plotcolors[1] = True

			else:
				ax.errorbar(feh[i], mnfe[i], yerr=mnfeerr[i], color=colors[i], marker='o', linestyle='', capsize=3, zorder=100)

	else:
		ax.errorbar(feh, mnfe, yerr=mnfeerr, color='k', marker='o', linestyle='', capsize=3, zorder=100)

	# Put sample size on plot
	ax.text(0.025, 0.9, 'N = '+str(len(name)), transform=ax.transAxes, fontsize=18)

	#for i in range(len(outlier)):
	#	idx = outlier[i]
	#	plt.text(feh[idx], mnfe[idx], name[idx])

	# Plot best-fit model
	ax.fill_between(xfit, yfit[2], yfit[0], color='r', alpha=0.25, zorder=200)
	ax.plot(xfit, yfit[1], 'r-', linewidth=3, zorder=200)

	# Plot Type Ia [Mn/Fe] yield
	mask = np.where(xfit > fehia)
	ax.fill_between(xfit[mask], mnfe_ia[2][mask], mnfe_ia[0][mask], color='#547980', alpha=0.25)
	ax.plot(xfit[mask], mnfe_ia[1][mask], color='#547980', linestyle=':', linewidth=3)

	fehmeasure = -1.5 # [Fe/H] at which to measure [Mn/Fe]_Ia
	idx_feh = np.argmin(np.abs(xfit - fehmeasure))
	print('[Mn/Fe] at most metal rich end:')
	print(mnfe_ia[1][-1])
	print(mnfe_ia[2][-1]-mnfe_ia[1][-1])
	print(mnfe_ia[1][-1]-mnfe_ia[0][-1])
	print('final Mn fit:')
	print(mnfe_ia[1][idx_feh])
	print(mnfe_ia[2][idx_feh]-mnfe_ia[1][idx_feh])
	print(mnfe_ia[1][idx_feh]-mnfe_ia[0][idx_feh])

	# Also plot core-collapse yield
	ax.fill_between(ax.get_xlim(), mnfe_cc[2], mnfe_cc[0], color='#45ADA8', alpha=0.25, zorder=0)
	ax.plot(ax.get_xlim(), mnfe_cc[1]*np.ones(2), color='#45ADA8', linestyle='--', linewidth=3, zorder=0)

	# Format plot
	ax.set_title(title, fontsize=18)
	ax.set_xlabel('[Fe/H]', fontsize=16)
	ax.set_ylabel('[Mn/Fe]', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)
	ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
	ax.set_xlim([-2.75,-0.75])
	#ax.set_ylim([-2,2])
	#ax.set_ylim([-1.5,1.1])
	leg = plt.legend(fancybox=True, framealpha=0.5, loc='best')
	for text in leg.get_texts():
		plt.setp(text, color='#594F4F', fontsize=18)

	# Output file
	plt.savefig(outfile+'_mnfe.png', bbox_inches='tight', transparent=True)
	plt.show()

def compare_mnfe(outfile):
	"""Plot [Mn/Fe] values on number line.

	Inputs:
	outfile 	-- name of output file

	Keywords:
	"""

	'''
	if typeii:
		typeii_Z  = np.array([0.000000000001, 0.001,0.004,0.02])			# Metallicities from Nomoto+2006 (in total Z)
		typeii_mn = np.array([8.72e-7, 9.72e-7, 1.16e-6, 1.54e-6])	# Mn yields from Nomoto+2006 (in units of Msun)
		typeii_fe54 = np.array([7.32e-6, 8.34e-6, 9.03e-6, 1.13e-5]) # Fe-54 yields from Nomoto+2006 (in units of Msun)
		typeii_fe56 = np.array([3.17e-4, 3.38e-4, 3.22e-4, 3.48e-4]) # Fe-56 yields from Nomoto+2006 (in units of Msun)
		typeii_fe57 = np.array([4.66e-6, 6.03e-6, 6.79e-6, 9.57e-6]) # Fe-57 yields from Nomoto+2006 (in units of Msun)
		typeii_fe58 = np.array([6.15e-12, 1.81e-7, 5.26e-7, 2.15e-6]) # Fe-58 yields from Nomoto+2006 (in units of Msun)

		solar_mn = 5.43
		solar_fe = 7.50

		typeii_mnh = np.log10((typeii_mn/55.) / ((typeii_fe54/54.) + (typeii_fe56/56.) + (typeii_fe57/57.) + (typeii_fe58/58.))) - (solar_mn - solar_fe)
		typeii_feh = np.log10(typeii_Z/0.0134) # Solar metallicity from Asplund+2009

		ax.plot(typeii_feh, typeii_mnh, 'b-', label='CCSNe yield')
	'''

	# Setup a plot so that only the bottom spine is shown
	def setup(ax):
		ax.spines['right'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.yaxis.set_major_locator(ticker.NullLocator())
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_position('center')
		ax.spines['bottom'].set_linewidth(2)
		ax.xaxis.set_ticks_position('bottom')
		ax.tick_params(which='major', width=2)
		ax.tick_params(which='major', length=12)
		ax.tick_params(which='minor', width=2)
		ax.tick_params(which='minor', length=8)
		ax.set_xlim(-2, 1)
		ax.set_ylim(-0.5, 0.5)
		ax.patch.set_alpha(0.0)

		ax.xaxis.set_ticklabels([])
		ax.yaxis.set_ticklabels([])

	plt.figure(figsize=(20,1))

	# Make plot
	ax = plt.subplot(1,1,1)
	setup(ax)
	ax.xaxis.set_major_locator(ticker.AutoLocator())
	ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
	#ax.text(0.0, 0.1, "AutoLocator()", fontsize=14, transform=ax.transAxes)
	#ax.set_xlabel('[Mn/Fe]', fontsize=16)

	# Plot observed [Mn/Fe]
	ax.errorbar(-0.87, 0, xerr=0.085, color='r', marker='o', linestyle='', capsize=10, elinewidth=6, capthick=6, markersize=20, zorder=100)

	# Plot models
	ax.axvline(0.18, color='#45ADA8', linestyle=':', linewidth=6, label='DDT(T16)')
	ax.axvspan(0.01, 0.53, color='#B0B0B0', hatch='\\', alpha=0.6, label='DDT(S13)')
	ax.axvspan(0.36, 0.52, color='#45ADA8', hatch='//', alpha=0.75, label='Def(F14)')
	ax.axvspan(-1.69, -1.21, color='#9DE0AD', alpha=0.8, label='Sub(B)')
	ax.axvspan(-1.52, -0.68, color='#547980', hatch='//', alpha=0.5, label='Sub(S18)')

	# Output file
	plt.savefig(outfile, bbox_inches='tight', transparent=True)
	plt.show()

	return

def main():

	# Plot for Sculptor
	fit_mnfe_feh(['data/bscl5_1200B_final3.csv'],[False],'figures/scl_fit3', 'Sculptor dSph', fehia=-2.12, maxerror=0.3, gratings=['#594F4F'])
	fit_mnfe_feh(['data/bscl5_1200B_final3.csv','data/hires_data_final/scl/north12_final.csv'],[False,True],'figures/scl_fit_total', 'Sculptor dSph', fehia=-2.34, maxerror=0.3, gratings=['#594F4F','#B0B0B0'])

	# Plot for Ursa Minor
	fit_mnfe_feh(['data/bumia_1200B_final3.csv'],[False],'figures/umi_fit3', 'Ursa Minor dSph', fehia=-2.42, maxerror=0.3, gratings=['#594F4F'])

	# Plot [Mn/Fe] values on number line
	#compare_mnfe('figures/scl_mnfe_comparison.png')

	return

if __name__ == "__main__":
	main()