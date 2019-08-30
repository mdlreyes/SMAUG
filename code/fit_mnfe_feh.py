# fit_mnfe_feh.py
# Use model described in Kirby (in prep.) to determine Type Ia SNe
# yields for [Mn/H]
#
# Created 20 Nov 18
# Updated 29 Nov 18
###################################################################

import matplotlib as mpl
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import os
import sys
import numpy as np
import emcee
import corner
import scipy.optimize as op
import math
from astropy.io import fits, ascii
import pandas as pd 
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from statsmodels.stats.weightstats import DescrStatsW
import cycler

def fit_mnfe_feh(file, mnfe_check, outfile, title, fehia, maxerror=None, nlte=False, bestfit=False, sne=False, literature=None, ia_comparison=False):
	"""Use model described in Kirby (in prep.) to determine Type Ia supernova yields for [Mn/H]

	Inputs:
	file 		-- input filenames
	mnfe_check 	-- list of bools indicating whether input file includes [Mn/Fe] (True) or [Mn/H] (False)
	outfile 	-- name of output file
	title 		-- title of graph
	fehia 		-- [Fe/H] when Type Ia SNe turn on

	Keywords:
	maxerror 	-- if not None, points with error > maxerror will not be used in computation
	nlte		-- if 'True', apply statistical NLTE correction
	bestfit, sne, ia_comparison -- if 'True', plot best-fit model / SNe yields / Z-dep Type Ia models
	literature  -- if not None, must be list of files with data that can be overplotted
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
	redchisq = []
	mnfeflag = []

	name 	= np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')

	data = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=[5,6,8,9,10])
	feh 	= data[:,0]
	feherr 	= data[:,1]
	mnh 	= data[:,2]
	mnherr 	= data[:,3]
	redchisq = data[:,4]
	mnfeflag = len(feh) * [mnfe_check]

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

	# Define outliers if necessary
	#outlier = np.where((mnfe > 0.7) & (feh < -2.25))[0]
	#outlier = np.where(((feh-feherr) < -2.5))[0]
	#outlier = np.where((mnfe > 0.25))
	#print(name[outlier])
	#notoutlier = np.ones(len(mnfe), dtype='bool')
	#notoutlier[outlier] = False

	# Remove points with error > maxerror
	if maxerror is not None:
		goodmask 	= np.where((mnfeerr < maxerror)) # & notoutlier) # & (redchisq < 3.0))
		name 	= name[goodmask]
		feh 	= feh[goodmask]
		feherr  = feherr[goodmask]
		mnfe 	= mnfe[goodmask]
		mnfeerr = mnfeerr[goodmask]

	if nlte:
		nltecorr = -0.096*feh + 0.173
		mnfe_lte = mnfe
		mnfe = mnfe + nltecorr
		print('NLTE average correction: '+str(np.average(nltecorr)))

	# Get R from Evan's catalog
	rfile = fits.open('/Users/miadelosreyes/Documents/Research/MnDwarfs/code/data/dsph_data/Evandata/chemev_scl.fits')
	R = rfile[1].data['R'].T
	xfit = np.linspace(-2.97,0, num=100)

	#######################
	# Fit a simple model! #
	#######################

	# Function to compute Type Ia (Mn/Fe) (NOT [Mn/Fe]!!!) yield
	def compute_mnfe_ia(theta, bperp, R, xfit):
		fitidx = np.where((R > 0.))[0]
		y = xfit[fitidx] * np.tan(theta) + bperp/np.cos(theta)

		mnfe_cc = (bperp + fehia*np.sin(theta))/np.cos(theta)
		mnfe_ia = (R[fitidx] + 1.)/(R[fitidx]) * np.power(10.,y) - (1./R[fitidx] * np.power(10.,mnfe_cc))

		return mnfe_cc, mnfe_ia

	# Start by defining log likelihood function
	def lnlike(params, x, y, xerr, yerr):
		theta, bperp = params

		# Make sure (Mn/Fe)_Ia is positive
		mnfe_cc, test = compute_mnfe_ia(theta, bperp, R, xfit)
		if (test < 0).any():
			return -np.inf

		'''
		L = 0.
		for i in range(len(x)):

			if x[i] <= fehia:
				delta = y[i] - (bperp + fehia*np.sin(theta))/(np.cos(theta))
				sigma = yerr[i]
				test = 1

			else:
				delta = y[i]*np.cos(theta) - x[i]*np.sin(theta) - bperp
				sigma = np.sqrt(yerr[i]**2. * np.cos(theta)**2. + xerr[i]**2. * np.sin(theta)**2.)
				test = 2

			if i==0:
				print(x[i], y[i], delta, sigma)

			L = L - np.log(np.sqrt(2.*np.pi)*sigma) - (delta**2.)/(2*sigma**2.)

		'''
		delta = y*np.cos(theta) - x*np.sin(theta) - bperp
		sigma = np.sqrt(yerr**2. * np.cos(theta)**2. + xerr**2. * np.sin(theta)**2.)

		lessthan = np.where((x <= fehia))[0]
		delta[lessthan] = y[lessthan] - (bperp + fehia*np.sin(theta))/(np.cos(theta))
		sigma[lessthan] = yerr[lessthan]

		L = np.sum( (-1.*np.log(np.sqrt(2.*np.pi)*sigma) - (np.power(delta,2.))/(2*np.power(sigma,2.))) )

		# Add in prior for [Mn/Fe]_CC ~ -0.3 from metal-poor MW halo
		delta_cc = -0.3 - mnfe_cc
		sigma_cc = 0.1
		L = L + (-1.*np.log(np.sqrt(2.*np.pi)*sigma_cc) - (np.power(delta_cc,2.))/(2*np.power(sigma_cc,2.)))

		return L

	# Define the priors
	def lnprior(params):
		theta, bperp = params
		if -np.pi/2. < theta < np.pi/2. and -10.0 < bperp < 10.0:
			return 0.0
		return -np.inf

	# Define the full log-probability function
	def lnprob(params, x, y, xerr, yerr):
		lp = lnprior(params)
		ll = lnlike(params, x, y, xerr, yerr)
		if np.isfinite(lp) and np.isfinite(ll):
			return lp + ll
		else:
			return lp + ll

	# Start by doing basic max-likelihood fit to get initial values
	nll = lambda *args: -lnlike(*args)
	maskinit = np.where((feh > -2.1))[0]
	result = op.minimize(nll, [0., 0.], args=(feh[maskinit], mnfe[maskinit], feherr[maskinit], mnfeerr[maskinit]))
	theta_init, b_init = result["x"]

	# Sample the log-probability function using emcee - first, initialize the walkers
	ndim = 2
	nwalkers = 100
	pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

	# Run MCMC for 11000 steps
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(feh,mnfe,feherr,mnfeerr))
	sampler.run_mcmc(pos,1e3)

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

	print(theta, bperp)

	##################
	# Compute yields #
	##################

	# Define axis for plot
	fig, ax = plt.subplots(figsize=(10,8))

	# Compute core-collapse yield
	all_thetas = samples[:,0] # all values in the chain
	all_bperps = samples[:,1]
	all_mnfe_cc = (all_bperps + fehia*np.sin(all_thetas))/np.cos(all_thetas)

	# Fiducial core-collapse yield
	mnfe_cc = np.array([np.percentile(all_mnfe_cc,16), np.percentile(all_mnfe_cc,50), np.percentile(all_mnfe_cc,84)])
	print('[Mn/Fe]_CC: ',mnfe_cc)

	frac_ia = R/(R+1.)
	'''
	plt.plot(xfit, frac_ia)
	plt.show()
	return
	'''

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

	# Now, determine the Type Ia yield of Mn based on best-fit model!
	mnfe_ia = np.zeros((3, len(R)))
	plotmask = np.zeros(len(xfit), dtype='bool')
	for i in range(len(R)):
		if R[i] > 0:
			test = np.log10( (R[i] + 1.)/(R[i]) * np.power(10.,yvalues[i]) - (1./R[i] * np.power(10.,all_mnfe_cc)) )
			mask = np.isnan(test)
			if mask.any() == False:
				plotmask[i] = True

			all_mnfe_ia = test[~mask]
			mnfe_ia[:,i] = np.array([np.percentile(all_mnfe_ia,16), np.percentile(all_mnfe_ia,50), np.percentile(all_mnfe_ia,84)])

	#################
	# Create figure #
	#################

	# List to hold all patches and labels
	patches = []
	labels = []

	# Plot literature values
	if literature is not None:

		# Plot scatter plot of our measurements
		ax.errorbar(feh, mnfe, markersize=8, color='k', marker='o', linestyle='', capsize=0, zorder=102, alpha=0.8) #, xerr=feherr, yerr=mnfeerr)
		handle1, = ax.plot([],[], color='k', marker='o', linestyle='None', mfc='k', markersize=8)
		patches.append(handle1)
		labels.append('This work')

		for litfile in literature:

			if litfile == 'Sobeck+06':
				data = np.genfromtxt('data/hires_data_final/sobeck06.csv', delimiter=[9,4,9,5,6,5,6,6,6], skip_header=24, usecols=[7,8])
				current_feh 	= data[:,0]
				current_feherr 	= np.zeros(len(current_feh))
				current_mnfe 	= data[:,1]
				current_mnfeerr = np.zeros(len(current_mnfe))

				current_label = r'Sobeck+06 ($N='+str(len(current_feh))+' $)'
				marker = '.'
				color = 'darkgray'
				zorder = 1

			if litfile == 'North+12':
				data = np.genfromtxt('data/hires_data_final/scl/north12_final.csv', delimiter='\t', skip_header=1, usecols=[5,6,8,9])
				current_feh 	= data[:,0]
				current_feherr 	= data[:,1]
				current_mnfe 	= data[:,2]
				current_mnfeerr = data[:,3]

				current_label = r'North+12 ($N='+str(len(current_feh))+' $)'
				marker = 's'
				color = 'darkgreen'
				zorder = 2

			ax.errorbar(current_feh, current_mnfe, markersize=8, color=color, marker=marker, linestyle='', capsize=0, xerr=current_feherr, yerr=current_mnfeerr, zorder=zorder)
			handle1, = ax.plot([],[], color=color, marker=marker, linestyle='None', mfc=color, markersize=8)
			patches.append(handle1)
			labels.append(current_label)

		outfile += '_lit'

	# Scatter plot of our observations
	elif nlte==False: 
		if ia_comparison==False:
			ax.errorbar(feh, mnfe, markersize=8, color='k', marker='o', linestyle='', capsize=0, zorder=100, xerr=feherr, yerr=mnfeerr)

	# NLTE corrections
	else:
		# Plot NLTE
		nlteplot = ax.errorbar(feh, mnfe, markersize=8, color='k', marker='o', linestyle='', capsize=0, zorder=100, xerr=feherr, yerr=mnfeerr, label='1D NLTE')
		handle1, = ax.plot([],[], color='k', marker='o', linestyle='None', mfc='k', markersize=8)
		patches.append(handle1)
		labels.append('1D NLTE')

		# Plot LTE
		lteplot, = ax.plot(feh, mnfe_lte, markersize=8, color='k', marker='o', mfc='None', linestyle='', zorder=100, label='1D LTE')
		patches.append(lteplot)
		labels.append('1D LTE')

	# Plot best-fit model
	if bestfit:
		ax.fill_between(xfit, yfit[2], yfit[0], color='C9', alpha=0.25, zorder=200)
		bestfit1 = mpatches.Patch(color='C9', alpha=0.25)
		bestfit2, = ax.plot(xfit, yfit[1], color='C9', linestyle='-', linewidth=3, zorder=200)

		patches.append((bestfit1, bestfit2))
		labels.append("Best fit model")

	if sne:

		# Plot Type Ia [Mn/Fe] yield
		mask = np.nonzero(mnfe_ia[1])
		ax.fill_between(xfit[plotmask], mnfe_ia[2][plotmask], mnfe_ia[0][plotmask], color='r', alpha=0.4, zorder=200)
		typeia1 = mpatches.Patch(color='r', alpha=0.4)
		typeia2, = ax.plot(xfit[mask], mnfe_ia[1][mask], color='r', linestyle='--', linewidth=3, zorder=200)

		fehmeasures = np.array((-1.5,-1.0,-2.0)) # [Fe/H] at which to measure [Mn/Fe]_Ia
		for fehmeasure in fehmeasures:
			idx_feh = np.argmin(np.abs(xfit - fehmeasure))
			print('[Mn/Fe] at '+str(fehmeasure))
			print(mnfe_ia[1][idx_feh])
			print(mnfe_ia[2][idx_feh]-mnfe_ia[1][idx_feh])
			print(mnfe_ia[1][idx_feh]-mnfe_ia[0][idx_feh])

		patches.append((typeia1, typeia2))
		labels.append("Type Ia yield")

		# Also plot core-collapse yield
		ax.fill_between(ax.get_xlim(), mnfe_cc[2], mnfe_cc[0], color='C0', alpha=0.4, zorder=150)
		typeii1 = mpatches.Patch(color='C0', alpha=0.4)
		typeii2, = ax.plot(ax.get_xlim(), mnfe_cc[1]*np.ones(2), color='C0', linestyle='-', linewidth=3, zorder=150)

		patches.append((typeii1, typeii2))
		labels.append("Core-collapse yield")

	ax.set_xlim([-2.8,-0.75])
	ax.set_ylim([-1.2,1.0])

	if ia_comparison:
		outfile += '_zdep'

		ax.set_xlim([-2.1,-0.8])
		ax.set_ylim([-2.0,0.1])

		# Get container for all legends
		legends = []

		# Plot Type Ia yield
		mask = np.nonzero(mnfe_ia[1])
		ax.fill_between(xfit[plotmask], mnfe_ia[2][plotmask], mnfe_ia[0][plotmask], color='r', alpha=0.4, zorder=200)
		typeia1 = mpatches.Patch(color='r', alpha=0.4)
		typeia2, = ax.plot(xfit[mask], mnfe_ia[1][mask], color='r', linestyle='--', linewidth=3, zorder=200)

		legends.append(ax.legend([(typeia1,typeia2)], ['This work'], bbox_to_anchor=(1.04,1.0), loc=2, borderaxespad=0, fontsize=20, frameon=False))

		# Get properties for each plot
		authors = ['sub(S18)','sub(L19)','sub(B19)']
		linestyles = ['-','--',':']

		# Get colors
		blues = plt.cm.Blues(np.linspace(0.3,1.,6))

		# Loop over each author and plot lines for each model
		for k in range(len(authors)):

			feh, mnfe, mass = get_theory(authors[k])

			if authors[k] == 'sub(L19)':
				line, = ax.plot(feh, mnfe, color=blues[4], linestyle=linestyles[k], label=r'1.10$M_{\odot}$', zorder=0, linewidth=1.5)

				l1 = ax.legend(handles=[line], title='sub(L19)', bbox_to_anchor=(1.04,0.88), loc=2, borderaxespad=0, fontsize=18, frameon=False)
				plt.setp(l1.get_title(), fontsize=20)
				l1._legend_box.align="left"

				legends.append(l1)

				print(feh,mnfe)

			if authors[k] == 'sub(S18)':

				lines = []

				# Loop over some masses
				whichmasses = [1,2,3,4]
				#whichmasses = [0,2,3,4]
				for m in range(len(whichmasses)):

					# Set correct color
					colorid=m

					line, = ax.plot(feh, mnfe[whichmasses[m]], color=blues[colorid], linestyle=linestyles[k], label=str(mass[whichmasses[m]])+r'$M_{\odot}$', zorder=0, linewidth=1.5)
					lines.append(line)

				l1 = ax.legend(handles=lines, title='sub(S18)', bbox_to_anchor=(1.04,0.68), borderaxespad=0, fontsize=18, frameon=False)
				plt.setp(l1.get_title(), fontsize=20)
				l1._legend_box.align="left"

				legends.append(l1)

			if authors[k] == 'sub(B19)':

				lines = []

				# Loop over some masses
				whichmasses = [0,1,2,3,4]
				#whichmasses = [0,1,2]
				for m in range(len(whichmasses)):

					# Set correct color
					#colorid = [1,3,4]
					colorid = m

					line, = ax.plot(feh, mnfe[whichmasses[m]], color=blues[colorid], linestyle=linestyles[k], label=str(mass[whichmasses[m]])+r'$M_{\odot}$', zorder=0, linewidth=1.5)
					lines.append(line)

				l1 = ax.legend(handles=lines, title='sub(B19)', bbox_to_anchor=(1.04,0.31), loc=2, borderaxespad=0, fontsize=18, frameon=False)
				plt.setp(l1.get_title(), fontsize=20)
				l1._legend_box.align="left"

				legends.append(l1)

	# Make main legend
	if ia_comparison:
		# Put all legends on plot
		for legend in legends:
			ax.add_artist(legend)

	else:
		if literature is None:
			leg = plt.legend(patches, labels, fancybox=True, framealpha=0.5, loc='best', title='N = '+str(len(name)))
		else:
			leg = plt.legend(patches, labels, fancybox=True, framealpha=0.5, loc='best')
		for text in leg.get_texts():
			plt.setp(text, color='k', fontsize=18)
		plt.setp(leg.get_title(), fontsize=20)
		leg._legend_box.align="left"

	# Format plot
	#ax.set_title(title, fontsize=18)
	ax.set_xlabel('[Fe/H]', fontsize=24)
	ax.set_ylabel('[Mn/Fe]', fontsize=24)
	if ia_comparison:
		ax.set_ylabel(r'[Mn/Fe]$_{\mathrm{Ia}}$', fontsize=24)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(18)
	ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)

	# Output file
	plt.savefig(outfile+'_mnfe.pdf', bbox_inches='tight') #, transparent=True)
	plt.show()

	return

def get_theory(author, linear=False):
	""" Return theoretical model yields.

	Keywords:
	linear -- if 'True', return {Z, absolute Mn mass, Fe mass} instead of {[Fe/H], [Mn/Fe]}
	"""
	mnsolar = 5.43
	fesolar = 7.50

	if author=='sub(L19)':
		# Get [Fe/H]
		Z = np.array((0.,0.2,1.,2.,4.,6.,10.))/100.
		feh = np.log10(Z/0.02)

		# Set low end of [Fe/H]
		feh[0] = -2.1
		equivZ = 10.**(feh[0])*0.02

		# Get [Mn/Fe]
		mnmass = np.array((1.79e-3,1.95e-3,1.89e-3,2.28e-3,2.60e-3,3.85e-3,7.26e-3))
		fe54mass = np.array((8.76e-4,1.36e-3,3.59e-3,7.80e-3,1.23e-2,1.94e-2,3.96e-2))
		fe56mass = np.array((6.73e-1,6.69e-1,6.34e-1,6.10e-1,5.83e-1,5.57e-1,5.15e-1))
		fe57mass = np.array((1.51e-2,1.60e-2,1.83e-2,2.12e-2,2.52e-2,2.84e-2,3.30e-2))
		fe58mass = np.array((6.90e-6,6.90e-6,9.56e-6,4.39e-4,9.61e-6,1.12e-5,3.94e-6))
		fe60mass = np.array((1.6e-13,1.1e-13,5.65e-13,1.34e-9,5.63e-13,4.99e-10,1.88e-14))
		femass = np.sum((fe54mass,fe56mass,fe57mass,fe58mass,fe60mass), axis=0)
		feamu = (54.*fe54mass + 56.*fe56mass + 57.*fe57mass + 58.*fe58mass)/femass

		# Properly interpolate to the lowest end ([Fe/H]~-2)
		mnmass[0] = np.interp(equivZ,Z[:2],mnmass[:2])
		femass[0] = np.interp(equivZ,Z[:2],femass[:2])

		mnfe = np.log10((mnmass/55.)/(femass/feamu)) - mnsolar + fesolar

		mass = None

	if author=='sub(S18)':
		file = 'data/theoryyields/shen18_decayed.txt'

		# Get mass
		mass = np.array((0.8,0.85,0.9,1.0,1.1))

		# Get [Fe/H]
		Z = np.array((0.000,0.005,0.010,0.020))
		feh = np.log10(Z/0.02)

		# Set low end of [Fe/H]
		feh[0] = -2.1
		equivZ = 10.**(feh[0])*0.02

		# Get [Mn/Fe]
		mnmass = np.zeros((len(mass),len(feh)))
		femass = np.zeros((len(mass),len(feh)))
		feamu = np.zeros((len(mass),len(feh)))

		for i in range(len(feh)):
			data = pd.read_csv(file,delimiter='\s+',skiprows=8*i+3,nrows=5)

			mnmass[:,i] = np.asarray(data['55Mn'])
			fe54mass = np.asarray(data['54Fe'])
			fe56mass = np.asarray(data['56Fe'])
			fe57mass = np.asarray(data['57Fe'])
			fe58mass = np.asarray(data['58Fe'])

			femass[:,i] = np.sum((fe54mass,fe56mass,fe57mass,fe58mass), axis=0)
			feamu[:,i] = (54.*fe54mass + 56.*fe56mass + 57.*fe57mass + 58.*fe58mass)/femass[:,i]

		# Properly interpolate to the lowest end ([Fe/H]~-2)
		for i in range(len(mass)):
			#print(mnmass[i,:2])
			#print(Z[:2])
			mnmass[i,0] = np.interp(equivZ,Z[:2],mnmass[i,:2])
			femass[i,0] = np.interp(equivZ,Z[:2],femass[i,:2])

			# Test for metallicity lag due to SFH
			'''
			if mass[i] > 0.9 and mass[i] < 1.1:
				tests = np.array((-1.4,-1.5))
				testZ = np.power(10.,tests)*0.02
				for iz in testZ:
					testmn = np.interp(iz,Z[:2],mnmass[i,:2])
					testfe = np.interp(iz,Z[:2],femass[i,:2])
					testamu = (54.*np.interp(iz,Z[:2],fe54mass[:2]) + 56.*np.interp(iz,Z[:2],fe56mass[:2]) + 57.*np.interp(iz,Z[:2],fe57mass[:2]) + 58.*np.interp(iz,Z[:2],fe58mass[:2]))/testfe

					print('Test:',mass[i],iz,np.log10((testmn/55.)/(testfe/testamu)) - mnsolar + fesolar)
			'''

		mnfe = np.log10((mnmass/55.)/(femass/feamu)) - mnsolar + fesolar

	if author=='sub(B19)':
		file = 'data/theoryyields/Table_4_v2.txt'

		# Arrange datafile into a nice format
		data = pd.read_fwf(file, skiprows=20)
		data['He2'] = data['He']+data['2.1'].map(str)
		data = data.transpose()
		data.columns = data.iloc[-1]
		data = data[4:-1]

		# Get mass
		mass = np.array((0.88,0.97,1.06,1.10,1.15))

		# Get [Fe/H]
		Z = np.array((2.25E-4,2.25E-3,9.00E-3,2.25E-2,6.75E-2))
		feh = np.log10(Z/0.02)

		# Get [Mn/Fe]
		mnmass = np.zeros((len(mass),len(feh)))
		femass = np.zeros((len(mass),len(feh)))
		feamu = np.zeros((len(mass),len(feh)))

		for i in range(len(feh)):
			seq = i+5*np.arange(len(mass))
			mnmass[:,i] = np.asarray(data['Mn55'])[seq]
			#femass[:,i] = np.asarray(data['Fe'])[seq]

			fe54mass = np.asarray(data['Fe54'])[seq]
			fe56mass = np.asarray(data['Fe56'])[seq]
			fe57mass = np.asarray(data['Fe57'])[seq]
			fe58mass = np.asarray(data['Fe58'])[seq]

			femass[:,i] = np.sum((fe54mass,fe56mass,fe57mass,fe58mass), axis=0)
			feamu[:,i] = (54.*fe54mass + 56.*fe56mass + 57.*fe57mass + 58.*fe58mass)/femass[:,i]

		mnfe = np.log10((mnmass/55.)/(femass/feamu)) - mnsolar + fesolar

	if linear:
		return Z, mnmass, femass, mass

	return feh, mnfe, mass

def compute_fractions(feh_input, mnfe_ia, subchmodel, mass_input, mchmodel='S13_N100'):
	""" Compute what fraction of Type Ia SNe must be near-MCh vs sub-MCh based on [Mn/Fe]_{Ia}.

	Inputs:
	feh_input 	-- desired metallicity
	mnfe_ia 	-- [Mn/Fe]_{Ia}
	subchmodel 	-- which model to use for sub-MCh yields
					options: 'sub(L19)','sub(S18)'
	mass_input 	-- mass of model to use

	Keywords:
	mchmodel 	-- which model to use for near-MCh yields
					options: 'S13_N100'
	"""

	# Some constants
	mnsolar = 5.43
	fesolar = 7.50
	feamu = 55.8

	# Convert observed [Mn/Fe]_{Ia} to (Mn mass)/(Fe mass)
	mnfemass_obs = 10.**(mnfe_ia + mnsolar - fesolar) * 55. / feamu

	# Interpolate sub-MCh model to desired Z
	modelZ, mnmass_model, femass_model, mass = get_theory(subchmodel, linear=True)
	print('Model:',subchmodel)

	testZ 	= np.power(10.,feh_input)*0.02

	if subchmodel == 'sub(L19)':
		print('Mass:',1.1)
		mnmass_subch = np.interp(testZ,modelZ[:2],mnmass_model[:2])
		femass_subch = np.interp(testZ,modelZ[:2],femass_model[:2])
	else:
		idx = np.where((mass < (mass_input + 0.1)) & (mass > (mass_input - 0.1)))[0]
		print('Mass:',mass[idx])

		print(testZ,modelZ[:2],mnmass_model[idx,:2])
		mnmass_subch = np.interp(testZ,modelZ[:2],mnmass_model[idx,:2][0])
		femass_subch = np.interp(testZ,modelZ[:2],femass_model[idx,:2][0])

	# Assume a near-MCh model abundance
	if mchmodel == 'S13_N100':
		mnmass_mch = 9.29e-3
		femass_mch = 9.94e-2 + 6.22e-1 + 1.88e-2 + 8.02e-5

	# Compute fraction of Type Ia SNe that are sub-MCh
	A = (mnfemass_obs * femass_mch - mnmass_mch)/((mnmass_subch-mnmass_mch) + mnfemass_obs*(femass_subch - femass_mch))
	print('Fraction of Type Ia SNe that are sub-MCh:', A)
	print('Fraction of Type Ia SNe that are near-MCh:', 1.-A)

	return

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
		ax.set_xlim(-1.75, 0.75)
		ax.set_ylim(0.5, 7.5)

		plt.tick_params(axis='y', which='both', left=False, right=False)

		#ax.xaxis.set_ticklabels([])
		ax.yaxis.set_ticklabels(['DDT(S13)','def(F14)','DDT(L18)','def(L18)','sub(L19)','sub(S18)','sub(B19)',''][::-1])

	plt.figure(figsize=(8,8))

	# Make plot
	ax = plt.subplot(1,1,1)
	setup(ax)
	#ax.xaxis.set_major_locator(ticker.AutoLocator())
	#ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
	#ax.text(0.0, 0.1, "AutoLocator()", fontsize=14, transform=ax.transAxes)
	ax.set_xlabel(r'$\mathrm{[Mn/Fe]}_{\mathrm{Ia}}$', fontsize=20)

	dy = 0.02 # distance between lines

	# Plot observed [Mn/Fe]
	#ax.errorbar(-0.28, 8, xerr=0.03, color='k', marker='o', linestyle='', markersize=8, zorder=100)
	ax.axvspan(xmin = -0.30 - 0.03, xmax = -0.30 + 0.03, color='gray', alpha=0.5)
	ax.text(-0.30 - 0.13, 6.5, 'This work', rotation = 90, fontsize=18)

	# Plot models
	
	reds = plt.cm.Reds_r(np.linspace(0.,0.8,8))
	mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', reds)
	redcwheel = [np.array(mpl.rcParams['axes.prop_cycle'])[x]['color'] for x in range(8)]

	blues = plt.cm.Blues_r(np.linspace(0.,0.8,8))
	mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', blues)
	bluecwheel = [np.array(mpl.rcParams['axes.prop_cycle'])[x]['color'] for x in range(8)]

	# DDT (S13)
	ddts13 = [0.01, -0.06, 0.01, 0.23, 0.50, 0.53][::-1]
	for i in range(len(ddts13)):
		ax.axvline(ddts13[i], ymin = 6./7. + dy, ymax = 7./7. - dy, color=redcwheel[i], alpha=1, linewidth=2)

	# def(F14)
	deff14 = [0.36, 0.42, 0.44, 0.48, 0.50, 0.52][::-1]
	for i in range(len(deff14)):
		ax.axvline(deff14[i], ymin = 5./7. + dy, ymax = 6./7. - dy, color=redcwheel[i], alpha=1, linewidth=2)

	# DDT(L18)
	ddtl18 = [-0.03, 0.18, 0.31][::-1]
	for i in range(len(ddtl18)):
		ax.axvline(ddtl18[i], ymin = 4./7. + dy, ymax = 5./7. - dy, color=redcwheel[i], alpha=1, linewidth=2)
	ax.axvline(0.15, ymin = 4./7. + dy, ymax = 5./7. - dy, color='r', linestyle=':', alpha=1, linewidth=2)

	# def(L18)
	defl18 = [0.19, 0.39, 0.39][::-1]
	for i in range(len(defl18)):
		ax.axvline(defl18[i], ymin = 3./7. + dy, ymax = 4./7. - dy, color=redcwheel[i], alpha=1, linewidth=2)
	ax.axvline(0.33, ymin = 3./7. + dy, ymax = 4./7. - dy, color='r', linestyle=':', alpha=1, linewidth=2)

	# sub(L19)
	subl19 = [0.25,-0.13,-0.23,-0.25,-0.45,-0.34][::-1]
	for i in range(len(subl19)):
		ax.axvline(subl19[i], ymin = 2./7. + dy, ymax = 3./7. - dy, color=redcwheel[i], alpha=1, linewidth=2)
	ax.axvline(-0.44, ymin = 2./7. + dy, ymax = 3./7. - dy, color=bluecwheel[0], linestyle=':', alpha=1, linewidth=2)

	# sub(S18)
	subs18 = [-0.64,-0.75,-1.05,-1.33][::-1]
	for i in range(len(subs18)):
		ax.axvline(subs18[i], ymin = 1./7. + dy, ymax = 2./7. - dy, color=blues[i], alpha=1, linewidth=2)
	subs18_special = [-0.55,-0.73,-1.00,-1.26][::-1]
	for i in range(len(subs18_special)):
		ax.axvline(subs18_special[i], ymin = 1./7. + dy, ymax = 2./7. - dy, color=blues[i], alpha=1, linewidth=2, linestyle=':')

	# sub(B19)
	subb19 = [-0.55,-0.81,-1.16,-1.28,-1.42][::-1]
	for i in range(len(subb19)):
		ax.axvline(subb19[i], ymin = 0./7. + dy, ymax = 1./7. - dy, color=blues[i], alpha=1, linewidth=2)
	subb19_special = [-0.50,-0.81,-1.16,-1.28,-1.42][::-1]
	for i in range(len(subb19_special)):
		ax.axvline(subb19_special[i], ymin = 0./7. + dy, ymax = 1./7. - dy, color=blues[i], alpha=1, linewidth=2, linestyle=':')

	'''
	ax.axvline(0.18, color='#45ADA8', linestyle=':', linewidth=6, label='DDT(T16)')
	ax.axvspan(0.01, 0.53, color='#B0B0B0', hatch='\\', alpha=0.6, label='DDT(S13)')
	ax.axvspan(0.36, 0.52, color='#45ADA8', hatch='//', alpha=0.75, label='Def(F14)')
	ax.axvspan(-1.69, -1.21, color='#9DE0AD', alpha=0.8, label='Sub(B)')
	ax.axvspan(-1.52, -0.68, color='#547980', hatch='//', alpha=0.5, label='Sub(S18)')
	'''

	# Line separating near vs sub Mch models
	ax.axhline(3.5, color='k', linestyle='--')
	ax.text(-1.5,3.5+0.1,r'Near-$M_{\mathrm{Ch}}$', fontsize=18)
	ax.text(-1.5,3.5-0.4,r'Sub-$M_{\mathrm{Ch}}$', fontsize=18)

	# NLTE arrow
	plt.arrow(-0.30 + 0.03, 6, 0.33, 0, head_width=0.1, head_length=0.05, color='gray')
	ax.text(-0.30 + 0.05, 6.1, 'NLTE?', fontsize=16, color='gray')

	# Format plot
	for label in (ax.get_xticklabels()):
		label.set_fontsize(16)
	for label in (ax.get_yticklabels()):
		label.set_fontsize(18)
	ax.tick_params(direction='in', bottom=True, top=True, left=False, right=False)

	# Output file
	plt.savefig(outfile, bbox_inches='tight', transparent=True)
	plt.show()

	return

def main():

	# Plot for Sculptor
	#fit_mnfe_feh('data/bscl5_1200B_final3.csv',False,'figures/scl_fit3', 'Sculptor dSph', fehia=-2.12, maxerror=0.3, bestfit=True)
	#fit_mnfe_feh('data/bscl5_1200B_final3.csv',False,'figures/scl_fit3', 'Sculptor dSph', fehia=-2.12, maxerror=0.3, sne=True, literature=['Sobeck+06','North+12'])
	#fit_mnfe_feh('data/bscl5_1200B_final3.csv',False,'figures/scl_fit3_nlte', 'Sculptor dSph', fehia=-2.12, maxerror=0.3, nlte=True) 
	#fit_mnfe_feh(['data/bscl5_1200B_final3.csv','data/hires_data_final/scl/north12_final.csv'],[False,True],'figures/scl_fit_total', 'Sculptor dSph', fehia=-2.34, maxerror=0.3, gratings=['#594F4F','#B0B0B0'])

	# Plot for Ursa Minor
	#fit_mnfe_feh(['data/bumia_1200B_final3.csv'],[False],'figures/umi_fit3', 'Ursa Minor dSph', fehia=-2.42, maxerror=0.3, gratings=['#594F4F'])

	# Plot [Mn/Fe] values on number line
	#compare_mnfe('figures/scl_mnfe_comparison.pdf')

	# Z-dep comparison
	#fit_mnfe_feh('data/bscl5_1200B_final3.csv',False,'figures/scl_fit3', 'Sculptor dSph', fehia=-2.12, maxerror=0.3, ia_comparison=True)

	# Compute fractions of Type Ia SNe
	compute_fractions(-2., -0.30, 'sub(L19)', 1.1, mchmodel='S13_N100')
	compute_fractions(-1., -0.29, 'sub(S18)', 1.0, mchmodel='S13_N100')
	compute_fractions(-2., -0.30, 'sub(S18)', 1.0, mchmodel='S13_N100')

	return

if __name__ == "__main__":
	main()