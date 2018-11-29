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
from matplotlib.ticker import NullFormatter
from statsmodels.stats.weightstats import DescrStatsW

def fit_mnfe_feh(filenames, outfile, title, fehia, maxerror=None, gratings=None):
	"""Use model described in Kirby (in prep.) to determine Type Ia supernova yields for [Mn/H]

	Inputs:
	filename 	-- list of input filenames
	outfile 	-- name of output file
	title 		-- title of graph
	fehia 		-- [Fe/H] when Type Ia SNe turn on

	Keywords:
	maxerror 	-- if not None, points with error > maxerror will not be used in computation
	gratings 	-- if not None, must be list of gratings used for input filenames.
					Plot points from different gratings in different colors.
	"""

	# Get data from files
	name 	= []
	feh 	= []
	feherr 	= []
	mnh 	= []
	mnherr	= []
	colors  = []
	redchisq = []

	for i in range(len(filenames)):

		file = filenames[i]
		current_name 	= np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')

		data = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=[5,6,8,9,10])
		current_feh 	= data[:,0]
		current_feherr 	= data[:,1]
		current_mnh 	= data[:,2]
		current_mnherr 	= data[:,3]
		current_redchisq = data[:,4]

		# Append to total data arrays
		name.append(current_name)
		feh.append(current_feh)
		feherr.append(current_feherr)
		mnh.append(current_mnh)
		mnherr.append(current_mnherr)
		redchisq.append(current_redchisq)

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

	# Compute [Mn/Fe]
	mnfe = mnh - feh
	mnfeerr = np.sqrt(np.power(feherr,2.)+np.power(mnherr,2.))

	outlier = np.where(mnfe > 0)[0]
	print(name[outlier])
	notoutlier = np.ones(len(mnfe), dtype='bool')
	notoutlier[outlier] = False

	# Remove points with error > maxerror
	if maxerror is not None:
		mask 	= np.where((mnfeerr < maxerror)) # & notoutlier) # & (redchisq < 3.0))
		name 	= name[mask]
		feh 	= feh[mask]
		mnfe 	= mnfe[mask]
		mnfeerr = mnfeerr[mask]
		colors  = colors[mask]

	# Fit a simple model!

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

	# Sample the log-probability function using emcee
	# Initialize the walkers
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

	# Create figure
	fig, ax = plt.subplots(figsize=(10,5))

	# Plot other lines
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

	if typei:
		#xmin=(-2.34+3)/2.5, 
		ax.axhline(0.18, color='b', linestyle='dashed', label='DDT(T16)')
		ax.axhspan(0.01, 0.53, color='g', hatch='\\', alpha=0.2, label='DDT(S13)')
		ax.axhspan(0.36, 0.52, color='darkorange', hatch='//', alpha=0.3, label='Def(F14)')
		ax.axhspan(-1.69, -1.21, color='r', alpha=0.2, label='Sub(B)')
		ax.axhspan(-1.52, -0.68, color='b', hatch='//', alpha=0.2, label='Sub(S18)')

	if solar:
		ax.axhline(0, color='gold', linestyle='solid')
	'''

	# Scatter plot
	#area = 2*np.reciprocal(np.power(mnfeerr,2.))
	#ax.scatter(feh, mnfe, s=area, c=colors, alpha=0.5, zorder=100) #, label='N = '+str(len(name)))
	ax.errorbar(feh, mnfe, yerr=mnfeerr, color='k', marker='o', linestyle='', capsize=3, zorder=100)
	ax.text(0.025, 0.9, 'N = '+str(len(name)), transform=ax.transAxes, fontsize=14)

	#for i in range(len(outlier)):
	#	idx = outlier[i]
	#	plt.text(feh[idx], mnfe[idx], name[idx])

	# Plot best fit
	xfit = np.linspace(np.min(feh), ax.get_xlim()[1], 100)
	mnfe_cc = (bperp + fehia*np.sin(theta))/np.cos(theta)

	yfit = np.zeros((3, len(xfit)))
	for i in range(len(xfit)):
		if xfit[i] <= fehia:
			yfit[:,i] = mnfe_cc
		else:
			yfit[:,i] = xfit[i]*np.tan(theta) + bperp/(np.cos(theta))

	ax.fill_between(xfit, yfit[2], yfit[0], color='r', alpha=0.25)
	ax.plot(xfit, yfit[1], 'r-', linewidth=2)

	# Determine the Type Ia yield of Mn

	# Start by determining the amount of Mg/Fe predicted from model
	theta_mg = -0.50 # XXX turn this into upper/median/lower sequence
	bperp_mg = -0.64 # XXX turn this into upper/median/lower sequence
	mgfe = xfit*np.tan(theta_mg) + bperp_mg/(np.cos(theta_mg))

	# Compute f_Ia
	mgfe_cc = 0.55
	mgfe_ia = -1.5
	frac_ia = (mgfe_cc - mgfe) / (mgfe - mgfe_ia)

	# Compute Mn yield based on best-fit model!
	mnfe_ia = np.zeros((3, len(xfit)))
	for i in range(3):
		for j in range(len(xfit)):
			mnfe_ia[i,j] = (frac_ia[j] + 1.)/(frac_ia[j]) * yfit[i,j] - (1./frac_ia[j] * mnfe_cc[i])

	# Plot it!
	mask = np.where(xfit > fehia)
	ax.fill_between(xfit[mask], mnfe_ia[2][mask], mnfe_ia[0][mask], color='g', alpha=0.25)
	ax.plot(xfit[mask], mnfe_ia[1][mask], 'g-', linewidth=2)

	# Format plot
	ax.set_title(title, fontsize=18)
	ax.set_xlabel('[Fe/H]', fontsize=16)
	ax.set_ylabel('[Mn/Fe]', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)

	ax.set_xlim([-3,-0.75])
	#ax.set_ylim([-2,2])
	ax.set_ylim([-1.5,1.1])
	#plt.legend(loc='best')

	# Output file
	plt.savefig(outfile+'_mnfe.png', bbox_inches='tight')
	plt.show()

def main():

	# Sculptor 1200B
	fit_mnfe_feh(['data/scl5_1200B.csv'],'figures/scl_fit', 'Sculptor dSph', -2.34, maxerror=None, gratings=['k'])

	return

if __name__ == "__main__":
	main()