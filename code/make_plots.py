# make_plots.py
# Make plots for other codes
# 
# Created 9 Nov 18
# Updated 9 Nov 18
###################################################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import math

# Code to make plots
def make_plots(lines, specname, obswvl, obsflux, synthflux, outputname, resids=True, ivar=None, title=None, synthfluxup=None, synthfluxdown=None):
	"""Make plots.

	Inputs:
	lines -- which linelist to use? Options: 'new', 'old'
	specname -- name of star
	obswvl 	-- observed wavelength array
	obsflux -- observed flux
	synthflux 	-- synthetic flux
	outputname 	-- where to output file

	Keywords:
	resids  -- plot residuals if 'True' (default); else, don't plot residuals
	ivar 	-- inverse variance; if 'None' (default), don't plot errorbars
	title 	-- plot title; if 'None' (default), then plot title = "Star + ID"
	synthfluxup & synthfluxdown -- if not 'None' (default), then plot synthetic spectrum as region between [Mn/H]_best +/- 0.3dex

	Outputs:
	"""

	# Define lines to plot
	if lines == 'new':
		#linelist = np.array([4739.,4754.,4761.5,4765.5,4783.,4823.,5394.,5399.,
		#					 5407.,5420.,5432.,5516.,5537.,6013.,6016.,6021.,6384.,6491.])
		linelist = np.array([4739.,4754.,4761.5,4765.5,4783.,4823.,5394.,
							 5407.,5420.,5432.,5516.,5537.,6013.,6016.,6021.])
		linewidth = np.array([1.,1.,1.5,1.5,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])

		nrows = 3
		ncols = 5
		#figsize = (24,12)
		figsize = (20,12)

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
	#f, axes = plt.subplots(nrows, ncols, sharey='row', num=1, figsize=figsize)
	plt.figure(num=1, figsize=figsize)
	plt.title(title)

	# Plot showing residuals
	if resids:
		plt.figure(num=2, figsize=figsize)
		plt.title(title)

	# Plot showing ivar
	if ivar is not None:
		plt.figure(num=3, figsize=figsize)
		plt.title(title)

	for i in range(len(linelist)):
		#f = plt.figure(1)
		#for i, ax in enumerate(f.axes):

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
				with plt.rc_context({'axes.linewidth':4, 'axes.edgecolor':'#594F4F', 'xtick.color':'#594F4F', 'ytick.color':'#594F4F'}):
					plt.figure(1)
					plt.subplot(nrows,ncols,i+1)
					plt.axvspan(linelist[i] - linewidth[i], linelist[i] + linewidth[i], color='#45ADA8', zorder=1)
					if (synthfluxup is not None) and (synthfluxdown is not None):
						plt.fill_between(obswvl[mask], synthfluxup[mask], synthfluxdown[mask], facecolor='red', edgecolor='red', alpha=0.75, linewidth=3, label='Synthetic', zorder=2)
					else:
						plt.plot(obswvl[mask], synthflux[mask], 'r-', label='Synthetic')
					plt.errorbar(obswvl[mask], obsflux[mask], yerr=yerr, color='#594F4F', fmt='o', markersize=8, elinewidth=2, label='Observed', zorder=3)

					plt.xticks([linelist[i]], fontsize=18)
					plt.yticks(fontsize=16)

					plt.xlim((lolim, uplim))
					plt.ylim((0.75, 1.10))

					if i==0:
						leg = plt.legend(fancybox=True, framealpha=0.5, loc='best')
						for text in leg.get_texts():
							plt.setp(text, color='#594F4F', fontsize=18)

				if resids:
					# Only plot residuals if synth spectrum has been smoothed to match obswvl
					plt.figure(2)
					plt.subplot(nrows,ncols,i+1)
					plt.axvspan(linelist[i] - linewidth[i], linelist[i] + linewidth[i], color='green', alpha=0.25)
					plt.errorbar(obswvl[mask], obsflux[mask] - synthflux[mask], yerr=yerr, color='k', fmt='o', label='Residuals')
					plt.axhline(0, color='r', linestyle='solid', label='Zero')

				if ivar is not None:
					# Plot ivar
					plt.figure(3)
					plt.subplot(nrows,ncols,i+1)
					plt.axvspan(linelist[i] - linewidth[i], linelist[i] + linewidth[i], color='green', alpha=0.25)
					plt.errorbar(obswvl[mask], ivar[mask], color='k', linestyle='-')
					#plt.axhline(0, color='r', linestyle='solid', label='Zero')

		except:
			#ax.set_visible(False)
			continue

	# Legend for plot showing fits
	fig = plt.figure(1)
	#fig.text(0.5, 0.04, 'Wavelength (A)', fontsize=18, ha='center', va='center', color='#594F4F')
	#fig.text(0.06, 0.5, 'Relative flux', fontsize=18, ha='center', va='center', rotation='vertical', color='#594F4F')
	#plt.ylabel('Relative flux')
	#plt.xlabel('Wavelength (A)')

	plt.savefig(outputname+'/'+specname+'finalfits.png',bbox_inches='tight',transparent=False)
	plt.close(1)

	if resids:
		fig2 = plt.figure(2)
		fig2.text(0.5, 0.04, 'Wavelength (A)', fontsize=18, ha='center', va='center')
		fig2.text(0.06, 0.5, 'Residuals', fontsize=18, ha='center', va='center', rotation='vertical')
		plt.legend(loc='best')
		plt.savefig(outputname+'/'+specname+'resids.png',bbox_inches='tight')
		plt.close(2)

	if ivar is not None:
		fig3 = plt.figure(3)
		fig3.text(0.5, 0.04, 'Wavelength (A)', fontsize=18, ha='center', va='center')
		fig3.text(0.06, 0.5, 'Inverse variance', fontsize=18, ha='center', va='center', rotation='vertical')
		#plt.legend(loc='best')
		plt.savefig(outputname+'/'+specname+'ivar.png',bbox_inches='tight')
		plt.close(3)

	return