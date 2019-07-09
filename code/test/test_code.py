# test_code.py
# Short programs to test code
#
# Created 20 Sept 18
# Updated 17 May 19
###################################################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from run_moog import runMoog
from smooth_gauss import smooth_gauss
from match_spectrum import open_obs_file, smooth_gauss_wrapper
from continuum_div import get_synth
import scipy.optimize
from astropy.io import fits
import chi_sq

def arcturus_test(resolution, temp, logg, fe, alpha, mn, fit=False):
	""" Degrade Arcturus spectrum to DEIMOS resolution
		and compare to synthetic spectrum with known Arcturus abundance.

	Inputs:
	resolution 	-- DEIMOS resolution
	temp 		-- known Arcturus T_eff
	logg 		-- known Arcturus log(g)
	fe 			-- known Arcturus [Fe/H]
	alpha 		-- known Arcturus [alpha/Fe]
	mn 			-- known Arcturus abundance

	Keywords:
	fit 		-- if 'False' (default), just compare w/ synthetic spectrum with known abundance;
					else, try to fit Arcturus spectrum and find best-fit Mn abundance
	"""

	# Open Arcturus spectrum
	hdu1 = fits.open('/raid/m31/solspec/ardata.fits')
	data = hdu1[1].data

	obswvl_orig  = data['WAVELENGTH']
	obsflux_orig = data['ARCTURUS']  	# Continuum-normalized by Hinkle et al.

	# Truncate arrays to fit within typical wavelength array
	mask = np.where(((obswvl_orig > 4501.) & (obswvl_orig < 7000.)))
	obsflux_orig = obsflux_orig[mask]
	obswvl_orig = obswvl_orig[mask]

	# Smooth observed spectrum to DEIMOS resolution
	obswvl = np.arange(4500.,7001.,0.3115234375) # typical wavelength array for a 1200B observation
	#obswvl = np.arange(4500.,7000.,0.40869140625) # typical wavelength array for a 900ZD observation
	obsflux = smooth_gauss_wrapper(obswvl_orig, obsflux_orig, obswvl, resolution)

	#print(obswvl_orig[0], obswvl_orig[-1])
	#print(obswvl[0], obswvl[-1])

	# Make up some errors
	sigma = 0.001
	fe_err = 0.02

	# Split observed spectrum into different regions
	lines = np.array([[4729.,4793.],[4813.,4833.],[5384.,5442.],[5506.,5547.],[6003.,6031.],[6374.,6394.],[6481.,6501.]])

	obswvl_fit = []
	obsflux_fit = []
	ivar_fit = []
	dlam_fit = []

	for line in range(len(lines)):
		linemask = np.where(((obswvl > lines[line][0]) & (obswvl < lines[line][1])))
		obswvl_fit.append( obswvl[linemask] )
		obsflux_fit.append( obsflux[linemask] )
		ivar_fit.append( np.ones(len(obswvl_fit[line]))*1./(sigma**2.) )
		#dlam_fit.append( np.ones(len(linemask))*resolution )

	if fit == False:

		# Produce synthetic spectrum
		synth = runMoog(temp=temp, logg=logg, fe=fe, alpha=alpha, elements=[25], abunds=[mn], solar=[5.43], lines='new')
		synthflux = []
		for i in range(len(synth)):

			# Loop over each line
			synthregion = synth[i]

			# Smooth each region of synthetic spectrum to match each region of continuum-normalized observed spectrum
			newsynth = get_synth(obswvl_fit[i], obsflux_fit[i], np.zeros(len(synthregion)), resolution, synth=synthregion)
			synthflux.append(newsynth)

		obswvl = np.hstack(obswvl_fit[:])
		obsflux = np.hstack(obsflux_fit[:])
		synthflux = np.hstack(synthflux[:])

		# Plot observed and synthetic spectra
		title = 'Arcturus: [Mn/H] = '+str(mn)
		chi_sq.make_plots(lines='new', specname='Arcturus', obswvl=obswvl, obsflux=obsflux, synthflux=synthflux, outputname='/raid/madlr/test', title='Arcturus')

		return

	else:

		# Set up obsspecial keyword
		skip = np.asarray(np.arange(len(obswvl_fit)))
		dlam_fit = np.ones(len(obswvl_fit))*resolution
		obsspecial = [np.asarray(obsflux_fit), np.asarray(obswvl_fit), np.asarray(ivar_fit), np.asarray(dlam_fit), skip, temp, logg, fe, alpha, fe_err]

		# Run fitting algorithm on Arcturus spectrum
		test = chi_sq.obsSpectrum('/raid/m31/solspec/ardata.fits', '/raid/m31/solspec/ardata.fits', 0, True, 'Arcturus', 'Arcturus-fit', False, 'new', obsspecial=obsspecial, plot=True).plot_chisq(mn)

def plot_vel(filename, outputname):
	"""Get velocities from moogify files, plot them in a histogram, and output them to a text file.

    Inputs:
    filename - name of file to open
    outputname - name of output file listing IDs

    Keywords:

    Outputs:
    """

    # Get data
	print('Opening ', filename)
	hdu1 = fits.open(filename)
	data = hdu1[1].data

	namearray = data['OBJNAME']
	wavearray = data['LAMBDA']
	fluxarray = data['SPEC']
	ivararray = data['IVAR']
	dlamarray = data['DLAM']

	velarray  = data['VR']
	velerrs   = data['VRERR']

	avg = np.average(velarray)
	stdev = np.std(velarray)

	# Throw out outliers
	def cut_outliers(velarray, velerrs, avg, stdev, namearray):
		''' Cut out outliers in velocity'''

		goodvel = np.where((velarray < (avg + 2.*stdev)) & (velarray > (avg - 2.*stdev)))[0]
		velarray = velarray[goodvel]
		velerrs = velerrs[goodvel]
		namearray = namearray[goodvel]

		avg = np.average(velarray)
		stdev = np.std(velarray)

		return velarray, velerrs, avg, stdev, namearray

	velarray, velerrs, avg, stdev, namearray = cut_outliers(velarray,velerrs,avg,stdev,namearray)
	velarray, velerrs, avg, stdev, namearray = cut_outliers(velarray,velerrs,avg,stdev,namearray)

	# Plot velocities
	plt.hist(velarray, bins=10)
	plt.axvline(avg, color='r', linestyle='--')
	plt.axvspan(avg-stdev, avg+stdev, color='r', alpha=0.5)
	plt.show()

	# Print names and velocities of stars for a later membership check
	with open(outputname, 'w+') as f:
		for i in range(len(namearray)):
			f.write(namearray[i]+'\n')

	return

def main():
	
	# DEIMOS resolutions
	#resolution = 0.47787237 # 1200B
	#resolution = 0.76595747 # 900ZD

	# Arcturus parameters from Ramirez & Allende Prieto (2011)
	# Note: Arcturus [Mn/H] = -0.73
	#arcturus_test(resolution=resolution, temp=4286, logg=1.66, fe=-0.52, alpha=0.4, mn=[-0.52], fit=True)

	# Plot velocity distributions for GCs
	#plot_vel('/raid/caltech/moogify/n5024b_1200B/moogify.fits.gz','/raid/madlr/glob/n5024/n5024b_1200B_velmembers.txt')
	plot_vel('/raid/caltech/moogify/7078l1_1200B/moogify.fits.gz','/raid/madlr/glob/n7078/7078l1_1200B_velmembers.txt')
	plot_vel('/raid/caltech/moogify/7089l1_1200B/moogify.fits.gz','/raid/madlr/glob/n7089/7089l1_1200B_velmembers.txt')

	return

if __name__ == "__main__":
	main()