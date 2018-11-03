# output_mn.py
# Gets Mn abundances for a list of stars
#
# Created 5 June 18
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
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas
import scipy.optimize
import chi_sq

def run_chisq(filename, paramfilename, galaxyname, slitmaskname, startstar=0, globular=False, lines='new'):
	""" Measure Mn abundances from a FITS file.

	Inputs:
	filename 		-- file with observed spectra
	paramfilename 	-- file with parameters of observed spectra
	galaxyname		-- galaxy name, options: 'scl'
	slitmaskname 	-- slitmask name, options: 'scl1'

	Keywords:
	startstar		-- if 0 (default), start at beginning of file and write new datafile;
						else, start at #startstar and just append to datafile
	globular 		-- if 'False' (default), put into output path of galaxy;
						else, put into globular cluster path
	lines 			-- if 'new' (default), use new revised linelist;
						else, use original linelist from Judy's code

	"""

	# Output filename
	if globular:
		outputname = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'.csv'
	else:
		outputname = '/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname+'.csv'

	# Open new file
	if startstar<1:
		with open(outputname, 'w+') as f:
			f.write('Name\tRA\tDec\tTemp\tlog(g)\t[Fe/H]\terror([Fe/H])\t[alpha/Fe]\tdlam\terror(dlam)\t[Mn/H]\terror([Mn/H])\tchisq(reduced)\n')

	# Get number of stars in file
	Nstars = open_obs_file(filename)

	# Get coordinates of stars in file
	RA, Dec = open_obs_file(filename, coords=True)

	# Run chi-sq fitting for all stars in file
	starname = []
	startemp = []
	starlogg = []
	starfe	 = []
	staralpha = []
	starmn 	 = []
	starmnerr = []
	for i in range(startstar, Nstars):

		try:
			# Get metallicity of star to use for initial guess
			print('Getting initial metallicity')
			temp, logg, fe, alpha, fe_err = open_obs_file(filename, retrievespec=i, specparams=True)

			# Get dlam (FWHM) of star to use for initial guess
			specname, obswvl, obsflux, ivar, dlam, zrest = open_obs_file(filename, retrievespec=i)

			if np.isclose(temp, 4750.) and np.isclose(fe,-1.5) and np.isclose(alpha,0.2):
				print('Bad parameter measurement! Skipped #'+str(i+1)+'/'+str(Nstars)+' stars')
				continue

			# Run optimization code
			star = chi_sq.obsSpectrum(filename, paramfilename, i, True, galaxyname, slitmaskname, globular, lines)
			#best_mn, error = star.minimize_scipy(fe)
			best_mn, error, finalchisq = star.plot_chisq(fe, dlam)

		except Exception as e:
			print(repr(e))
			print('Skipped star #'+str(i+1)+'/'+str(Nstars)+' stars')
			continue

		print('Finished star '+star.specname, '#'+str(i+1)+'/'+str(Nstars)+' stars')

		with open(outputname, 'a') as f:
			f.write(star.specname+'\t'+str(RA[i])+'\t'+str(Dec[i])+'\t'+str(star.temp)+'\t'+str(star.logg[0])+'\t'+str(star.fe[0])+'\t'+str(star.fe_err[0])+'\t'+str(star.alpha[0])+'\t'+str(best_mn[1])+'\t'+str(error[1])+'\t'+str(best_mn[0])+'\t'+str(error[0])+'\t'+str(finalchisq)+'\n')

	return

def match_hires(hiresfile, obsfile):
	"""Measure Mn abundances for stars that have hi-resolution measurements.

	Inputs:
	hiresfile 	-- name of hi-resolution catalog
	obsfile 	-- name of not-hi-res catalog
	"""

	# Output filename
	outputname = hiresfile[:-4]+'_matched.csv'

	# Get RA and Dec of stars from (lit) hi-res datafile
	hiresRA = np.genfromtxt(hiresfile, skip_header=1, delimiter='\t', usecols=2, dtype='str')
	hiresDec = np.genfromtxt(hiresfile, skip_header=1, delimiter='\t', usecols=3, dtype='str')

	hiresabundance = np.genfromtxt(hiresfile, skip_header=1, delimiter='\t', usecols=8, dtype='str')

	# Get RA and Dec of stars from (our) med-res file
	medresRA, medresDec = open_obs_file(obsfile, coords=True)
	medrescatalog = SkyCoord(ra=medresRA*u.degree, dec=medresDec*u.degree) 

	# Initialize arrays to hold stuff
	medresMn 	  = np.zeros(len(hiresRA))
	medresMnerror = np.zeros(len(hiresRA))
	hiresMn 	  = np.zeros(len(hiresRA))
	hiresMnerror  = np.zeros(len(hiresRA))

	# Loop over each star from hires list
	medres_starnum = np.zeros(len(hiresRA), dtype='int')
	for i in range(len(hiresRA)):

		# Convert coordinates to decimal form
		coord = SkyCoord(hiresRA[i], hiresDec[i], unit=(u.hourangle, u.deg))

		# Search for matching star in our catalog 
		idx, sep, _ = coord.match_to_catalog_sky(medrescatalog) 
		print('Separation: ', sep.arcsecond)
		print('ID: ', idx)

		if sep.arcsec < 10:
			print('Got one!')

			# Get metallicity of this matching star to use for initial guess
			temp, logg, fe, alpha, fe_err = open_obs_file(obsfile, retrievespec=idx, specparams=True)

			# Measure med-res Mn abundance of the star
			medresMn[i], medresMnerror[i] = chi_sq.obsSpectrum(obsfile, idx).minimize_scipy(fe)

			# Get hi-res Mn abundance
			hiresMn[i] = float(hiresabundance[i][:5])
			hiresMnerror[i] = float(hiresabundance[i][-4:])

			np.savetxt(outputname, np.asarray((medresMn, medresMnerror, hiresMn, hiresMnerror, sep.arcsec)).T, delimiter='\t', header='our [Mn/H]\terror(our [Mn/H])\thires [Mn/H]\terror(hires [Mn/H])\tseparation (arcsec)')

	return medresMn, medresMnerror, hiresMn, hiresMnerror, sep.arcsec

def make_chisq_plots(filename, paramfilename, galaxyname, slitmaskname, startstar=0, globular=False):
	""" Plot chisq contours for stars whose [Mn/H] abundances have already been measured.

	Inputs:
	filename 		-- file with observed spectra
	paramfilename 	-- file with parameters of observed spectra
	galaxyname		-- galaxy name, options: 'scl'
	slitmaskname 	-- slitmask name, options: 'scl1'

	Keywords:
	globular 		-- if 'False', put into output path of galaxy; else, put into globular cluster path

	"""

	# Input filename
	if globular:
		file = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'.csv'
	else:
		file = '/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname+'.csv'

	name  = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')
	mn    = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=10)
	mnerr = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=11)
	dlam = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=8)
	dlamerr = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=9)

	# Get number of stars in file with observed spectra
	Nstars = open_obs_file(filename)

	# Plot chi-sq contours for each star
	for i in range(startstar, Nstars):

		try:

			# Check if parameters are measured
			temp, logg, fe, alpha, fe_err = open_obs_file(filename, retrievespec=i, specparams=True)
			if np.isclose(1.5,logg) and np.isclose(fe,-1.5) and np.isclose(fe_err, 0.0):
				print('Bad parameter measurement! Skipped #'+str(i+1)+'/'+str(Nstars)+' stars')
				continue

			# Open star
			star = chi_sq.obsSpectrum(filename, paramfilename, i, False, galaxyname, slitmaskname, globular)

			# Check if star has already had [Mn/H] measured
			if star.specname in name:

				# If so, plot chi-sq contours if error is < 1 dex
				idx = np.where(name == star.specname)
				if mnerr[idx][0] < 1:
					params0 = [[mn[idx][0], dlam[idx][0]],[mnerr[idx][0],dlamerr[idx][0]]]
					best_mn, error = star.plot_chisq(params0, minimize=False)

		except Exception as e:
			print(repr(e))
			print('Skipped star #'+str(i+1)+'/'+str(Nstars)+' stars')
			continue

		print('Finished star '+star.specname, '#'+str(i+1)+'/'+str(Nstars)+' stars')

	return

def plot_fits_postfacto(filename, paramfilename, galaxyname, slitmaskname, startstar=0, globular=False):
	""" Plot fits and residuals for stars whose [Mn/H] abundances have already been measured.

	Inputs:
	filename 		-- file with observed spectra
	paramfilename 	-- file with parameters of observed spectra
	galaxyname		-- galaxy name, options: 'scl'
	slitmaskname 	-- slitmask name, options: 'scl1'

	Keywords:
	globular 		-- if 'False', put into output path of galaxy; else, put into globular cluster path

	"""

	# Input filename
	if globular:
		file = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'.csv'
	else:
		file = '/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname+'.csv'

	name  = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')
	mn    = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=10)
	mnerr = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=11)
	dlam = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=8)
	dlamerr = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=9)

	# Get number of stars in file with observed spectra
	Nstars = open_obs_file(filename)

	# Plot chi-sq contours for each star
	for i in range(startstar, Nstars):

		try:

			# Check if parameters are measured
			temp, logg, fe, alpha, fe_err = open_obs_file(filename, retrievespec=i, specparams=True)
			if np.isclose(1.5,logg) and np.isclose(fe,-1.5) and np.isclose(fe_err, 0.0):
				print('Bad parameter measurement! Skipped #'+str(i+1)+'/'+str(Nstars)+' stars')
				continue

			# Open star
			star = chi_sq.obsSpectrum(filename, paramfilename, i, False, galaxyname, slitmaskname, globular)

			# Check if star has already had [Mn/H] measured
			if star.specname in name:

				# If so, open data file for star
				if globular:
					datafile = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'/'+star.specname+'_data.csv'
				else:
					datafile = '/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname+'/'+star.specname+'_data.csv'

				# Get observed and synthetic spectra and inverse variance array
				obswvl 		= np.genfromtxt(datafile, delimiter=',', skip_header=3, usecols=0)
				obsflux 	= np.genfromtxt(datafile, delimiter=',', skip_header=3, usecols=1)
				synthflux 	= np.genfromtxt(datafile, delimiter=',', skip_header=3, usecols=2)
				synthfluxup = np.genfromtxt(datafile, delimiter=',', skip_header=3, usecols=3)
				synthfluxdown = np.genfromtxt(datafile, delimiter=',', skip_header=3, usecols=4)
				ivar 		= np.genfromtxt(datafile, delimiter=',', skip_header=3, usecols=5)

				idx = np.where(name == star.specname)
				if mnerr[idx][0] < 1:
					chi_sq.make_plots('new', name, obswvl, obsflux, synthflux, outputname, ivar=ivar, synthfluxup=synthfluxup, synthfluxdown=synthfluxdown, title=None)

		except Exception as e:
			print(repr(e))
			print('Skipped star #'+str(i+1)+'/'+str(Nstars)+' stars')
			continue

		print('Finished star '+star.specname, '#'+str(i+1)+'/'+str(Nstars)+' stars')

	return

def main():
	# Match Sculptor hi-res file to bscl1 (for AAS)
	#medresMn, medresMnerror, hiresMn, hiresMnerror, sep.arcsec = match_hires('Sculptor_hires.tsv','/raid/caltech/moogify/bscl1/moogify.fits.gz')

	# Measure Mn abundances for Sculptor
	#run_chisq('/raid/caltech/moogify/bscl1/moogify.fits.gz', '/raid/gduggan/moogify/bscl1_moogify.fits.gz', 'scl', 'scl1', startstar=0, lines='new')
	#run_chisq('/raid/caltech/moogify/bscl2/moogify.fits.gz', '/raid/gduggan/moogify/bscl2_moogify.fits.gz', 'scl', 'scl2', startstar=0, lines='new')
	#run_chisq('/raid/caltech/moogify/bscl6/moogify.fits.gz', '/raid/gduggan/moogify/bscl6_moogify.fits.gz', 'scl', 'scl6', startstar=0, lines='new')

	# Measure Mn abundances for Sculptor using new 1200B data
	#run_chisq('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', '/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'scl5_1200B', startstar=0, lines='new')

	'''
	# Measure Mn abundances for Ursa Minor
	run_chisq('/raid/caltech/moogify/bumi1/moogify.fits.gz', '/raid/gduggan/moogify/bumi1_moogify.fits.gz', 'umi', 'umi1', startstar=0)
	run_chisq('/raid/caltech/moogify/bumi2/moogify.fits.gz', '/raid/gduggan/moogify/bumi2_moogify.fits.gz', 'umi', 'umi2', startstar=0)
	run_chisq('/raid/caltech/moogify/bumi3/moogify.fits.gz', '/raid/gduggan/moogify/bumi3_moogify.fits.gz', 'umi', 'umi3', startstar=0)

	# Measure Mn abundances for Draco
	run_chisq('/raid/caltech/moogify/bdra1/moogify.fits.gz', '/raid/gduggan/moogify/bdra1_moogify.fits.gz', 'dra', 'dra1', startstar=0)
	run_chisq('/raid/caltech/moogify/bdra2/moogify.fits.gz', '/raid/gduggan/moogify/bdra2_moogify.fits.gz', 'dra', 'dra2', startstar=0)
	run_chisq('/raid/caltech/moogify/bdra3/moogify.fits.gz', '/raid/gduggan/moogify/bdra3_moogify.fits.gz', 'dra', 'dra3', startstar=0)

	# Measure Mn abundances for Sextans
	run_chisq('/raid/caltech/moogify/bsex2/moogify.fits.gz', '/raid/gduggan/moogify/bsex2_moogify.fits.gz', 'sex', 'sex2', startstar=0)
	run_chisq('/raid/caltech/moogify/bsex3/moogify.fits.gz', '/raid/gduggan/moogify/bsex3_moogify.fits.gz', 'sex', 'sex3', startstar=0)

	# Measure Mn abundances for Fornax
	run_chisq('/raid/caltech/moogify/bfor6/moogify.fits.gz', '/raid/gduggan/moogify/bfor6_moogify.fits.gz', 'for', 'for6', startstar=0)
	'''

	# Measure Mn abundances for Fornax using new 1200B data
	run_chisq('/raid/caltech/moogify/bfor7_1200B/moogify.fits.gz', '/raid/caltech/moogify/bfor7_1200B/moogify.fits.gz', 'for', 'for7_1200B', startstar=0, lines='new')

	# Measure Mn abundances for a test globular cluster (NGC 2419)
	#run_chisq('/raid/caltech/moogify/n2419b_blue/moogify.fits.gz', '/raid/gduggan/moogify/n2419b_blue_moogify.fits.gz', 'n2419', 'n2419b_blue', startstar=0, globular=True, lines='new')

	# Measure Mn abundances for a test globular cluster (M15) using new 1200B data
	#run_chisq('/raid/caltech/moogify/7078l1_1200B/moogify.fits.gz', '/raid/caltech/moogify/7078l1_1200B/moogify.fits.gz', 'm15', '7078l1_1200B', startstar=0, globular=True, lines='new')

	# Plot chi-sq contours for stars that already have [Mn/H] measured
	#make_chisq_plots('/raid/caltech/moogify/n2419b_blue/moogify.fits.gz', '/raid/gduggan/moogify/n2419b_blue_moogify.fits.gz', 'n2419b_blue', 'n2419b_blue', startstar=11, globular=True)

if __name__ == "__main__":
	main()