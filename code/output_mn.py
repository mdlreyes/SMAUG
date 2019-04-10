# output_mn.py
# Produces nice outputs
#
# Created 5 June 18
# Updated 17 Nov 18
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
from make_plots import make_plots

def run_chisq(filename, paramfilename, galaxyname, slitmaskname, startstar=0, globular=False, lines='new', plots=False, wvlcorr=True, membercheck=None, memberlist=None):
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
	plots 			-- if 'False' (default), don't plot final fits/resids while doing the fits;
						else, plot them
	wvlcorr 		-- if 'True' (default), do linear wavelength corrections following G. Duggan's code for 900ZD data;
						else (for 1200B data), don't do corrections
	membercheck 	-- do membership check for this object
	memberlist		-- member list

	"""

	# Output filename
	if globular:
		outputname = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'.csv'
	else:
		outputname = '/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname+'.csv'

	# Open new file
	if startstar<1:
		with open(outputname, 'w+') as f:
			f.write('Name\tRA\tDec\tTemp\tlog(g)\t[Fe/H]\terror([Fe/H])\t[alpha/Fe]\t[Mn/H]\terror([Mn/H])\tchisq(reduced)\n')

	# Prep for member check
	if membercheck is not None:
		table = ascii.read(memberlist)
		memberindex = np.where(table.columns[0] == membercheck)
		membernames = table.columns[1][memberindex]

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

			# Check for bad parameter measurement
			if np.isclose(temp, 4750.) and np.isclose(fe,-1.5) and np.isclose(alpha,0.2):
				print('Bad parameter measurement! Skipped #'+str(i+1)+'/'+str(Nstars)+' stars')
				continue

			# Do membership check
			if membercheck is not None:
				if specname not in membernames:
					continue

			# Run optimization code
			star = chi_sq.obsSpectrum(filename, paramfilename, i, wvlcorr, galaxyname, slitmaskname, globular, lines)
			best_mn, error, finalchisq = star.plot_chisq(fe, output=True, plots=plots)

		except Exception as e:
			print(repr(e))
			print('Skipped star #'+str(i+1)+'/'+str(Nstars)+' stars')
			continue

		print('Finished star '+star.specname, '#'+str(i+1)+'/'+str(Nstars)+' stars')

		with open(outputname, 'a') as f:
			f.write(star.specname+'\t'+str(RA[i])+'\t'+str(Dec[i])+'\t'+str(star.temp)+'\t'+str(star.logg[0])+'\t'+str(star.fe[0])+'\t'+str(star.fe_err[0])+'\t'+str(star.alpha[0])+'\t'+str(best_mn[0])+'\t'+str(error[0])+'\t'+str(finalchisq)+'\n')

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
					best_mn, error = star.plot_chisq(params0, minimize=False, plots=True)

		except Exception as e:
			print(repr(e))
			print('Skipped star #'+str(i+1)+'/'+str(Nstars)+' stars')
			continue

		print('Finished star '+star.specname, '#'+str(i+1)+'/'+str(Nstars)+' stars')

	return

def plot_fits_postfacto(filename, paramfilename, galaxyname, slitmaskname, startstar=0, globular=False, lines='new', mn_cluster=None):
	""" Plot fits, residuals, and ivar for stars whose [Mn/H] abundances have already been measured.

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
	mn_cluster 		-- if not None (default), also plot spectrum with [Mn/H] = mean [Mn/H] of cluster

	"""

	# Input filename
	if globular:
		file = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'.csv'
	else:
		file = '/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname+'.csv'

	# Output filepath
	if globular:
		outputname = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname
	else:
		outputname = '/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname

	name  = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')
	mn    = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=8)
	mnerr = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=9)

	# Get number of stars in file with observed spectra
	Nstars = open_obs_file(filename)

	# Plot spectra for each star
	for i in range(startstar, Nstars):

		try:

			# Check if parameters are measured
			temp, logg, fe, alpha, fe_err = open_obs_file(filename, retrievespec=i, specparams=True)
			if np.isclose(1.5,logg) and np.isclose(fe,-1.5) and np.isclose(fe_err, 0.0):
				print('Bad parameter measurement! Skipped #'+str(i+1)+'/'+str(Nstars)+' stars')
				continue

			# Open star
			star = chi_sq.obsSpectrum(filename, paramfilename, i, False, galaxyname, slitmaskname, globular, lines)

			# Check if star has already had [Mn/H] measured
			if star.specname in name:

				# If so, open data file for star
				if globular:
					datafile = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'/'+str(star.specname)+'_data.csv'
				else:
					datafile = '/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname+'/'+str(star.specname)+'_data.csv'

				# Get observed and synthetic spectra and inverse variance array
				obswvl 		= np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=0)
				obsflux 	= np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=1)
				synthflux 	= np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=2)
				#synthfluxup = np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=3)
				#synthfluxdown = np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=4)
				ivar 		= np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=5)

				idx = np.where(name == star.specname)

				synthfluxup 	= star.synthetic(obswvl, mn[idx] + 0.15, full=True)
				synthfluxdown 	= star.synthetic(obswvl, mn[idx] - 0.15, full=True)
				synthflux_nomn 	= star.synthetic(obswvl, -10.0, full=True)

				if mn_cluster is not None:
					synthflux_cluster = [mn_cluster, star.synthetic(obswvl, mn_cluster, full=True)]
				else:
					synthflux_cluster=None

				if mnerr[idx][0] < 1:
					make_plots(lines, star.specname+'_', obswvl, obsflux, synthflux, outputname, ivar=ivar, resids=True, synthfluxup=synthfluxup, synthfluxdown=synthfluxdown, synthflux_nomn=synthflux_nomn, synthflux_cluster=synthflux_cluster, title=None)

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
	#run_chisq('/raid/caltech/moogify/bscl1/moogify.fits.gz', '/raid/gduggan/moogify/bscl1_moogify.fits.gz', 'scl', 'scl1', startstar=36, lines='new', plots=True)
	#run_chisq('/raid/caltech/moogify/bscl2/moogify.fits.gz', '/raid/gduggan/moogify/bscl2_moogify.fits.gz', 'scl', 'scl2', startstar=0, lines='new', plots=True)
	#run_chisq('/raid/caltech/moogify/bscl6/moogify.fits.gz', '/raid/gduggan/moogify/bscl6_moogify.fits.gz', 'scl', 'scl6', startstar=0, lines='new', plots=True)

	# Measure Mn abundances for Sculptor using new 1200B data
	#run_chisq('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', '/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'scl5_1200B', startstar=0, lines='new', plots=True, wvlcorr=False)
	#plot_fits_postfacto('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', '/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'scl5_1200B', startstar=0, globular=False, lines='new')
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
	#run_chisq('/raid/caltech/moogify/bfor7_1200B/moogify.fits.gz', '/raid/caltech/moogify/bfor7_1200B/moogify.fits.gz', 'for', 'for7_1200B', startstar=0, lines='new')

	# Measure Mn abundances for globular clusters
	#run_chisq('/raid/caltech/moogify/n2419b_blue/moogify.fits.gz', '/raid/gduggan/moogify/n2419b_blue_moogify.fits.gz', 'n2419', 'n2419b_blue', startstar=0, globular=True, lines='new')
	run_chisq('/raid/caltech/moogify/7078l1_1200B/moogify.fits.gz', '/raid/caltech/moogify/7078l1B/moogify7_flexteff.fits.gz', 'n7078', '7078l1_1200B', startstar=0, globular=True, lines='new', plots=True, wvlcorr=False, membercheck='M15', memberlist='/raid/caltech/articles/kirby_gclithium/table_catalog.dat')
	run_chisq('/raid/caltech/moogify/7089l1_1200B/moogify.fits.gz', '/raid/caltech/moogify/7089l1/moogify7_flexteff.fits.gz', 'n7089', '7089l1_1200B', startstar=0, globular=True, lines='new', plots=True, wvlcorr=False, membercheck='M2', memberlist='/raid/caltech/articles/kirby_gclithium/table_catalog.dat')
	run_chisq('/raid/caltech/moogify/7089l3_1200B/moogify.fits.gz', '/raid/caltech/moogify/7089l3/moogify7_flexteff.fits.gz', 'n7089', '7089l3_1200B', startstar=0, globular=True, lines='new', plots=True, wvlcorr=False, membercheck='M2', memberlist='/raid/caltech/articles/kirby_gclithium/table_catalog.dat')
	#run_chisq('/raid/caltech/moogify/ng1904_1200B/moogify.fits.gz', '/raid/caltech/moogify/ng1904_1200B/moogify.fits.gz', 'n1904', 'ng1904_1200B', startstar=0, globular=True, lines='new', plots=True, wvlcorr=False)
	#plot_fits_postfacto('/raid/caltech/moogify/7089l1_1200B/moogify.fits.gz', '/raid/caltech/moogify/7089l1/moogify7_flexteff.fits.gz', 'n7089', '7089l1_1200B', startstar=0, globular=True, lines='new', mn_cluster=-1.66)
	#plot_fits_postfacto('/raid/caltech/moogify/7089l3_1200B/moogify.fits.gz', '/raid/caltech/moogify/7089l3/moogify7_flexteff.fits.gz', 'n7089', '7089l3_1200B', startstar=0, globular=True, lines='new', mn_cluster=-1.66)
	#plot_fits_postfacto('/raid/caltech/moogify/7078l1_1200B/moogify.fits.gz', '/raid/caltech/moogify/7078l1B/moogify7_flexteff.fits.gz', 'n7078', '7078l1_1200B', startstar=0, globular=True, lines='new', mn_cluster=-2.38)

	# Plot chi-sq contours for stars that already have [Mn/H] measured
	#make_chisq_plots('/raid/caltech/moogify/n2419b_blue/moogify.fits.gz', '/raid/gduggan/moogify/n2419b_blue_moogify.fits.gz', 'n2419b_blue', 'n2419b_blue', startstar=11, globular=True)

if __name__ == "__main__":
	main()