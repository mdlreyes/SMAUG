# output_mn_mpi.py
# Runs SMAUG fitting code on multiple processors
#
# Created 9 July 19
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
from astropy.io import fits, ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas
import scipy.optimize
import chi_sq
from make_plots import make_plots

# Packages for parallelization
import multiprocessing
import functools

def prep_run(filename, galaxyname, slitmaskname, membercheck=None, memberlist='/raid/caltech/articles/kirby_gclithium/table_catalog.dat', velmemberlist=None, globular=False):
	""" Get data in preparation for measuring chi-sq values.

	Inputs:
	filename 		-- file with observed spectra
	galaxyname		-- galaxy name, options: 'scl'
	slitmaskname 	-- slitmask name, options: 'scl1'

	Keywords:
	membercheck 	-- do membership check for this object
	memberlist		-- member list (from Evan's Li-rich giants paper)
	velmemberlist 	-- member list (from radial velocity check)
	globular 		-- if 'False' (default), put into output path of galaxy; else, put into globular cluster path

	Outputs:
	Nstars 			-- total number of stars in the data file
	RA, Dec 		-- coordinate arrays for stars in file
	membernames		-- list of all members (based on Evan's Li-rich giants paper and/or velocity check)
	galaxyname, slitmaskname, filename, membercheck -- from input

	"""

	# Prep for member check
	if membercheck is not None:

		# Check if stars are in member list from Evan's Li-rich giants paper
		table = ascii.read(memberlist)
		memberindex = np.where(table.columns[0] == membercheck)
		membernames = table.columns[1][memberindex]

		# Also check if stars are in member list from velocity cut
		if velmemberlist is not None:
			oldmembernames = membernames
			membernames = []
			velmembernames = np.genfromtxt(velmemberlist,dtype='str')
			for i in range(len(oldmembernames)):
				if oldmembernames[i] in velmembernames:
					membernames.append(oldmembernames)

			membernames = np.asarray(membernames)

	else:
		membernames = None

	# Get number of stars in file
	Nstars = open_obs_file(filename)

	# Get coordinates of stars in file
	RA, Dec = open_obs_file(filename, coords=True)

	return galaxyname, slitmaskname, filename, Nstars, RA, Dec, membercheck, membernames, globular

def mp_worker(i, filename, paramfilename, wvlcorr, galaxyname, slitmaskname, globular, lines, plots, Nstars, RA, Dec, membercheck, membernames, correctionslist, corrections):
	""" Function to parallelize: chi-sq fitting for a single star """

	try:
		# Get metallicity of star to use for initial guess
		print('Getting initial metallicity')
		temp, logg, fe, alpha, fe_err = open_obs_file(filename, retrievespec=i, specparams=True)

		# Get dlam (FWHM) of star to use for initial guess
		specname, obswvl, obsflux, ivar, dlam, zrest = open_obs_file(filename, retrievespec=i)

		# Check for bad parameter measurement
		if np.isclose(temp, 4750.) and np.isclose(fe,-1.5) and np.isclose(alpha,0.2):
			print('Bad parameter measurement! Skipped #'+str(i+1)+'/'+str(Nstars)+' stars')
			return None

		# Do membership check
		if membercheck is not None:
			if specname not in membernames:
				print('Not in member list! Skipped '+specname)
				return None

		# Vary stellar parameters
		if int(specname) in np.asarray(correctionslist['ID']):
			idx = np.where(np.asarray(correctionslist['ID']) == int(specname))[0]

			# Vary the quantity required
			if corrections[0] == 'Teff':
				colstring = 'Teff'+str(int(np.abs(corrections[2])))
				if corrections[1] == 'up':
					temp = temp + corrections[2] # Make the correction go the right direction
					fe = fe + float(correctionslist['FeH_'+colstring][idx])
				else:
					temp = temp - corrections[2]
					fe = fe - float(correctionslist['FeH_'+colstring][idx])

			elif corrections[0] == 'logg':
				colstring = 'logg0'+str(int(10*np.abs(corrections[2])))
				if corrections[1] == 'up':
					logg = logg + corrections[2] # Make the correction go the right direction
					fe = fe - float(correctionslist['FeH_'+colstring][idx])
				else:
					logg = logg - corrections[2]
					fe = fe + float(correctionslist['FeH_'+colstring][idx])

			# Now determine the direction to vary alpha
			'''
			key = corrections[0]+str(corrections[2])
			if corrections[3] == 'up':
				alpha = alpha + float(correctionslist['MgFe_'+key][idx] + 4*correctionslist['SiFe_'+key][idx] + 2*correctionslist['CaFe_'+key][idx] + 6*correctionslist['TiFe_'+key][idx])/13
			else:
				alpha = alpha - float(correctionslist['MgFe_'+key][idx] + 4*correctionslist['SiFe_'+key][idx] + 2*correctionslist['CaFe_'+key][idx] + 6*correctionslist['TiFe_'+key][idx])/13
			'''

		else:
			print('No stellar parameter corrections listed!')
			return None

		# Run optimization code
		star = chi_sq.obsSpectrum(filename, paramfilename, i, wvlcorr, galaxyname, slitmaskname, globular, lines, plot=True, specialparams=[temp, logg, fe, alpha])
		best_mn, error, finalchisq = star.plot_chisq(fe, output=True, plots=plots)

		print('Finished star '+star.specname, '#'+str(i+1)+'/'+str(Nstars)+' stars')

		result = np.array([star.temp, star.logg, star.fe, star.fe_err, star.alpha, best_mn, error])
		for i in range(len(result)):
			if np.isscalar(result[i])==False:
				result[i] = result[i][0]

		return star.specname, str(RA[i]), str(Dec[i]), str(result[0]), str(result[1]), str(result[2]), str(result[3]), str(result[4]), str(result[5]), str(result[6]), str(finalchisq)

	except Exception as e:
		print(repr(e))
		print('Skipped star #'+str(i+1)+'/'+str(Nstars)+' stars')
		return None

def mp_handler(galaxyname, slitmaskname, filename, Nstars, RA, Dec, membercheck, membernames, globular, startstar=0, paramfilename=None, lines='new', plots=True, wvlcorr=False, corrections=None):
	""" Measure Mn abundances using parallel processing and write to file.

	Inputs: takes all inputs from prep_run

	Keywords:
	startstar		-- if 0 (default), start at beginning of file and write new output file;
						else, start at #startstar and just append to output file
	paramfile 		-- name of parameter file; if 'None' (default), just use main datafile
	lines 			-- if 'new' (default), use new revised linelist; else, use original linelist from Judy's code
	plots 			-- if 'True' (default), plot final fits/resids while doing the fits
	wvlcorr 		-- if 'True' (default), do linear wavelength corrections following G. Duggan's code for 900ZD data;
						else (for 1200B data), don't do corrections
	corrections 	-- array describing stellar parameter variations:
						- corrections[0]: 'Teff' or 'logg'
						- corrections[1]: 'up' or 'down' for Teff/logg
						- corrections[2]: amount by which to vary
						- corrections[3]: 'up' or 'down' for [alpha/Fe]
	"""

	# Create output file
	if corrections is not None:
		outputname = '/raid/madlr/dsph/'+galaxyname+'/'+corrections[0]+corrections[1]+str(corrections[2])+'.csv' #+'_alpha'+corrections[3]+'.csv'
	elif globular:
		outputname = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'.csv'
	else:
		outputname = '/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname+'.csv'

	# Open new output file if necessary
	if startstar<1:
		with open(outputname, 'w+') as f:
			f.write('Name\tRA\tDec\tTemp\tlog(g)\t[Fe/H]\terror([Fe/H])\t[alpha/Fe]\t[Mn/H]\terror([Mn/H])\tchisq(reduced)\n')

	# Unless paramfilename is defined, assume it's the same as the main data file
	if paramfilename is None:
		paramfilename = filename

	# Upload correctionlist
	if corrections is not None:
		correctionslist = pandas.read_csv('Kirby10_stellarparam_corrections_scl.txt', delimiter='\s+')
	else:
		correctionslist = None

	# Define function to parallelize
	func = functools.partial(mp_worker, filename=filename, paramfilename=paramfilename, wvlcorr=wvlcorr, galaxyname=galaxyname, slitmaskname=slitmaskname, globular=globular, lines=lines, plots=plots, Nstars=Nstars, RA=RA, Dec=Dec, membercheck=membercheck, membernames=membernames, correctionslist=correctionslist, corrections=corrections)

	# Begin the parallelization
	p = multiprocessing.Pool()

	with open(outputname, 'a') as f:
		for result in p.imap(func, range(startstar, Nstars)):
			if result is not None:
				f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % result)

	return

def main():

	# Measure Mn abundances for globular clusters
	#mp_handler(*prep_run('/raid/caltech/moogify/n5024b_1200B/moogify.fits.gz', 'n5024', 'n5024b_1200B', membercheck='M53', velmemberlist='/raid/madlr/glob/n5024/n5024b_1200B_velmembers.txt', globular=True))
	#mp_handler(*prep_run('/raid/caltech/moogify/7078l1_1200B/moogify.fits.gz', 'n7078', '7078l1_1200B', membercheck='M15', velmemberlist='/raid/madlr/glob/n7078/7078l1_1200B_velmembers.txt', globular=True))
	#mp_handler(*prep_run('/raid/caltech/moogify/7089l1_1200B/moogify.fits.gz', 'n7089', '7089l1_1200B', membercheck='M2', velmemberlist='/raid/madlr/glob/n7089/7089l1_1200B_velmembers.txt', globular=True))

	# Measure Mn abundances for dSphs
	#mp_handler(*prep_run('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'bscl5_1200B', globular=False))
	#mp_handler(*prep_run('/raid/caltech/moogify/bfor7_1200B/moogify.fits.gz', 'for', 'bfor7_1200B', globular=False))
	#mp_handler(*prep_run('/raid/caltech/moogify/LeoIb_1200B/moogify.fits.gz', 'leoi', 'LeoIb_1200B', globular=False))
	#mp_handler(*prep_run('/raid/madlr/test/CVnIa_1200B_moogifynew.fits.gz', 'cvni', 'CVnIa_1200B', globular=False))
	#mp_handler(*prep_run('/raid/caltech/moogify/umaIIb_1200B/moogify.fits.gz', 'umaii', 'umaIIb_1200B', globular=False))
	#mp_handler(*prep_run('/raid/madlr/test/bumia_1200B_moogifynew.fits.gz', 'umi', 'bumia_1200B', globular=False))

	# Check stellar param variations
	#mp_handler(*prep_run('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'bscl5_1200B', globular=False), corrections=['Teff','up',125,'up'])
	#mp_handler(*prep_run('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'bscl5_1200B', globular=False), corrections=['Teff','up',125])
	#mp_handler(*prep_run('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'bscl5_1200B', globular=False), corrections=['Teff','down',125])
	#mp_handler(*prep_run('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'bscl5_1200B', globular=False), corrections=['Teff','up',250])
	#mp_handler(*prep_run('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'bscl5_1200B', globular=False), corrections=['Teff','down',250])

	#mp_handler(*prep_run('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'bscl5_1200B', globular=False), corrections=['logg','up',0.3])
	#mp_handler(*prep_run('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'bscl5_1200B', globular=False), corrections=['logg','down',0.3])
	#mp_handler(*prep_run('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'bscl5_1200B', globular=False), corrections=['logg','up',0.6])
	#mp_handler(*prep_run('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 'scl', 'bscl5_1200B', globular=False), corrections=['logg','down',0.6])

if __name__ == "__main__":
	main()
