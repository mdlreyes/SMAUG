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
from multiprocessing import Pool
import functools

def run_chisq(filename, paramfilename, galaxyname, slitmaskname, startstar=0, globular=False, lines='new', plots=False, wvlcorr=True, membercheck=None, memberlist=None, velmemberlist=None):
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
	memberlist		-- member list (from Evan's Li-rich giants paper)
	velmemberlist 	-- member list (from radial velocity check)

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

	# Get number of stars in file
	Nstars = open_obs_file(filename)

	# Get coordinates of stars in file
	RA, Dec = open_obs_file(filename, coords=True)

	# Function to parallelize: run chi-sq fitting for all stars in file
	def mp_worker(i, filename, Nstars, RA, Dec, membercheck, membernames):

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
					print('Not in member list! Skipped '+specname)
					continue

			# Run optimization code
			star = chi_sq.obsSpectrum(filename, paramfilename, i, wvlcorr, galaxyname, slitmaskname, globular, lines, plot=True)
			best_mn, error, finalchisq = star.plot_chisq(fe, output=True, plots=plots)

		except Exception as e:
			print(repr(e))
			print('Skipped star #'+str(i+1)+'/'+str(Nstars)+' stars')
			continue

		print('Finished star '+star.specname, '#'+str(i+1)+'/'+str(Nstars)+' stars')

		return star.specname, str(RA[i]), str(Dec[i]), str(star.temp), str(star.logg[0]), str(star.fe[0]), str(star.fe_err[0]), str(star.alpha[0]), str(best_mn[0]), str(error[0]), str(finalchisq)

	# Begin the parallelization
	p = multiprocessing.Pool()

	with open(outputname, 'w') as f:
		for result in p.imap(functools.partial(mp_worker, filename=filename, Nstars=Nstars, RA=RA, Dec=Dec, membercheck=membercheck, membernames=membernames), range(startstar, Nstars)):
			f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % result)

	return

def main():

	# Measure Mn abundances for globular clusters
	#run_chisq('/raid/caltech/moogify/n2419b_blue/moogify.fits.gz', '/raid/gduggan/moogify/n2419b_blue_moogify.fits.gz', 'n2419', 'n2419b_blue', startstar=0, globular=True, lines='new')
	#run_chisq('/raid/caltech/moogify/7089l3_1200B/moogify.fits.gz', '/raid/caltech/moogify/7089l3/moogify7_flexteff.fits.gz', 'n7089', '7089l3_1200B', startstar=0, globular=True, lines='new', plots=True, wvlcorr=False, membercheck='M2', memberlist='/raid/caltech/articles/kirby_gclithium/table_catalog.dat')
	#run_chisq('/raid/caltech/moogify/ng1904_1200B/moogify.fits.gz', '/raid/caltech/moogify/ng1904_1200B/moogify.fits.gz', 'n1904', 'ng1904_1200B', startstar=0, globular=True, lines='new', plots=True, wvlcorr=False)
	run_chisq('/raid/caltech/moogify/7089l1_1200B/moogify.fits.gz', '/raid/caltech/moogify/7089l1_1200B/moogify.fits.gz', 'n7089', '7089l1_1200B', startstar=0, globular=True, lines='new', plots=True, wvlcorr=False, membercheck='M2', memberlist='/raid/caltech/articles/kirby_gclithium/table_catalog.dat', velmemberlist='/raid/madlr/glob/n7089/7089l1_1200B_velmembers.txt')
	run_chisq('/raid/caltech/moogify/7078l1_1200B/moogify.fits.gz', '/raid/caltech/moogify/7078l1_1200B/moogify.fits.gz', 'n7078', '7078l1_1200B', startstar=0, globular=True, lines='new', plots=True, wvlcorr=False, membercheck='M15', memberlist='/raid/caltech/articles/kirby_gclithium/table_catalog.dat', velmemberlist='/raid/madlr/glob/n7078/7078l1_1200B_velmembers.txt')
	run_chisq('/raid/caltech/moogify/n5024b_1200B/moogify.fits.gz', '/raid/caltech/moogify/n5024b_1200B/moogify.fits.gz', 'n5024', 'n5024b_1200B', startstar=0, globular=True, lines='new', plots=True, wvlcorr=False, membercheck='M53', memberlist='/raid/caltech/articles/kirby_gclithium/table_catalog.dat', velmemberlist='/raid/madlr/glob/n5024/n5024b_1200B_velmembers.txt')

if __name__ == "__main__":
	main()