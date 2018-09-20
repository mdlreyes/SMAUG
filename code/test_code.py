# test_code.py
# Short programs to test code
#
# Created 20 Sept 18
# Updated 20 Sept 18
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
from smooth_gauss import smooth_gauss
from match_spectrum import open_obs_file, smooth_gauss_wrapper
import scipy.optimize
from chi_sq import make_plots

def arcturus_test(resolution, temp, logg, fe, alpha, mn): #, fit=False):
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
	#fit 		-- if 'False' (default), just compare w/ synthetic spectrum with known abundance;
	#				else, try to fit Arcturus spectrum and find best-fit Mn abundance

	Outputs:
	"""

	# Open Arcturus spectrum

	# Smooth to DEIMOS resolution

	# Produce synthetic spectrum
	synth = runMoog(temp=temp, logg=logg, fe=fe, alpha=alpha, elements=[25], abunds=[mn], solar=[5.43], lines='new')
	synthflux = []

	for i in range(len(synth)):

		# Loop over each line
		synthregion = synth[i]

		# Smooth each region of synthetic spectrum to match each region of continuum-normalized observed spectrum
		newsynth = get_synth(obswvl_fit[i], obsflux_fit[i], ivar_fit[i], dlam_fit[i], synth=synthregion)
		synthflux.append(newsynth)

	synthflux = np.hstack(synthflux[:])

	# Plot observed and synthetic spectra
	make_plots(lines='new', specname='Arcturus', obswvl=obswvl, obsflux=obsflux, synthflux=synthflux, outputname='/raid/madlr/test', title='Arcturus')

	return

def main():
	arcturus_test()
	return

if __name__ == "__main__":
	main()