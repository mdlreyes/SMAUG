# phaseA_Mnvalidation.py
# Runs MOOG on full list of Mn lines, outputs spectrum
#
# - plot_mn_spectrum
# 
# Created 31 Jul 18
# Updated 1 Aug 18
###################################################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import glob
import numpy as np
import math
from interp_atmosphere import checkFile, getAtm, writeAtm
from run_moog import createPar
import subprocess
import pandas
from smooth_gauss import smooth_gauss
from match_spectrum import smooth_gauss_wrapper

def runMoog_phaseA(temp, logg, fe, alpha, elements=None, abunds=None, solar=None):
	"""Run MOOG for a single linelist of all eligible Mn lines in DEIMOS range.

	Inputs:
	temp 	 -- effective temperature (K)
	logg 	 -- surface gravity
	fe 		 -- [Fe/H]
	alpha 	 -- [alpha/Fe]

	Keywords:
	elements -- list of atomic numbers of elements to add to the *.atm file
	abunds 	 -- list of elemental abundances corresponding to list of elements

	Outputs:
	spectrum -- synthetic spectrum
	"""

	# Define list of Mn linelists
	linelist = '/raid/madlr/validatelinelist/linelist_Mn_full'

	# Create identifying filename (including all parameters + linelist used)
	name = getAtm(temp, logg, fe, alpha, directory='') # Add all parameters to name

	# Add the new elements to filename, if any
	if elements is not None:

		for i in range(len(elements)):

			abund = int(abunds[i]*10)
			if elements[i] == 25:
				elementname = 'mn'

			# Note different sign conventions for abundances
			if abund < 0:
				elementstr 	= elementname + '{:03}'.format(abund)
			else:
				elementstr	= elementname + '_' + '{:02}'.format(abund)

			name = name + elementstr

	# Create *.atm file (for use with each linelist)
	print('Running MOOG with parameters: ')
	print('temp = ', temp)
	print('logg = ', logg)
	print('fe = ', fe)
	print('alpha = ', alpha)
	print('extra elements: ', elements, ' with abundances ', abunds)
	atmfile = writeAtm(temp, logg, fe, alpha, elements=elements, abunds=abunds, solar=solar)

	# Create *.par file
	parname = name
	parfile, wavelengthrange = createPar(parname, atmfile, linelist, directory='/raid/madlr/par/')

	# Run MOOG
	p = subprocess.Popen(['MOOG', parfile], cwd='/raid/madlr/moog17scat/', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		
	# Wait for MOOG to finish running
	p.communicate()

	# Create arrays of wavelengths and fluxes
	outfile = '/raid/madlr/moogout/'+parname+'.out2'

	wavelength = np.linspace(wavelengthrange[0],wavelengthrange[1],math.ceil((wavelengthrange[1]-wavelengthrange[0])/0.02), endpoint=True)
	data = pandas.read_csv(outfile, skiprows=[0,1,-1], delimiter=' ').as_matrix()

	#print(data)
	flux = data[~np.isnan(data)][:-1]

	spectrum = [1.-flux, wavelength]

	return spectrum

def phaseA_Mnvalidation(temp, logg, fe, alpha, mn_abund, delta, snr):
	"""Run MOOG for variety of Mn abundances, to test which Mn lines are important.
	Plot the output synthetic spectra.

	Inputs:
	temp 	 -- effective temperature (K)
	logg 	 -- surface gravity
	fe 		 -- [Fe/H]
	alpha 	 -- [alpha/Fe]
	mn_abund -- [Mn/H] abundance to test
	delta 	 -- desired [Mn/H] error (in dex)
	snr 	 -- desired S/N ratio

	Outputs:
	spectrum -- synthetic spectrum
	"""

	# Clean out the output directories
	for fl in glob.glob('/raid/madlr/moogout/*'):
		os.remove(fl)
	for fl in glob.glob('/raid/madlr/par/*'):
		os.remove(fl)
	for fl in glob.glob('/raid/madlr/atm/*'):
		os.remove(fl)

	# Create list of Mn abundances to run
	mn_abund_list = [mn_abund, mn_abund + delta, mn_abund - delta]

	# Run MOOG to produce spectra
	spectrum = []
	#spectrum.append(runMoog_phaseA(temp=temp, logg=logg, fe=fe, alpha=alpha, elements=[25], abunds=[-5.0], solar=[5.43]))
	for i in range(len(mn_abund_list)):
		curr_spectrum = runMoog_phaseA(temp=temp, logg=logg, fe=fe, alpha=alpha, elements=[25], abunds=[mn_abund_list[i]], solar=[5.43])
		oldflux = curr_spectrum[0]
		oldwvl = curr_spectrum[1]
		newflux = smooth_gauss_wrapper(oldwvl, oldflux, np.arange(4500, 9100, 0.421875), 0.33)
		spectrum.append([newflux, np.arange(4500, 9100, 0.421875)])

	# Plot spectra around each line
	lines = [4502, 4672, 4701, 4710, 4727, 4739, 4754, 4761, 4762, 4765, 
			4766, 4783, 4823, 4942, 4966, 4987, 5005, 5030, 5043, 5118, 
			5149, 5150, 5196, 5197, 5255, 5260, 5292, 5317, 5334, 5348, 
			5388, 5394, 5399, 5407, 5420, 5432, 5504, 5510, 5516, 5537, 
			6013, 6016, 6021, 6344, 6349, 6378, 6382, 6384, 6414, 6443,
			6491, 6519]

	for j in range(len(lines)):

		# Format plot
		fig, ax = plt.subplots(figsize=(10,6))
		ax.set_xlabel(r'$\lambda$', fontsize=16)
		ax.set_ylabel('Flux', fontsize=16)
		for label in (ax.get_xticklabels() + ax.get_yticklabels()):
			label.set_fontsize(14)

		# Mask out stuff not around line
		mask = np.where((spectrum[0][1] > (lines[j] - 5)) & (spectrum[0][1] < (lines[j] + 5)))

		# Compute differences between spectra
		diff_up = spectrum[1][0][mask] - spectrum[0][0][mask]
		diff_low = spectrum[0][0][mask] - spectrum[2][0][mask]

		# Compute inverse signal-to-noise (relative error)
		ratio_up = -diff_up / spectrum[0][0][mask]
		ratio_low = -diff_low / spectrum[0][0][mask]

		# Make plot
		ax.plot(spectrum[0][1][mask], ratio_up, label=r'$+\Delta$/Flux')
		ax.plot(spectrum[0][1][mask], ratio_low, label=r'$-\Delta$/Flux')
		plt.axhline(1./snr, color='r', linestyle='--', label='Threshold = '+str(snr))
		#for i in range(len(mn_abund_list)):
		#	ax.plot(spectrum[i][1][mask], spectrum[i][0][mask], label='[Mn/H] = '+str(mn_abund_list[i]))

		# Output file
		ax.get_xaxis().get_major_formatter().set_useOffset(False)
		plt.title(r'$\lambda = $'+str(lines[j])+'A')
		plt.legend(loc='best')
		#ax.set_ylim((0,1.1))
		outfile = '/raid/madlr/validatelinelist/phaseA_validation'+str(lines[j])+'.png'
		plt.savefig(outfile, bbox_inches='tight')
		plt.close()
		#plt.show()

	return

def main():
	phaseA_Mnvalidation(temp=4500, logg=1.0, fe=-1.0, alpha=-0.3, mn_abund=0, delta=0.5, snr=100)

if __name__ == "__main__":
	main()