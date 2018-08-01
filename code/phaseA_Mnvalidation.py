# phaseA_Mnvalidation.py
# Runs MOOG on full list of Mn lines, outputs spectrum
#
# - plot_mn_spectrum
# 
# Created 31 Jul 18
# Updated 1 Aug 18
###################################################################

import os
import sys
import glob
import numpy as np
import math
from interp_atmosphere import checkFile, getAtm, writeAtm
from run_moog import createPar
import subprocess
import pandas

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

	# Clean out the output directories
	for fl in glob.glob('/raid/madlr/moogout/*'):
		os.remove(fl)
	for fl in glob.glob('/raid/madlr/par/*'):
		os.remove(fl)
	for fl in glob.glob('/raid/madlr/atm/*'):
		os.remove(fl)

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
	flux = data[~np.isnan(data)][:-1]

	spectrum.append([1.-flux, wavelength])

	return spectrum

def phaseA_Mnvalidation(temp, logg, fe, alpha, mn_abund):
	"""Run MOOG for variety of Mn abundances, to test which Mn lines are important.
	Plot the output synthetic spectra.

    Inputs:
    temp 	 -- effective temperature (K)
    logg 	 -- surface gravity
    fe 		 -- [Fe/H]
    alpha 	 -- [alpha/Fe]
    mn_abund -- list of [Mn/H] abundances to test

    Outputs:
    spectrum -- synthetic spectrum
    """

    # Create figure
	fig, ax = plt.subplots(figsize=(10,6))

	# Format plot
	ax.set_title(title, fontsize=18)
	ax.set_xlabel(r'$\lambda$', fontsize=16)
	ax.set_ylabel('Flux', fontsize=16)
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(14)

    # Run MOOG to produce spectra
    for abund in mn_abund:
		spectrum = runMoog(temp=temp, logg=logg, fe=fe, alpha=alpha, elements=[25], abunds=[abund], solar=[5.43])
		ax.plot(spectrum[1], spectrum[0], label='[Mn/H] = '+str(abund))

	# Output file
	outfile = '/raid/madlr/validatelinelist/phaseA_validation.png'
	plt.savefig(outfile, bbox_inches='tight')
	#plt.show()

	return

def main():
	phaseA_Mnvalidation(temp=5900, logg=0.1, fe=-0.5, alpha=0.5, mn_abund=[0.0, -1.5, -10.0])

if __name__ == "__main__":
	main()