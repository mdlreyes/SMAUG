# run_moog.py
# Runs MOOG 8 times, one for each Mn linelist
#
# - createPar: for a given *.atm file and linelist, output *.par file
# 
# Created 4 Jan 18
# Updated 22 Jan 18
###################################################################

import os
import numpy as np
import math
from interp_atmosphere import *

linelists = np.array(['linelist_Mn4754','linelist_Mn4783','linelist_Mn4823','linelist_Mn5394','linelist_Mn5537','linelist_Mn60136021'])
	
def createPar(name, atmfile, linelist, directory=''):
	"""Create *.par file using *.atm file and linelist."""

	# Define filename
	filestr = directory + name + '.par'

	# Open linelist and get wavelength range to synthesize spectrum
	wavelengths = np.genfromtxt(linelist, skip_header=1, usecols=0)
	wavelengthrange = [ math.floor(wavelengths[0]),math.ceil(wavelengths[-1]) ]

	# Check if file already exists
	exists, readytowrite = checkFile(filestr)
	if readytowrite:

		# Outfile names:
		out1 = '\''+directory+name+'.out1\''
		out2 = '\''+directory+name+'.out2\''

		# If file exists, open file
		with open(filestr, 'w+') as file:

			# Print lines of .par file
			file.write('synth'+'\n')
			file.write('terminal       '+'\'x11\''+'\n')
			file.write('standard_out   '+out1+'\n')
			file.write('summary_out    '+out2+'\n')
			file.write('model_in       '+'\''+atmfile+'.atm'+'\''+'\n')
			file.write('lines_in       '+'\''+linelist+'.list'+'\''+'\n')
			file.write('strong        0'+'\n')
			file.write('atmosphere    1'+'\n')
			file.write('molecules     1'+'\n')
			file.write('damping       1'+'\n')
			file.write('trudamp       0'+'\n')
			file.write('lines         1'+'\n')
			file.write('flux/int      0'+'\n')
			file.write('plot          0'+'\n')
			file.write('synlimits'+'\n')
			file.write('  '+'{0:.3f}'.format(wavelengthrange[0])+' '+'{0:.3f}'.format(wavelengthrange[1])+'  0.01  1.00'+'\n')
			file.write('obspectrum    0')

	return filestr

def runMoog(temp, logg, fe, alpha, directory='/raid/madlr/moogspectra/', elements=None, abunds=None):
	"""Run MOOG for each Mn linelist.

    Inputs:
    temp 	 -- effective temperature (K)
    logg 	 -- surface gravity
    fe 		 -- [Fe/H]
    alpha 	 -- [alpha/Fe]

    Keywords:
    dir 	 -- directory to write MOOG output to [default = '/raid/madlr/moogspectra/']
    elements -- list of atomic numbers of elements to add to the *.atm file
    abunds 	 -- list of elemental abundances corresponding to list of elements
    """


	# Create identifying filename (including all parameters + linelist used)
	name = getAtm(temp, logg, fe, alpha, directory='') # Add all parameters to name
	name = name[:-4] # remove .atm

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
	atmfile = writeAtm(temp=5900, logg=0.1, fe=-0.5, alpha=0.5, elements=[25], abunds=[0.5])

	# Loop over all linelists
	for i in range(len(linelists)):

		parname = name + '_' + linelists[i][9:]

		# Create *.par file
		parfile = createPar(parname, atmfile, '/raid/madlr/linelists/'+linelists[i], directory='/raid/madlr/par/')

		# Run MOOG
		os.system('MOOG')

	return 

#createPar()
runMoog(temp=5900, logg=0.1, fe=-0.5, alpha=0.5, elements=[25], abunds=[0.5])