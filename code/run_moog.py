# run_moog.py
# Runs MOOG 8 times, one for each Mn linelist
# Outputs spliced synthetic spectrum
#
# - createPar: for a given *.atm file and linelist, output *.par file
# - runMoog: runs MOOG for each Mn linelist (calls createPar), splices output spectrum
# 
# Created 4 Jan 18
# Updated 10 Aug 18
###################################################################

import os
import sys
import glob
import numpy as np
import math
from interp_atmosphere import checkFile, getAtm, writeAtm
import subprocess
import pandas
	
def createPar(name, atmfile='', linelist='', directory=''):
	"""Create *.par file using *.atm file and linelist.
	"""

	# Open linelist and get wavelength range to synthesize spectrum
	wavelengths = np.genfromtxt(linelist, skip_header=1, usecols=0)
	wavelengthrange = [ math.floor(wavelengths[0]),math.ceil(wavelengths[-1]) ]

	# Define filename
	filestr = directory + name + '.par'

	# Check if file already exists
	exists, readytowrite = checkFile(filestr)
	if readytowrite:

		# Outfile names:
		out1 = '\''+'/raid/madlr/moogout/'+name+'.out1\''
		out2 = '\''+'/raid/madlr/moogout/'+name+'.out2\''

		# If file exists, open file
		with open(filestr, 'w+') as file:

			# Print lines of .par file
			file.write('synth'+'\n')
			file.write('terminal       '+'\'x11\''+'\n')
			file.write('standard_out   '+out1+'\n')
			file.write('summary_out    '+out2+'\n')
			file.write('model_in       '+'\''+atmfile+'\''+'\n')
			file.write('lines_in       '+'\''+linelist+'\''+'\n')
			file.write('strong        0'+'\n')
			file.write('atmosphere    1'+'\n')
			file.write('molecules     1'+'\n')
			file.write('damping       1'+'\n')
			file.write('trudamp       0'+'\n')
			file.write('lines         1'+'\n')
			file.write('flux/int      0'+'\n')
			file.write('plot          0'+'\n')
			file.write('synlimits'+'\n')
			file.write('  '+'{0:.3f}'.format(wavelengthrange[0])+' '+'{0:.3f}'.format(wavelengthrange[1])+'  0.02  1.00'+'\n')
			file.write('obspectrum    0')

	return filestr, wavelengthrange

def runMoog(temp, logg, fe, alpha, directory='/raid/madlr/moogspectra/', elements=None, abunds=None, solar=None, lines='new'):
	"""Run MOOG for each Mn linelist and splice spectra.

    Inputs:
    temp 	 -- effective temperature (K)
    logg 	 -- surface gravity
    fe 		 -- [Fe/H]
    alpha 	 -- [alpha/Fe]

    Keywords:
    dir 	 -- directory to write MOOG output to [default = '/raid/madlr/moogspectra/']
    elements -- list of atomic numbers of elements to add to the *.atm file
    abunds 	 -- list of elemental abundances corresponding to list of elements
    lines    -- if 'new', use new revised linelist; else, use original linelist from Judy's code

    Outputs:
    spectrum -- spliced synthetic spectrum
    """

	# Clean out the output directories
	for fl in glob.glob('/raid/madlr/moogout/*'):
		os.remove(fl)
	for fl in glob.glob('/raid/madlr/par/*'):
		os.remove(fl)
	for fl in glob.glob('/raid/madlr/atm/*'):
		os.remove(fl)

	# Define list of Mn linelists
	if lines == 'new':
		#linelists = np.array(['Mn47394783','Mn4823','Mn53945432','Mn55165537','Mn60136021','Mn6384','Mn6491'])
		linelists = np.array(['Mn47394783_new','Mn4823','Mn54075420_new','Mn55165537','Mn60136021','Mn6384','Mn6491'])
	elif lines == 'old':
		linelists = np.array(['linelist_Mn47544762','linelist_Mn4783','linelist_Mn4823','linelist_Mn5394','linelist_Mn5537','linelist_Mn60136021']) 
	
	spectrum  = []

	# Create identifying filename (including all parameters + linelist used)
	name = getAtm(temp, logg, fe, alpha, directory='') # Add all parameters to name
	#name = name[:-4] # remove .atm

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
	#print('Running MOOG with parameters: ')
	'''
	print('temp = ', temp)
	print('logg = ', logg)
	print('fe = ', fe)
	print('alpha = ', alpha)
	'''
	#print('extra elements: ', elements, ' with abundances ', abunds)
	atmfile = writeAtm(temp, logg, fe, alpha, elements=elements, abunds=abunds, solar=solar)

	# Loop over all linelists
	for i in range(len(linelists)):

		# Create *.par file
		parname = name + '_' + linelists[i][9:]
		parfile, wavelengthrange = createPar(parname, atmfile, '/raid/madlr/linelists/'+linelists[i], directory='/raid/madlr/par/')

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

	#spectrum = np.vstack((wavelength, flux)).T
	#np.savetxt(directory+parname, spectrum)

	# Output synthetic spectrum in a format that continuum_div functions will understand (list of arrays) 

	return spectrum

#runMoog(temp=5900, logg=0.1, fe=-0.5, alpha=0.5, elements=[25], abunds=[0.5], solar=[5.43])