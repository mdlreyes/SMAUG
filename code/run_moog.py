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

linelists = np.array(['linelist_Mn4754','linelist_Mn4783','linelist_Mn4823','linelist_Mn5394','linelist_Mn5537','linelist_Mn60136021'])
	
def createPar(name, atmfile, linelist, directory=''):
	"""Create *.par file using *.atm file and linelist."""

	# Define filename
	filestr = directory + name + '.par'

	# Open linelist and get wavelength range to synthesize spectrum
	wavelengths = np.genfromtxt(linelist, skip_header=1, usecols=0)
	wavelengthrange = [ math.floor(wavelengths[0]),math.ceil(wavelegnths[-1]) ]

	# Check if file already exists
	exists, readytowrite = checkFile(filestr)
	if readytowrite:

		# Outfile names:
		out1 = '\''+directory+'/'+name+'.out1\''
		out2 = '\''+directory+'/'+name+'.out2\''

		# If file exists, open file
		with open(filestr, 'wr') as file:

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

for i in range(len(linelists)):
	filestr = 'raid/madlr'+
	createPar()