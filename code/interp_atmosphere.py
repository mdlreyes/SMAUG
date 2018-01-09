# interp_atmosphere.py
# Given atmospheric params and [Mn/H], creates *.atm file
# 
# Contains the following functions: 
# - checkFile: check if input file exists and is non-empty
# - round_to: round either up/down to a given precision
# - find_nearest_temp: find nearest temperatures above/below an 
#						input temp, given uneven temp array
# - getAtm: get path to a *.atm file
# - readAtm: read in a *.atm file and put contents into a list
# - interpolateAtm: for given parameters (T, logg, Fe/H, alpha/Fe), 
#				interpolate from ATLAS9 grid7 and output atmosphere
# - writeAtm: given atmosphere, put contents into a *.atm file
# 
# Created 2 Nov 17
# Updated 4 Jan 18
###################################################################

import os
import numpy as np
import math

def checkFile(filestr, overridecheck=True):
	"""Check if file exists and is non-empty.

	Inputs:
	filestr -- filename

	Keyword arguments:
    overridecheck -- 'True' if file is writeable and I want to make sure I don't accidentally overwrite it

    Outputs:
    exists -- 'True' if file exists already, 'False' if not
    readytowrite -- 'True' if writeable file is safe to overwrite, 'False' if not
    """

	# Check if file exists and has contents
	if os.path.isfile(filestr) and os.path.getsize(filestr) > 0:

		# If this is for a writeable file, make sure I don't accidentally overwrite something
		if overridecheck:
			override = raw_input('Warning: File '+filestr+' already exists! Do you want to overwrite? (y/n) ')
			if override == 'y' or override == 'Y' or override == 'yes':
				exists = True
				readytowrite = True
			else:
				exists = True
				readytowrite = False

		# Otherwise, don't need to worry about overwriting
		else:
			exists = True
			readytowrite = True

	# If file doesn't exist or has no contents, don't have to worry about overwriting
	else:
		exists = False
		readytowrite = True
		print('File '+filestr+' doesn\'t exist... yet?')

	return exists, readytowrite

def round_to(n, precision, updown):
	"""Round number n (either up or down) to given precision."""
	precision = float(precision)

	if updown == 'up':
		roundn = (math.ceil(n / precision) * precision)
	else:
		roundn = (math.floor(n / precision) * precision)
    
	return roundn

def find_nearest_temp(temp):
	"""Given a temperature, find nearest temperature (both up and down) in grid."""

	# Array of temperatures in grid
	tarray = np.array([3500, 3600, 3700, 3800, 3900,
				4000, 4100, 4200, 4300, 4400,
				4500, 4600, 4700, 4800, 4900,
				5000, 5100, 5200, 5300, 5400,
				5500, 5600, 5800, 6000, 6200,
				6400, 6600, 6800, 7000, 7200,
				7400, 7600, 7800, 8000])

	# Find what grid temp is closest to the input temp
	idx = (np.abs(tarray-temp)).argmin()
	neartemp 	= tarray[idx]

	# Define output variables
	uptemp 		= 0
	downtemp	= 0
	error 		= False

	# If the closest grid temp is **lower** than input temp
	if neartemp < temp:

		# Check to make sure the input temp isn't outside the grid
		if idx == len(tarray) - 1:
			error = True

		# If ok, then can find the closest grid temp that's **higher** than input temp
		else:
			downtemp = neartemp
			uptemp 	 = tarray[idx+1]

	# If the closest grid temp is **higher** than input temp
	elif neartemp > temp:

		# Check to make sure the input temp isn't outside the grid
		if idx == 0:
			error = True 

		# If ok, then can find the closest grid temp that's **lower** than input temp
		else:
			downtemp = tarray[idx-1]
			uptemp	 = neartemp

	# Check if input temp is equal to one of the grid temps
	else:
		uptemp	 = neartemp
		downtemp = neartemp 

	# Return temperatures and error message if input temp is outside grid range
	return uptemp, downtemp, error

def getAtm(temp, logg, fe, alpha, directory):
	"""Get path of *.atm file."""

	filebase	= 't' + str(temp) + 'g_' + '{:02}'.format(logg)

	# Note different sign conventions for [Fe/H], [alpha/Fe]
	if fe < 0:
		fepart 	= 'f' + '{:03}'.format(fe)
	else:
		fepart	= 'f_' + '{:02}'.format(fe)

	if alpha < 0:
		alphapart 	= 'a' + '{:03}'.format(alpha) + '.atm'
	else:
		alphapart 	= 'a_' + '{:02}'.format(alpha) + '.atm'

	filestr	= directory + filebase + fepart + alphapart

	return filestr

def readAtm(temp, logg, fe, alpha):
	"""Read *.atm file."""

	temp  = temp
	logg  = int(logg*10)
	fe 	  = int(fe*10)
	alpha = int(alpha*10)

	# Directory to read atmospheres from
	directory	= '/raid/grid7/atmospheres/t' + str(temp) + '/g_' + '{:02}'.format(logg) + '/' 

	# Atmosphere to read
	filestr = getAtm(temp, logg, fe, alpha, directory)

	# Check if file already exists
	exists, readytowrite = checkFile(filestr, overridecheck=False)
	if exists:

		#If file exists, read contents
		contents = np.genfromtxt(filestr, skip_header=3, max_rows=72, usecols=None, autostrip=True)
		return contents

	else:
		print('File doesn\'t exist!')
		return

def interpolateAtm(temp, logg, fe, alpha):
	"""Interpolate atmosphere from ATLAS9 grid.

    Inputs:
    temp -- effective temperature (K)
    logg -- surface gravity
    fe -- [Fe/H]
    alpha -- [alpha/Fe]
    """

	# Change input parameters to correct format for filenames
	#temp  = temp
	#logg  = int(logg*10)
	#fe 	  = int(fe*10)
	#alpha = int(alpha*10)

	# Get nearest gridpoints for each parameter
	tempUp, tempDown, tempError = find_nearest_temp(temp)

	loggUp = round_to(logg, 0.5, 'up')
	loggDown = round_to(logg, 0.5, 'down')

	feUp = round_to(fe, 0.1, 'up')
	feDown = round_to(fe, 0.1, 'down')


	alphaUp = round_to(alpha, 0.1, 'up')
	alphaDown = round_to(alpha, 0.1, 'down')

	# Check that points are within range of grid
	if tempError:
		print('Error: T = ' + str(temp) + ' is out of range!')
		return

	if loggUp > 5.0 or loggDown < 0:
		print('Error: log(g) = ' + str(logg) + ' is out of range!')
		return

	elif feUp > 0 or feDown < -5.0:
		print('Error: [Fe/H] = ' + str(fe) + ' is out of range!')
		return

	elif alphaUp > 1.2 or alphaDown < -0.8:
		print('Error: [alpha/Fe] = ' + str(alpha) + ' is out of range!')
		return

	# Grid isn't uniform, so do additional checks to make sure points are within range of grid
	elif logg < 0.5 and temp >= 7000:
		print('Error: T = ' + str(temp) + ' and log(g) = ' + str(logg) + ' out of range!')
		return

	#elif (temp > 3700) & (temp < 4200) & (logg > 40) & (fe <= -4.8):
	#	print('Error: Out of range!') 
	#	return

	# If within grid, interpolate
	else:
		print('Interpolating: ')
		print('Temps = ', tempUp, tempDown)
		print('logg = ', loggUp, loggDown)
		print('fe = ', feUp, feDown)
		print('alpha = ', alphaUp, alphaDown)

		# Comment out for actual run
		return
		
		# Calculate intervals for each variable
		# (quantities needed for interpolation)
		#######################################

		# Temperature interval
		## Check if input temp exactly matches one of the grid points
		if tempUp == tempDown:

			# If so, interpolation interval is just one point,
			# so interval is just one point, and delta(T) and n(T) are both 1
			tempInterval = [temp]
			tempDelta 	 = [1]
			nTemp 		 = 1

		## If not, then input temp is between two grid points
		else:
			tempInterval = [tempDown, tempUp]
			tempDelta	 = np.absolute([tempUp, tempDown] - temp)
			nTemp 		 = 2

		# Repeat for other variables:
		if loggUp == loggDown:
			loggInterval = [logg]
			loggDelta 	 = [1]
			nLogg		 = 1
		else:
			loggInterval = [loggDown, loggUp]
			loggDelta	 = np.absolute([loggUp, loggDown] - logg)
			nLogg 		 = 2

		if feUp == feDown:
			feInterval	= [fe]
			feDelta 	= [1]
			nFe		 	= 1
		else:
			feInterval	= [feDown, feUp]
			feDelta		= np.absolute([feUp, feDown] - fe)
			nFe 		= 2

		if alphaUp == alphaDown:
			alphaInterval	= [alpha]
			alphaDelta 		= [1]
			nAlpha	 		= 1
		else:
			alphaInterval	= [alphaDown, alphaUp]
			alphaDelta		= np.absolute([alphaUp, alphaDown] - alpha)
			nAlpha 			= 2

		# Do interpolation!
		###################
		for i in range(nTemp):
			for j in range(nLogg):
				for m in range(nFe):
					for n in range(nAlpha):

						# Read in grid point (atmosphere file)
						iflux = read_atm(tempInterval[i],loggInterval[j],feInterval[m],alphaInterval[n])[:,0]

						# Compute weighted sum of grid points
						## If first iteration, initialize flux as weighted value of lowest grid point 
						if (i==0) & (j==0) & (m==0) & (n==0):
							flux = tempDelta[i]*loggDelta[j]*feDelta[m]*alphaDelta[n] * iflux

						## Else, start adding weighted values of other grid points
						else:
							flux = flux + tempDelta[i]*loggDelta[j]*feDelta[m]*alphaDelta[n] * iflux

		# Normalize by dividing by correct intervals
		quotient = 1.0
		if nTemp == 2:
			quotient = quotient * (tempUp - tempDown)
		if nLogg == 2:
			quotient = quotient * (loggUp - loggDown)
		if nFe == 2:
			quotient = quotient * (feUp - feDown)
		if nAlpha == 2:
			quotient = quotient * (alphaUp - alphaDown)

		flux = flux/(quotient*1.0)

		return flux

def writeAtm(temp, logg, fe, alpha, dir='/raid/madlr', elements=None, abunds=None):
	"""Create *.atm file

    Inputs:
    temp 	 -- effective temperature (K)
    logg 	 -- surface gravity
    fe 		 -- [Fe/H]
    alpha 	 -- [alpha/Fe]

    Keywords:
    dir 	 -- directory to write atmospheres to [default = '/raid/madlr']
    elements -- list of elements to add to the list of atoms
    abunds 	 -- list of elemental abundances corresponding to list of elements
    """

	# Atmosphere to write
	filestr = getAtm(temp, logg, fe, alpha, directory)
	printstr = str(temp) + './' + ('%.2f' % float(logg)) + '/' + ('%5.2f' % float(fe)) + '/' + ('%5.2f' % float(alpha))

	# Check if file already exists
	exists, readytowrite = checkFile(filestr, overridecheck=False)
	if readytowrite:

		# Get atmosphere data
		#####################
		atmosphere = interpolateSpectrum(temp,logg,fe,alpha)

		# Header text
		#############
		headertxt = 'KURUCZ\n' +
					printstr +
					'\nntau=      72'

		# Footer text
		#############
		microturbvel 	= atmosphere[0,6]

		# If not adding any elements, use default NATOMS footer
		if elements is None:
			natoms = 6
			atomstxt = ('%.3E' % microturbvel) +
					'\nNATOMS    ' + str(natoms) + '   ' + ('%5.2f' % float(fe)) +
					'\n      12      ' + ('%5.2f' % float(7.38 + fe)) +
					'\n      14      ' + ('%5.2f' % float(7.35 + fe)) +
					'\n      16      ' + ('%5.2f' % float(7.01 + fe)) +
					'\n      18      ' + ('%5.2f' % float(6.36 + fe)) +
					'\n      20      ' + ('%5.2f' % float(6.16 + fe)) +
					'\n      22      ' + ('%5.2f' % float(4.79 + fe))

		# If adding elements, first make sure that number of elements matches number of abundances
		elif len(elements) != len(abunds):
			print('ERROR: length of element array doesn\'t match length of abundances array')
			return

		else:
			natoms = 6 + len(elements)
			atomstxt = ('%.3E' % microturbvel) +
					'\nNATOMS    ' + str(natoms) + '   ' + ('%5.2f' % float(fe)) +
					'\n      12      ' + ('%5.2f' % float(7.38 + fe)) +
					'\n      14      ' + ('%5.2f' % float(7.35 + fe)) +
					'\n      16      ' + ('%5.2f' % float(7.01 + fe)) +
					'\n      18      ' + ('%5.2f' % float(6.36 + fe)) +
					'\n      20      ' + ('%5.2f' % float(6.16 + fe)) +
					'\n      22      ' + ('%5.2f' % float(4.79 + fe))

			# Add the new elements
			for i in range(len(elements)):
				atomstxt = atomstxt + 
					'\n      '+str(elements[i])+'      ' + ('%5.2f' % float(abunds[i]))

		# Create final footer by adding NMOL footer to NATOMS footer
		footertxt = atomstxt +
					'\nNMOL       15' +
					'\n   101.0   106.0   107.0   108.0   606.0   607.0   608.0   707.0' +
					'\n   708.0   808.0 10108.0 60808.0     6.1     7.1     8.1'

		# Save file
		###########
		np.savetxt(filestr, atmosphere, header=headertxt, delimiter=' ', 
			fmt=['%10.9E','%9.1f','%10.4E','%10.4E','%10.4E','%10.4E','%10.4E'],
			footer=footertxt)

		return filestr

test = interpolateSpectrum(temp=6900, logg=0.1, fe=-0.5, alpha=0.5)
writeAtm(temp=6900, logg=0.1, fe=-0.5, alpha=0.5)