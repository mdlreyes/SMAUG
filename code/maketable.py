# Script to make LaTeX tables
#
######################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

def make_catalog(filenames, obj, outfile, outfilepath='/Users/miadelosreyes/Documents/Research/MnDwarfs/code/tables/'):
	"""Code to make catalog of [Mn/Fe] data.
		
		Inputs:
		filenames	-- list of files to input
		obj 		-- list of object names (one for each file)
		outfile 	-- output file
	"""

	# Set up list of all columns that need to be prepped
	dsph = []
	name = []
	RA = []
	Dec = []
	Teff = []
	logg = []
	vt = []
	alphafe = []
	feh = []
	mnfe = []

	# Loop over each file
	for fileid in range(len(filenames)):

		# Read file
		file = pd.read_csv(filenames[fileid], delimiter='\t')

		# Remove all stars with error > 0.3 dex
		mnfeerr = np.sqrt(np.asarray(file['error([Mn/H])'])**2. + np.asarray(file['error([Fe/H])'])**2. + 0.08**2.)
		print(np.where((mnfeerr > 0.3))[0])
		file = file.drop(np.where((mnfeerr > 0.3))[0])

		N = len(file['Name'])

		# Get names
		dsph.append(np.asarray([obj[fileid]]*N))
		name.append(np.asarray(file['Name'].astype('str')))

		# Get coordinates
		tempRA 	= np.zeros(N, dtype='<U12')
		tempDec = np.zeros(N, dtype='<U12')
		for i in range(N):
			c = SkyCoord(np.asarray(file['RA'])[i],np.asarray(file['Dec'])[i],unit='deg')
			tempRA[i] = c.to_string('hmsdms',precision=2)[:12]
			tempDec[i] = c.to_string('hmsdms',precision=2)[13:]
		RA.append(tempRA)
		Dec.append(tempDec)

		# Add stuff without errors
		Teff.append(["{:0.0f}".format(i) for i in file['Temp']])
		logg.append(["{:0.2f}".format(i) for i in file['log(g)']])
		alphafe.append(["{:+0.2f}".format(i) for i in file['[alpha/Fe]']])

		try:
			vt.append(["{:0.2f}".format(i) for i in file['vt']])
		except:
			vt.append(['1.90']*N)

		# Add stuff with errors
		tempfeh = np.zeros(N, dtype='<U15')
		tempmnfe = np.zeros(N, dtype='<U15')
		for i in range(N):
			tempfeh[i]  = '$'+"{:+0.2f}".format(np.asarray(file['[Fe/H]'])[i])+r'\pm'+"{:0.2f}".format(np.asarray(file['error([Fe/H])'])[i])+'$'
			current_mnfe = np.asarray(file['[Mn/H]'])[i] - np.asarray(file['[Fe/H]'])[i]
			current_mnfeerr = np.sqrt(np.asarray(file['error([Mn/H])'])[i]**2. + np.asarray(file['error([Fe/H])'])[i]**2. + 0.08**2.)
			tempmnfe[i] = '$'+"{:+0.2f}".format(current_mnfe)+r'\pm'+"{:0.2f}".format(current_mnfeerr)+'$'
		feh.append(tempfeh)
		mnfe.append(tempmnfe)

	# Stack all the data together
	dsph = np.hstack((dsph[:]))
	name = np.hstack((name[:]))
	RA   = np.hstack((RA[:]))
	Dec  = np.hstack((Dec[:]))
	Teff = np.hstack((Teff[:]))
	logg = np.hstack((logg[:]))
	vt   = np.hstack((vt[:]))
	alphafe = np.hstack((alphafe[:]))
	feh  = np.hstack((feh[:]))
	mnfe = np.hstack((mnfe[:]))

	# List containing all columns to be put into table
	listcol = [dsph, name, RA, Dec, Teff, logg, vt, alphafe, feh, mnfe]
	#print(listcol)
	Ncols	= len(listcol)

	# Open text file
	workfile	= outfilepath+outfile+'.txt'
	f = open(workfile, 'w')

	for i in range(len(dsph)):
		for j in range(Ncols):
			f.write(listcol[j][i])
			if j < Ncols - 1:
				f.write(' & ')
			else:
				f.write(' \\\\\n')

	return

def main():
	make_catalog(['data/bscl5_1200B_final3.csv', 'data/bfor7_1200B_final3.csv', 'data/LeoIb_1200B_final3.csv','data/CVnIa_1200B_final3.csv','data/bumia_1200B_final3.csv','data/UMaIIb_1200B_final3.csv'], obj=['Scl','For','LeoI','CVnIa','UMi','UMaII'], outfile='dsph_catalog')
	make_catalog(['data/7078l1_1200B_final.csv', 'data/7089l1_1200B_final.csv', 'data/n5024b_1200B_final.csv'], obj=['M15','M2','M53'], outfile='gc_catalog')

	return

if __name__ == "__main__":
	main()