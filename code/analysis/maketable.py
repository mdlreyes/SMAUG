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

	all_mnfeerr = []

	# Loop over each file
	for fileid in range(len(filenames)):

		# Read file
		file = pd.read_csv('/Users/miadelosreyes/Documents/Research/MnDwarfs/code/'+filenames[fileid], delimiter='\t')

		# Remove all stars with error > 0.3 dex
		mnfeerr = np.sqrt(np.asarray(file['error([Mn/H])'])**2. + 0.1**2.)
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
		'''
		Teff.append(["{:0.0f}".format(i) for i in file['Temp']])
		logg.append(["{:0.2f}".format(i) for i in file['log(g)']])
		alphafe.append(["{:+0.2f}".format(i) for i in file['[alpha/Fe]']])
		try:
			vt.append(["{:0.2f}".format(i) for i in file['vt']])
		except:
			vt.append(['1.90']*N)
		'''

		# Add stuff with errors
		tempTeff = np.zeros(N, dtype='<U15')
		templogg = np.zeros(N, dtype='<U15')
		tempvt = np.zeros(N, dtype='<U15')
		tempalpha = np.zeros(N, dtype='<U15')
		tempfeh = np.zeros(N, dtype='<U15')
		tempmnfe = np.zeros(N, dtype='<U15')
		for i in range(N):

			# Stellar parameters
			tempTeff[i]  = '$'+"{:0.0f}".format(np.asarray(file['Temp'])[i])+r'\pm'+"{:0.0f}".format(np.asarray(file['error(Teff)'])[i])+'$'
			templogg[i]  = '$'+"{:+0.2f}".format(np.asarray(file['log(g)'])[i])+r'\pm'+"{:0.2f}".format(np.asarray(file['error(logg)'])[i])+'$'
			tempvt[i]  = '$'+"{:0.2f}".format(np.asarray(file['vt'])[i])+r'\pm'+"{:0.2f}".format(np.asarray(file['error(xi)'])[i])+'$'
			tempalpha[i]  = '$'+"{:+0.2f}".format(np.asarray(file['[alpha/Fe]'])[i])+r'\pm'+"{:0.2f}".format(np.asarray(file['error([alpha/Fe])'])[i])+'$'

			# [Fe/H]
			tempfeh[i]  = '$'+"{:+0.2f}".format(np.asarray(file['[Fe/H]'])[i])+r'\pm'+"{:0.2f}".format(np.asarray(file['error([Fe/H])'])[i])+'$'

			# [Mn/Fe]
			current_mnfe = np.asarray(file['[Mn/H]'])[i] - np.asarray(file['[Fe/H]'])[i]
			current_mnfeerr = np.sqrt(np.asarray(file['error([Mn/H])'])[i]**2. + 0.1**2.)
			tempmnfe[i] = '$'+"{:+0.2f}".format(current_mnfe)+r'\pm'+"{:0.2f}".format(current_mnfeerr)+'$'

		Teff.append(tempTeff)
		logg.append(templogg)
		vt.append(tempvt)
		alphafe.append(tempalpha)
		feh.append(tempfeh)
		mnfe.append(tempmnfe)
		all_mnfeerr.append(np.asarray(file['error([Mn/H])']))#np.sqrt(mnfeerr**2. - 0.08**2.))

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

	# Test
	all_mnfeerr = np.hstack((all_mnfeerr[:]))
	all_mnfeerr = all_mnfeerr[all_mnfeerr < 0.3]
	print(len(all_mnfeerr))
	print('Average: ', np.average(all_mnfeerr))

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

def stellarparamstable(inputfolder):
	''' Compute effect of stellar parameter variations on [Mn/Fe] measurements. '''

	# Set up list of all columns that need to be prepped
	dsph = []
	name = []
	Teff125 = []
	Teff250 = []
	logg03 = []
	logg06 = []

	# Open original table with [Mn/Fe] measurements
	orig = pd.read_csv(inputfolder+'bscl5_1200B_final3.csv', delimiter='\t')

	# Remove anything with bad errors!
	orig = orig[orig['error([Mn/H])'] < 0.3]
	orig = orig.reset_index(drop=True)

	# Compute [Mn/Fe]
	orig_MnFe = orig['[Mn/H]'] - orig['[Fe/H]']

	def computevariation(parameter, value):
		'''
		Inputs:
		parameter  	-- 'Teff' or 'logg'
		value 		-- ('125' or '250') for 'Teff', ('03' or '06') for 'logg'
		'''

		# Initialize array to hold variations
		diff = -999*np.ones(len(orig_MnFe))

		# Open files
		upfile = pd.read_csv(inputfolder+'bscl5_stellarparams/'+parameter+'up'+value+'.csv', delimiter='\t')
		downfile = pd.read_csv(inputfolder+'bscl5_stellarparams/'+parameter+'down'+value+'.csv', delimiter='\t')

		# Compute [Mn/Fe] from each file
		up_MnFe = upfile['[Mn/H]'] - upfile['[Fe/H]']
		down_MnFe = downfile['[Mn/H]'] - downfile['[Fe/H]']

		# Loop over all objects in original table
		for i in (orig.index):

			# Check if object is in both the up/down files
			if (np.asarray(orig['Name'])[i] in np.asarray(upfile['Name'])) and (np.asarray(orig['Name'])[i] in np.asarray(upfile['Name'])):

				idx_up = np.where(np.asarray(upfile['Name']) == np.asarray(orig['Name'])[i])[0]
				idx_down = np.where(np.asarray(downfile['Name']) == np.asarray(orig['Name'])[i])[0]

				# Calculate difference from original measurement
				up_diff = np.asarray(orig_MnFe)[i] - np.asarray(up_MnFe)[idx_up]
				down_diff = np.asarray(orig_MnFe)[i] - np.asarray(down_MnFe)[idx_down]

				# Average diff
				avg_diff = (np.abs(up_diff) + np.abs(down_diff))/2.
				diff[i] = avg_diff

				#IDlist.append(np.asarray(orig['ID'])[i])
				#difflist.append(avg_diff)

		print('average: ', np.average(diff))

		return diff #np.asarray(IDlist), np.asarray(difflist)

	# Get average differences for each parameter variation
	orig['temp125'] = computevariation('Teff','125')
	orig['temp250'] = computevariation('Teff','250')
	orig['logg03'] = computevariation('logg','03')
	orig['logg06'] = computevariation('logg','06')

	orig.to_csv(inputfolder+'bscl5_1200B_paramfinal.csv', sep='\t', index=False, columns=['Name','temp125','temp250','logg03','logg06'], float_format="%.2f")

	return

def main():
	make_catalog(['data/bscl5_1200B_final3.csv', 'data/bfor7_1200B_final3.csv', 'data/LeoIb_1200B_final3.csv','data/CVnIa_1200B_final3.csv','data/bumia_1200B_final3.csv','data/UMaIIb_1200B_final3.csv'], obj=['Scl','For','LeoI','CVnIa','UMi','UMaII'], outfile='dsph_catalog')
	#make_catalog(['data/bscl5_1200B_final3.csv', 'data/bfor7_1200B_final3.csv', 'data/LeoIb_1200B_final3.csv','data/CVnIa_1200B_final3.csv'], obj=['Scl','For','LeoI','CVnIa'], outfile='dsph_catalog_abridged')
	make_catalog(['data/7078l1_1200B_final.csv', 'data/7089l1_1200B_final.csv', 'data/n5024b_1200B_final.csv'], obj=['M15','M2','M53'], outfile='gc_catalog')

	# Test
	#make_catalog(['data/bscl5_1200B_final3.csv', 'data/bfor7_1200B_final3.csv', 'data/LeoIb_1200B_final3.csv','data/CVnIa_1200B_final3.csv','data/bumia_1200B_final3.csv','data/UMaIIb_1200B_final3.csv','data/7078l1_1200B_final.csv', 'data/7089l1_1200B_final.csv', 'data/n5024b_1200B_final.csv'], obj=['Scl','For','LeoI','CVnIa','UMi','UMaII','M15','M2','M53'], outfile='dsph_catalog')

	# Stellar parameter variation table
	#stellarparamstable('/Users/miadelosreyes/Documents/Research/MnDwarfs/code/data/')

	return

if __name__ == "__main__":
	main()