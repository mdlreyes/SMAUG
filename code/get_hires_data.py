# get_hires_data.py
# Format high-resolution data into files that can be read by other
# codes (plot_mn.py, fit_mnfe_feh.py)
#
# Created 29 Nov 18
# Updated 29 Nov 18
###################################################################

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import math
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from matplotlib.ticker import NullFormatter
from statsmodels.stats.weightstats import DescrStatsW

def north_etal_12(coordfile, datafile1, datafile2, outfile, scl=True):
	"""Get data from North+12 paper

	Inputs:
	coordfile 	-- table of coordinates
	datafile1 	-- table with abundances for lines 5407, 5420
	datafile1 	-- table with abundances for lines 5432, 5516
	outfile 	-- name of output file

	Keywords:
	scl 		-- if 'True' (default), use Sculptor coord file; else, use Letarte+10 (for Fornax)
	"""

	# Open files
	if scl:
		coorddata = pd.read_fwf(coordfile, colspecs=[(0,6),(7,17),(18,30)])
		N = len(coorddata['Name'])

	else:
		coorddata = pd.read_csv(coordfile, delimiter='\t')
		N = len(coorddata['ID'])

	data1 	= pd.read_csv(datafile1, delimiter='\t', na_values = ' ')
	data2 	= pd.read_csv(datafile2, delimiter='\t', na_values = ' ')

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile
	with open(outputname, 'w+') as f:
		f.write('Name\tRA\tDec\tTemp\tlog(g)\t[Fe/H]\terror([Fe/H])\t[alpha/Fe]\t[Mn/Fe]\terror([Mn/Fe])\tchisq(reduced)\n')

	# Loop over all stars listed in coordinate file
	for i in range(N):

		if scl:
			starname = coorddata['Name'][i][:2] + coorddata['Name'][i][3:]
		else:
			starname = coorddata['ID'][i]

		name1 = data1['Star']
		name2 = data2['Star']

		# Check to see if star has values in datafiles
		if (starname in name1.values) and (starname in name2.values):

			idx1 = np.where((name1.values == starname))[0]
			idx2 = np.where((name2.values == starname))[0]

			# Check to see if star has [Mn/Fe] for all the requisite lines
			if (pd.isnull(data1['[Mn/Fe]5407'].values[idx1][0]) == False) and (pd.isnull(data1['[Mn/Fe]5420'].values[idx1][0]) == False) and (pd.isnull(data2['[Mn/Fe]5516'].values[idx2][0]) == False):
			
				# Get [Mn/H], [Mn/H]error for each line, [Fe/H]
				mnfe5407 = pd.to_numeric(data1['[Mn/Fe]5407'])[idx1].values[0]
				mnfe5420 = pd.to_numeric(data1['[Mn/Fe]5420'])[idx1].values[0]
				mnfe5516 = pd.to_numeric(data2['[Mn/Fe]5516'])[idx2].values[0]

				mnh5407 = pd.to_numeric(data1['[Mn/H]5407'])[idx1].values[0]
				mnh5420 = pd.to_numeric(data1['[Mn/H]5420'])[idx1].values[0]
				mnh5516 = pd.to_numeric(data2['[Mn/H]5516'])[idx2].values[0]

				mnfeerr5407 = pd.to_numeric(data1['[Mn/Fe]error5407'])[idx1].values[0]
				mnfeerr5420 = pd.to_numeric(data1['[Mn/Fe]error5420'])[idx1].values[0]
				mnfeerr5516 = pd.to_numeric(data2['[Mn/Fe]error5516'])[idx2].values[0]

				feh = mnh5407 - mnfe5407

				# Average all of the [Mn/Fe] to get a final abundance
				mnfe = np.average([mnfe5407, mnfe5420, mnfe5516], weights=[1./(mnfeerr5407**2.),1./(mnfeerr5420**2.),1./(mnfeerr5516**2.)])
				mnfeerr = 1./np.sqrt((1./mnfeerr5407)**2. + (1./mnfeerr5420)**2. + (1./mnfeerr5516)**2.)

				# Get coordinates
				starra = coorddata['RA'][i]
				stardec = coorddata['Dec'][i]
				coord = SkyCoord(starra+' '+stardec, frame='icrs', unit=(u.hourangle, u.deg))

				# Put all other info together
				RA 		= coord.ra.degree
				Dec 	= coord.dec.degree
				temp 	= 0.
				logg	= 0.
				feherr 	= 0.
				alphafe = 0.
				chisq 	= 0.

				# Write star data to file
				with open(outputname, 'a') as f:
					f.write(starname+'\t'+str(RA)+'\t'+str(Dec)+'\t'+str(temp)+'\t'+str(logg)+'\t'+str(feh)+'\t'+str(feherr)+'\t'+str(alphafe)+'\t'+str(mnfe)+'\t'+str(mnfeerr)+'\t'+str(chisq)+'\n')


		else:
			continue

	return

def yong_etal_14(datafile, outfile):
	"""Get data from Yong+14 paper

	Inputs:
	datafile 	-- datafile
	outfile 	-- name of output file
	"""

	# Open files
	data = pd.read_csv(datafile, delimiter='\t')
	N = len(data['Name'])

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile

	# Prep output array
	name 	= data['Name']
	ra 		= np.zeros(N)
	dec 	= np.zeros(N)
	temp 	= np.zeros(N)
	logg 	= np.zeros(N)
	feh 	= np.zeros(N)
	sigfeh 	= np.zeros(N)
	alphafe = np.zeros(N)
	mnfe 	= data['[Mn/Fe]']
	sigmnfe = data['Error']
	rchisq 	= np.zeros(N)

	# Get coordinates
	for i in range(N):
		starra  = data['RA'][i]
		stardec = data['Dec'][i]
		coord = SkyCoord(starra+' '+stardec, frame='icrs', unit=(u.hourangle, u.deg))

		ra[i] 		= coord.ra.degree
		dec[i] 	= coord.dec.degree

	# Write output
	cols = ['Name','RA','Dec','Temp','log(g)','[Fe/H]','error([Fe/H])','[alpha/Fe]','[Mn/Fe]','error([Mn/Fe])','chisq(reduced)']
	output = pd.DataFrame(np.array([name,ra,dec,temp,logg,feh,sigfeh,alphafe,mnfe,sigmnfe,rchisq]).T, columns=cols)
	output.to_csv(outputname, sep='\t', index=False)

	return

def apogee(datafile, outfile):
	"""Get data from APOGEE

	Inputs:
	datafile 	-- datafile
	outfile 	-- name of output file
	"""

	# Open files
	data = pd.read_csv(datafile, delimiter=',')
	N = len(data['apstar_id'])

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile

	# Prep output array
	name 	= data['apogee_id']
	ra 		= data['ra']
	dec 	= data['dec']
	temp 	= data['teff']
	logg 	= data['logg']
	feh 	= data['fparam_m_h']
	sigfeh 	= np.zeros(N)
	alphafe = data['fparam_alpha_m']
	mnfe 	= data['mn_fe']
	sigmnfe = data['mn_fe_err']
	rchisq 	= np.zeros(N)

	# Write output
	cols = ['Name','RA','Dec','Temp','log(g)','[Fe/H]','error([Fe/H])','[alpha/Fe]','[Mn/Fe]','error([Mn/Fe])','chisq(reduced)']
	output = pd.DataFrame(np.array([name,ra,dec,temp,logg,feh,sigfeh,alphafe,mnfe,sigmnfe,rchisq]).T, columns=cols)
	output.to_csv(outputname, sep='\t', index=False)

	return

def ivans_etal_01(datafile, paramfile, outfile):
	"""Get data from Ivans+01 paper

	Inputs:
	paramfile 	-- paramfile
	datafile 	-- datafile with metallicities
	outfile 	-- name of output file
	"""

	# Open files
	params = pd.read_csv(paramfile, delimiter='\t')
	data = pd.read_csv(datafile, delimiter='\t')
	N = len(data['Star'])

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile

	# Prep output array
	name 	= params['Star']
	ra 		= params['RA']
	dec 	= params['Dec']
	temp 	= params['spec_Teff']
	logg 	= params['spec_logg']
	feh 	= params['spec_feh_mean']
	sigfeh 	= np.zeros(N)
	alphafe = np.zeros(N)
	mnfe 	= data['Mn(I)']
	sigmnfe = np.zeros(N)
	rchisq 	= np.zeros(N)

	# Get coordinates
	for i in range(N):
		starra  = params['RA'][i]
		stardec = params['Dec'][i]
		coord = SkyCoord(starra+' '+stardec, frame='icrs', unit=(u.hourangle, u.deg))

		ra[i] 	= coord.ra.degree
		dec[i] 	= coord.dec.degree

	# Write output
	cols = ['Name','RA','Dec','Temp','log(g)','[Fe/H]','error([Fe/H])','[alpha/Fe]','[Mn/Fe]','error([Mn/Fe])','chisq(reduced)']
	output = pd.DataFrame(np.array([name,ra,dec,temp,logg,feh,sigfeh,alphafe,mnfe,sigmnfe,rchisq]).T, columns=cols)
	output.to_csv(outputname, sep='\t', index=False)

	return

def ramirez_cohen_03(datafile, paramfile, outfile):
	"""Get data from Ramirez & Cohen 03 paper

	Inputs:
	paramfile 	-- paramfile
	datafile 	-- datafile with metallicities
	outfile 	-- name of output file
	"""

	# Open files
	params = pd.read_csv(paramfile, delimiter='\t')
	data = pd.read_csv(datafile, delimiter='\t')
	N = len(data['Star'])

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile

	# Prep output array
	name 	= params['ID']
	ra 		= data['RA']
	dec 	= data['Dec']
	temp 	= params['Teff']
	logg 	= params['logg']
	feh 	= np.zeros(N)
	sigfeh 	= np.zeros(N)
	alphafe = np.zeros(N)
	mnfe 	= data['[Mn/Fe]']
	sigmnfe = data['err[Mn/Fe]']
	rchisq 	= np.zeros(N)

	# Get coordinates
	for i in range(N):
		starra  = ra[i]
		stardec = dec[i]
		print(stardec)
		coord = SkyCoord(str(starra)+' '+str(stardec), frame='icrs', unit=(u.hourangle, u.deg))

		ra[i] 	= coord.ra.degree
		dec[i] 	= coord.dec.degree

	# Write output
	cols = ['Name','RA','Dec','Temp','log(g)','[Fe/H]','error([Fe/H])','[alpha/Fe]','[Mn/Fe]','error([Mn/Fe])','chisq(reduced)']
	output = pd.DataFrame(np.array([name,ra,dec,temp,logg,feh,sigfeh,alphafe,mnfe,sigmnfe,rchisq]).T, columns=cols)
	output.to_csv(outputname, sep='\t', index=False)

	return

def lai_etal_11(datafile, outfile):
	"""Get data from Lai+11 paper

	Inputs:
	paramfile 	-- paramfile
	datafile 	-- datafile with metallicities
	outfile 	-- name of output file
	"""

	# Open files
	data = pd.read_csv(datafile, delimiter='\t')
	N = len(data['Name'])

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile

	# Prep output array
	name 	= data['Name']
	ra 		= data['_RA']
	dec 	= data['_DE']
	temp 	= data['Teff']
	logg 	= data['logg']
	feh 	= data['[Fe/H]']
	sigfeh 	= data['e_[Fe/H]']
	alphafe = np.zeros(N)
	mnfe 	= data['[Mn/Fe]']
	sigmnfe = data['e_[Mn/Fe]']
	rchisq 	= np.zeros(N)

	# Write output
	cols = ['Name','RA','Dec','Temp','log(g)','[Fe/H]','error([Fe/H])','[alpha/Fe]','[Mn/Fe]','error([Mn/Fe])','chisq(reduced)']
	output = pd.DataFrame(np.array([name,ra,dec,temp,logg,feh,sigfeh,alphafe,mnfe,sigmnfe,rchisq]).T, columns=cols)
	output.to_csv(outputname, sep='\t', index=False)

	return

def sobeck_etal_11(datafile, outfile):
	"""Get data from Sobeck+11 paper

	Inputs:
	datafile 	-- datafile with metallicities
	outfile 	-- name of output file
	"""

	# Open files
	data = pd.read_csv(datafile, delimiter='\t', skiprows=10)
	N = len(data['Star'])

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile

	# Prep output array
	name 	= data['Star']
	ra 		= data['RA']
	dec 	= data['Dec']
	temp 	= data['spec_T_eff']
	logg 	= data['spec_log(g)']
	feh 	= data['spec_[Fe/H]']
	sigfeh 	= np.zeros(N)
	alphafe = np.zeros(N)
	mnfe 	= data['[Mn/Fe]']
	sigmnfe = data['sig[Mn/Fe]']
	rchisq 	= np.zeros(N)

	# Write output
	cols = ['Name','RA','Dec','Temp','log(g)','[Fe/H]','error([Fe/H])','[alpha/Fe]','[Mn/Fe]','error([Mn/Fe])','chisq(reduced)']
	output = pd.DataFrame(np.array([name,ra,dec,temp,logg,feh,sigfeh,alphafe,mnfe,sigmnfe,rchisq]).T, columns=cols)
	output.to_csv(outputname, sep='\t', index=False)

	return

def lamb_etal_14(datafile, outfile):
	"""Get data from Lamb+14 paper

	Inputs:
	datafile 	-- datafile with metallicities
	outfile 	-- name of output file
	"""

	# Open files
	data = pd.read_csv(datafile, delimiter='\t')
	N = len(data['Star'])

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile

	# Prep output array
	name 	= data['Star']
	ra 		= data['RA']
	dec 	= data['Dec']
	temp 	= data['Teff']
	logg 	= data['logg']
	feh 	= data['[Fe/H]']
	sigfeh 	= data['sig[Fe/H]']
	alphafe = np.zeros(N)
	mnfe 	= data['[Mn/Fe]']
	sigmnfe = data['sig[Mn/Fe]']
	rchisq 	= np.zeros(N)

	# Write output
	cols = ['Name','RA','Dec','Temp','log(g)','[Fe/H]','error([Fe/H])','[alpha/Fe]','[Mn/Fe]','error([Mn/Fe])','chisq(reduced)']
	output = pd.DataFrame(np.array([name,ra,dec,temp,logg,feh,sigfeh,alphafe,mnfe,sigmnfe,rchisq]).T, columns=cols)
	output.to_csv(outputname, sep='\t', index=False)

	return

def sobeck_etal_06(datafile, cluster, outfile):
	"""Get data from Sobeck+06 paper

	Inputs:
	datafile 	-- datafile with metallicities
	cluster 	-- name of GC to get data for
	outfile 	-- name of output file
	"""

	# Open files
	data = pd.read_csv(datafile, delimiter='\t')

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile

	# Get data for a given cluster
	gc = data['OName']
	mask = np.where(np.asarray(gc)==cluster)[0]
	N = len(mask)

	# Prep output array
	name 	= data['ID'][mask]
	ra 		= data['_RA'][mask]
	dec 	= data['_DE'][mask]
	temp 	= data['Teff'][mask]
	logg 	= data['log(g)'][mask]
	feh 	= data['[Fe/H]'][mask]
	sigfeh 	= np.zeros(N)
	alphafe = np.zeros(N)
	mnfe 	= data['[Mn/Fe]'][mask]
	sigmnfe = np.zeros(N)
	rchisq 	= np.zeros(N)

	# Write output
	cols = ['Name','RA','Dec','Temp','log(g)','[Fe/H]','error([Fe/H])','[alpha/Fe]','[Mn/Fe]','error([Mn/Fe])','chisq(reduced)']
	output = pd.DataFrame(np.array([name,ra,dec,temp,logg,feh,sigfeh,alphafe,mnfe,sigmnfe,rchisq]).T, columns=cols)
	output.to_csv(outputname, sep='\t', index=False)

	return

def saga_hires(datafile, outfile):
	"""Get data from a SAGA database output file

	Inputs:
	datafile 	-- datafile
	outfile 	-- name of output file
	"""

	# Open files
	data = pd.read_csv(datafile, delimiter='\t')
	N = len(data['Object'])

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile

	# Prep output array
	name 	= data['Object']
	ra 		= data['RA']
	dec 	= data['Decl']
	temp 	= data['Teff']
	logg 	= data['logg']
	feh 	= data['[Fe/H]']
	sigfeh 	= data['[Fe/H]error']
	alphafe = np.zeros(N)
	mnfe 	= data['[Mn/Fe]']
	sigmnfe = data['[Mn/Fe]error']
	rchisq 	= np.zeros(N)

	# Get coordinates
	for i in range(N):
		starra  = data['RA'][i]
		stardec = data['Decl'][i]
		coord = SkyCoord(starra+' '+stardec, frame='icrs', unit=(u.hourangle, u.deg))

		ra[i] 		= coord.ra.degree
		dec[i] 	= coord.dec.degree

	# Write output
	cols = ['Name','RA','Dec','Temp','log(g)','[Fe/H]','error([Fe/H])','[alpha/Fe]','[Mn/Fe]','error([Mn/Fe])','chisq(reduced)']
	output = pd.DataFrame(np.array([name,ra,dec,temp,logg,feh,sigfeh,alphafe,mnfe,sigmnfe,rchisq]).T, columns=cols)
	output.to_csv(outputname, sep='\t', index=False)

	return

def match_hires(hiresfilelist, sourcelist, obsfile, outputname):
	"""Match hi-res and med-res Mn catalogs

	Inputs:
	hiresfilelist 	-- list of filenames of hi-resolution catalogs
	sourcelist  -- list of names of hi-resolution catalogs (for plotting purposes)
	obsfile 	-- name of not-hi-res catalog
	"""

	# Create lists for final outputs
	matched_medres = []
	matched_hires = []
	matched_medres_err = []
	matched_hires_err = []
	separation = []
	source = []

	# Get names of stars for checking
	names = np.genfromtxt(obsfile, skip_header=1, delimiter='\t', usecols=0, dtype='str')

	# Get RA and Dec of stars from (our) med-res file
	medresRA = np.genfromtxt(obsfile, skip_header=1, delimiter='\t', usecols=1, dtype='str')
	medresDec = np.genfromtxt(obsfile, skip_header=1, delimiter='\t', usecols=2, dtype='str')
	medrescatalog = SkyCoord(ra=medresRA, dec=medresDec, unit=(u.deg, u.deg)) 

	# Compute [Mn/Fe] abundance of stars from med-res file
	medresFeH = np.genfromtxt(obsfile, skip_header=1, delimiter='\t', usecols=5, dtype='float')
	medresMnH = np.genfromtxt(obsfile, skip_header=1, delimiter='\t', usecols=8, dtype='float')

	medresFeHerror = np.genfromtxt(obsfile, skip_header=1, delimiter='\t', usecols=6, dtype='float')
	medresMnHerror = np.genfromtxt(obsfile, skip_header=1, delimiter='\t', usecols=9, dtype='float')

	medresMnFe 	  = medresMnH - medresFeH
	medresMnFeerror = np.sqrt(np.power(medresFeHerror,2.) + np.power(medresMnHerror,2.))

	for filenum in range(len(hiresfilelist)):

		hiresfile = hiresfilelist[filenum]

		# Get RA and Dec of stars from (lit) hi-res datafile
		hiresRA = np.genfromtxt(hiresfile, skip_header=1, delimiter='\t', usecols=1, dtype='str')
		hiresDec = np.genfromtxt(hiresfile, skip_header=1, delimiter='\t', usecols=2, dtype='str')

		# Get [Mn/Fe] abundance of stars from hi-res file
		hiresMnFe = np.genfromtxt(hiresfile, skip_header=1, delimiter='\t', usecols=8, dtype='float')
		hiresMnFeerror  = np.genfromtxt(hiresfile, skip_header=1, delimiter='\t', usecols=9, dtype='float')

		# Do some formatting in case there's only one hi-res star
		if len(np.atleast_1d(hiresRA)) == 1:
			hiresRA = np.array([hiresRA])
			hiresDec = np.array([hiresDec])
			hiresMnFe = np.array([hiresMnFe])
			hiresMnFeerror = np.array([hiresMnFeerror])

		# Loop over each star from hires list
		for i in range(len(np.atleast_1d(hiresRA))):

			# Convert coordinates to decimal form
			coord = SkyCoord(hiresRA[i], hiresDec[i], unit=(u.deg, u.deg))

			# Search for matching star in our catalog 
			idx, sep, _ = coord.match_to_catalog_sky(medrescatalog) 

			if sep.arcsecond[0] < 10.:

				#print('Separation: ', sep.arcsecond)
				print(names[idx])

				matched_medres.append(medresMnFe[idx])
				matched_medres_err.append(medresMnFeerror[idx])
				matched_hires.append(hiresMnFe[i])
				matched_hires_err.append(hiresMnFeerror[i])
				separation.append(sep.arcsecond[0])
				source.append(sourcelist[filenum])

				# Test code
				#if medresMnFe[idx] > 0.2:
				#	print(names[idx])

	np.savetxt(outputname, np.asarray((matched_medres, matched_medres_err, matched_hires, matched_hires_err, separation)).T, delimiter='\t', header='our [Mn/H]\terror(our [Mn/H])\thires [Mn/H]\terror(hires [Mn/H])\tseparation (arcsec)')

	# Add source name to list
	with open(outputname, 'r') as f:
		file_lines = [''.join([x.strip(), '\t'+sourcelist[filenum], '\n']) for x in f.readlines()]

	with open(outputname, 'w') as f:
		f.writelines(file_lines)

	return

def main():
	# Sculptor
	#north_etal_12('data/hires_data/scl/scl_north_sample.coord','data/hires_data/scl/Sculptor_north_tab1.tsv','data/hires_data/scl/Sculptor_north_tab2.tsv','scl/north12_final.csv')
	#for file in ['shetrone03','geisler05','jablonka15','simon15','tafelmeyer10','skuladottir15']:
	#	saga_hires('data/hires_data/'+file+'.csv','scl/'+file+'_final.csv')

	# Fornax
	#north_etal_12('data/hires_data/for_north_sample.coord', 'data/hires_data/Fornax_north_tab1.tsv', 'data/hires_data/Fornax_north_tab2.tsv', 'for/north12_final.csv', scl=False)
	for file in ['shetrone03','tafelmeyer10']:
		saga_hires('data/hires_data/for/'+file+'.tsv','for/'+file+'_final.csv')

	# Leo I
	file = 'shetrone03'
	saga_hires('data/hires_data/leoi/'+file+'.csv','leoi/'+file+'_final.csv')

	# Ursa Major II
	#file = 'frebel10'
	#saga_hires('data/hires_data/umaii/'+file+'.csv','umaii/'+file+'_final.csv')

	# Ursa Minor
	#for file in ['cohen10','sadakane04','shetrone01','ural15']:
	#	saga_hires('data/hires_data/umi/'+file+'.csv','umi/'+file+'_final.csv')

	# M2
	#yong_etal_14('data/hires_data/M2_yong.csv','M2_yong_final.csv')
	#apogee('data/hires_data/M2_apogee.csv','M2_apogee_final.csv')

	# M5
	#ivans_etal_01('data/hires_data/M5_ivans_tab.txt', 'data/hires_data/M5_ivans_params.txt', 'M5_ivans_final.csv')
	#ramirez_cohen_03('data/hires_data/M5_ramirez_tab.txt','data/hires_data/M5_ramirez_params.txt', 'M5_ramirez_final.csv')
	#lai_etal_11('data/hires_data/M5_lai.tsv','M5_lai_final.csv')
	#apogee('data/hires_data/M5_apogee.csv','M5_apogee_final.csv')
	#sobeck_etal_06('data/hires_data/M5_M15_sobeck.tsv','M5','M5_sobeck06_final.csv')

	# M15
	#sobeck_etal_11('data/hires_data/M15_sobeck.txt','M15_sobeck11_final.csv')
	#apogee('data/hires_data/M15_apogee.csv','M15_apogee_final.csv')
	#sobeck_etal_06('data/hires_data/M5_M15_sobeck.tsv','M15','M15_sobeck06_final.csv')

	# M53
	#lamb_etal_14('data/hires_data/M53_lamb.txt', 'M53_lamb_final.csv')
	#apogee('data/hires_data/M53_apogee.csv','M53_apogee_final.csv')

	# Match hi-res files for GCs
	#match_hires(hiresfilelist=['data/hires_data_final/GCs/M2_apogee_final.csv','data/hires_data_final/GCs/M2_yong_final.csv'], sourcelist=['APOGEE','Yong et al. (2014)'], obsfile='data/7089l1_1200B_final.csv', outputname='data/hires_data_final/GCs/M2_matched.csv')
	#match_hires(hiresfilelist=['data/hires_data_final/GCs/M15_apogee_final.csv','data/hires_data_final/GCs/M15_sobeck06_final.csv','data/hires_data_final/GCs/M15_sobeck11_final.csv'], sourcelist=['APOGEE', 'Sobeck et al. (2006)', 'Sobeck et al. (2011)'], obsfile='data/7078l1_1200B_final.csv', outputname='data/hires_data_final/GCs/M15_matched.csv')
	#match_hires(hiresfilelist=['data/hires_data_final/GCs/M53_apogee_final.csv','data/hires_data_final/GCs/M53_lamb_final.csv'], sourcelist=['APOGEE', 'Lamb et al. (2014)'], obsfile='data/n5024b_1200B_final.csv', outputname='data/hires_data_final/GCs/M53_matched.csv')

	# Match hi-res files for Sculptor
	#files = ['north12','geisler05','jablonka15','shetrone03','simon15','skuladottir15','tafelmeyer10']
	#names = ['North+12','Geiser+05','Jablonka+15','Shetrone+03','Simon+15','Skuladottir+15','Tafelmeyer+10']
	#for i in range(len(files)):
	#	match_hires(hiresfilelist=['data/hires_data_final/scl/'+files[i]+'_final.csv'], sourcelist=[names[i]], obsfile='data/bscl5_1200B_final3.csv', outputname='data/hires_data_final/scl/'+files[i]+'_matched.csv')

	# Match hi-res files for Fornax
	files = ['north12','shetrone03','tafelmeyer10']
	names = ['North+12','Shetrone+03','Tafelmeyer+10']
	for i in range(len(files)):
		match_hires(hiresfilelist=['data/hires_data_final/for/'+files[i]+'_final.csv'], sourcelist=[names[i]], obsfile='data/bfor7_1200B_final3.csv', outputname='data/hires_data_final/for/'+files[i]+'_matched.csv')

	# Match hi-res files for Leo I
	match_hires(hiresfilelist=['data/hires_data_final/leoi/shetrone03_final.csv'], sourcelist=['Shetrone+03'], obsfile='data/LeoIb_1200B_final3.csv', outputname='data/hires_data_final/leoi/shetrone03_matched.csv')

	# Match hi-res files for Ursa Major II
	#match_hires(hiresfilelist=['data/hires_data_final/umaii/frebel10_final.csv'], sourcelist=['Frebel+10'], obsfile='data/UMaIIb_1200B_final3.csv', outputname='data/hires_data_final/umaii/frebel10_matched.csv')

	# Match hi-res files for Ursa Minor
	#files = ['cohen10','sadakane04','shetrone01','ural15']
	#names = ['Cohen+10','Sadakane+04','Shetrone+01','Ural+15']
	#for i in range(len(files)):
	#	match_hires(hiresfilelist=['data/hires_data_final/umi/'+files[i]+'_final.csv'], sourcelist=[names[i]], obsfile='data/bumia_1200B_final3.csv', outputname='data/hires_data_final/umi/'+files[i]+'_matched.csv')

if __name__ == "__main__":
	main()