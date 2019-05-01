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

def north_etal_12(coordfile, datafile1, datafile2, outfile):
	"""Get data from North+12 paper

	Inputs:
	coordfile 	-- table of coordinates
	datafile1 	-- table with abundances for lines 5407, 5420
	datafile1 	-- table with abundances for lines 5432, 5516
	outfile 	-- name of output file

	Keywords:
	"""

	# Open files
	coorddata = pd.read_fwf(coordfile, colspecs=[(0,6),(7,17),(18,30)])
	N = len(coorddata['Name'])

	data1 	= pd.read_csv(datafile1, delimiter='\t', na_values = ' ')

	data2 	= pd.read_csv(datafile2, delimiter='\t', na_values = ' ')

	# Prep the output data file
	outputname = 'data/hires_data_final/'+outfile
	with open(outputname, 'w+') as f:
		f.write('Name\tRA\tDec\tTemp\tlog(g)\t[Fe/H]\terror([Fe/H])\t[alpha/Fe]\t[Mn/H]\terror([Mn/H])\tchisq(reduced)\n')

	# Loop over all stars listed in coordinate file
	for i in range(N):

		starname = coorddata['Name'][i][:2] + coorddata['Name'][i][3:]
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

				mnherr5407 = pd.to_numeric(data1['[Mn/Fe]error5407'])[idx1].values[0]
				mnherr5420 = pd.to_numeric(data1['[Mn/Fe]error5420'])[idx1].values[0]
				mnherr5516 = pd.to_numeric(data2['[Mn/Fe]error5516'])[idx2].values[0]

				feh = mnh5407 - mnfe5407

				# Average all of the [Mn/H] to get a final abundance
				mnh = np.average([mnh5407, mnh5420, mnh5516], weights=[1./(mnherr5407**2.),1./(mnherr5420**2.),1./(mnherr5516**2.)])
				mnherr = np.sqrt(mnherr5407**2. + mnherr5420**2. + mnherr5516**2.)

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
					f.write(starname+'\t'+str(RA)+'\t'+str(Dec)+'\t'+str(temp)+'\t'+str(logg)+'\t'+str(feh)+'\t'+str(feherr)+'\t'+str(alphafe)+'\t'+str(mnh)+'\t'+str(mnherr)+'\t'+str(chisq)+'\n')


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

def main():
	# Sculptor
	#north_etal_12('data/Sculptor_hires_data/scl_sample.coord','data/Sculptor_hires_data/Sculptor_north_tab1.tsv','data/Sculptor_hires_data/Sculptor_north_tab2.tsv','north12_final.csv')

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
	sobeck_etal_06('data/hires_data/M5_M15_sobeck.tsv','M15','M15_sobeck06_final.csv')

	# M53
	#lamb_etal_14('data/hires_data/M53_lamb.txt', 'M53_lamb_final.csv')
	#apogee('data/hires_data/M53_apogee.csv','M53_apogee_final.csv')


if __name__ == "__main__":
	main()