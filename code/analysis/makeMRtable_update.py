# Script to make machine readable tables for ApJ
#
# Updated: 5 Oct 2018
######################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def makeTables(tablename, outfilepath='Mnlinelists/finallinelists/'):
	"""Code to make tables.
		
		Inputs:
		- tablename: name of table
			options: 	- tab1: general sample properties
						- photometryresults: fluxes, SFRs, and surface densities
						- otherdata: secondary parameters
	"""

	if tablename == 'tab3':
		##Import file
		database = '/Users/miadelosreyes/Documents/Research/code/Mnlinelists/finallinelists/Mnlinelist_total'
		data = pd.read_csv(database, delimiter='\s+', dtype='str')

		index	= data['N']
		NEDid	= data['NEDid']

		##Remove BCDs and compact starbursts
		bcdlist = np.array(['NGC 1569','NGC 5253','IC 0010','UGC 05720','UGCA 116','UGCA 166','IC 0745','NGC 0625','NGC 1705','NGC 2915','NGC 4670','UGCA 281','UGCA 439','UM 461','NGC 4861','NGC 4026','Aquarius dIrr'])
		bcdcheck = np.ones(len(index), dtype='bool')
		for i in range(len(index)):
			if np.any(bcdlist == NEDid[i]):
				bcdcheck[i] = False

		N = len(index[bcdcheck])

		# Re-import index and NED ID for non-BCDs
		index = np.asarray(np.arange(1, N+1, 1), dtype='str')
		NEDid = np.asarray(NEDid[bcdcheck])

	######################################
	#Import data
	######################################
	if tablename == 'tab1':

		##Import columns
		catalog	= np.asarray(data['finalcatalog'])[bcdcheck]

		RA	= np.asarray(data['RA'])[bcdcheck]
		Dec	= np.asarray(data['Dec'])[bcdcheck]

		dist	= np.asarray(data['Distance'])[bcdcheck]
		Bmag	= np.asarray(data['Bmag'])[bcdcheck]
		Bmagerr	= np.asarray(data['Bmag_err'])[bcdcheck]
		BVcolor = np.asarray(data['BVcolor'])[bcdcheck]
		BVerr 	= np.asarray(data['errBVcolor'])[bcdcheck]

		morph	= np.asarray(data['morphology'])[bcdcheck]
		posang	= np.asarray(data['SFposangle'])[bcdcheck]
		Ebv		= np.asarray(pd.to_numeric(data['E(B-V)']))[bcdcheck]
		Ebverr	= np.asarray(pd.to_numeric(data['E(B-V)_err']))[bcdcheck]

		D25a 	= np.asarray(pd.to_numeric(data['D25a']))[bcdcheck]
		D25b 	= np.asarray(pd.to_numeric(data['D25b']))[bcdcheck]

		SFa	= np.asarray(pd.to_numeric(data['SFa']))[bcdcheck]

		#Make catalog
		catalog[np.where(catalog=='spiral')] = 's'
		catalog[np.where(catalog=='dwarf')]  = 'd'

		print('Spirals: ', len(np.where(catalog=='s')[0]))
		print('Dwarfs: ', len(np.where(catalog=='d')[0]))

		mask = np.where((D25a > 0) & (SFa > 0))
		check = D25a[mask]/SFa[mask]
		print('Diam check: ', np.median(check))

		##Format coordinates
		RAh = np.empty(N, dtype=object)
		RAm = np.empty(N, dtype=object)
		RAs = np.empty(N, dtype=object)
		Decsign = np.empty(N, dtype=object)
		Decd = np.empty(N, dtype=object)
		Decm = np.empty(N, dtype=object)
		Decs = np.empty(N, dtype=object)
		for i in range(N):
			RAh[i] = RA[i][0:2]
			RAm[i] = RA[i][3:5]
			RAs[i] = RA[i][6:11]

			Decsign[i] = Dec[i][0:1]
			Decd[i] = Dec[i][1:3]
			Decm[i] = Dec[i][4:6]
			Decs[i] = Dec[i][7:11]

		##Format data with errorbars
		Ebvfinal	= np.empty(N, dtype=object)
		Ebverrfinal	= np.empty(N, dtype=object)
		for i in range(N):
			if (Bmag[i] == '-999') or (Bmagerr[i] == '-999'):
				Bmag[i] = ''
				Bmagerr[i] = ''

			if (BVcolor[i] == '-999') or (BVerr[i] == '-999'):
				BVcolor[i] = ''
				BVerr[i] = ''

			if (Ebv[i] > -990) and (Ebverr[i] > -990):
				Ebvfinal[i] = "{0:.2f}".format(Ebv[i])
				Ebverrfinal[i] = "{0:.2f}".format(Ebverr[i])
			else:
				Ebvfinal[i] = ''
				Ebverrfinal[i] = ''

		##Check that all columns have non-empty data
		dist[(dist == '-999')]	= ''
		morph[(morph == '-999')]	= ''

		#diam = np.asarray(np.round(D25a*2.), dtype='str')
		diam = np.asarray(['%i' % np.round(n*2.) for n in D25a])
		diam[(D25a < -990)] = ''

		#diamHa = np.asarray(np.round(SFa*2.), dtype='str')
		diamHa = np.asarray(['%.12g' % np.round(n*2.) for n in SFa])
		diamHa[np.where(SFa < -990)] = ''

		axisratio = np.asarray(np.round(D25b/D25a, decimals=2), dtype='str')
		axisratio[(np.logical_or(D25a < -990, D25b < -990))] = ''

		##List containing all columns to be put into table
		listcol = [index, NEDid, RAh, RAm, RAs, Decsign, Decd, Decm, Decs,
					dist, Bmag, Bmagerr, BVcolor, BVerr, catalog,
					posang, Ebvfinal, diam, diamHa, axisratio]

		Ncols	= len(listcol)
		#for i in range(Ncols):
		#	print(listcol[i].dtype)

	elif tablename == 'tab2':

		## Total UV fluxes
		UVflux	= np.asarray(pd.to_numeric(data['fluxUV']))[bcdcheck]
		UVerr	= np.asarray(pd.to_numeric(data['errUV']))[bcdcheck]
		UVflag	= np.asarray(data['limUV'])[bcdcheck]
		UVcat	= np.asarray(data['catalogUV'])[bcdcheck]

		## UV fluxes from measurements
		photdb 		= '/Users/miadelosreyes/Documents/Research/SF/ForPaper/Data/UVphotometryinfo.csv'
		photIDs 	= np.genfromtxt(photdb, delimiter=',', skip_header=1, usecols=0, dtype='str')
		photexptime = np.genfromtxt(photdb, delimiter=',', skip_header=1, usecols=1)
		phottile 	= np.genfromtxt(photdb, delimiter=',', skip_header=1, usecols=2, dtype='str')
		photNUVflag	= np.genfromtxt(photdb, delimiter=',', skip_header=1, usecols=3, dtype='int')

		## Format data with errorbars
		UVfluxstr 	= np.empty(N, dtype=object)
		UVerrstr 	= np.empty(N, dtype=object)
		UVexptime 	= np.empty(N, dtype=object)
		UVtilename 	= np.empty(N, dtype=object)
		NUVflag 		= np.empty(N, dtype=object)

		for i in range(N):
			if UVflux[i] > 0:

				# Convert Jy to AB mag
				UVfluxnew = -5./2. * np.log10(UVflux[i]/3631.)
				UVerrnew  = np.abs(-1.0857362/(UVflux[i]) * UVerr[i])

				UVfluxstr[i] = "{0:.1f}".format(UVfluxnew)

				if UVflag[i] == '1':
					UVerrstr[i] = ''
				else:
					UVerrstr[i] = "{0:.1f}".format(UVerrnew)

			else:
				UVfluxstr[i] = ''
				UVerrstr[i] = ''

			if UVcat[i] == 'my measurement':

				# Index from measured flux database
				idx = np.where(photIDs==NEDid[i])[0]

				#print(i, idx)

				UVexptime[i]  = str(photexptime[idx])[1:-1]
				UVtilename[i] = str(phottile[idx])[2:-2]
				NUVflag[i]	  = str(photNUVflag[idx])[1:-1]

			else:
				UVexptime[i]  = ''
				UVtilename[i] = ''
				NUVflag[i]	  = '0'

			# Format catalogs by numbers
			UVcatalogs 	= {'my measurement' : '1', 'Gil de Paz+07' : '2', 'LVL (Spitzer apertures)' : '3', 
						'SINGS' : '4', 'LVL (GALEX apertures)' : '5', 'Bai+15' : '6', 
						'Virgo Cluster Survey' : '7', 'Herschel Ref Survey' : '8', '-999': ''}
			UVcat[i] = UVcatalogs[UVcat[i]]

		print('My UV measurement: ', len(np.where(UVcat == '1')[0]))
		print('NUV: ', len(np.where((NUVflag == '1') & (UVcat == '1'))[0]))

		## List containing all columns to be put into table
		listcol = [index, NEDid, UVflag, UVfluxstr, UVerrstr, UVcat, UVexptime, UVtilename, NUVflag]
		Ncols	= len(listcol)

	elif tablename == 'tab3':

		## Total IR fluxes
		IRflux	= np.asarray(pd.to_numeric(data['fluxIR']))[bcdcheck]
		IRerr	= np.asarray(pd.to_numeric(data['errIR']))[bcdcheck]
		IRflag	= np.asarray(data['limIR'])[bcdcheck]
		IRcat	= np.asarray(data['catalogIR'])[bcdcheck]

		## IR fluxes from measurements
		photdb_spitz 		= '/Users/miadelosreyes/Documents/Research/SF/ForPaper/Data/IRphotometryinfo_Spitzer.csv'
		photIDs_spitz 	= np.genfromtxt(photdb_spitz, delimiter=',', usecols=0, dtype='str')
		photexptime_spitz = np.genfromtxt(photdb_spitz, delimiter=',', usecols=1, dtype='str')
		photsurvey_spitz 	= np.genfromtxt(photdb_spitz, delimiter=',', usecols=2, dtype='str')

		photdb_wise 		= '/Users/miadelosreyes/Documents/Research/SF/ForPaper/Data/IRphotometryinfo_WISE.csv'
		photIDs_wise 	= np.genfromtxt(photdb_wise, delimiter=',', usecols=0, dtype='str')
		photnframes_wise = np.genfromtxt(photdb_wise, delimiter=',', usecols=1, dtype='str')
		photcoadd_wise 	= np.genfromtxt(photdb_wise, delimiter=',', usecols=2, dtype='str')

		## Format data with errorbars
		IRfluxstr 	= np.empty(N, dtype=object)
		IRerrstr 	= np.empty(N, dtype=object)
		IRexptime 	= np.empty(N, dtype=object)
		IRsurvey 	= np.empty(N, dtype=object)
		IRnframes	= np.empty(N, dtype=object)
		IRcoadd 	= np.empty(N, dtype=object)

		for i in range(N):
			if (IRflux[i] > 0) and (np.isnan(IRflux[i]) == False):
				if IRflag[i] == '1':
					IRfluxstr[i] = "{0:.2e}".format(IRflux[i])
					IRerrstr[i] = ''
				else:
					IRfluxstr[i] = "{0:.2e}".format(IRflux[i])
					IRerrstr[i] = "{0:.1e}".format(IRerr[i])
			else:
				IRfluxstr[i] = ''
				IRerrstr[i] = ''

			if IRcat[i] == 'my measurement':

				# Index from measured flux database
				idx = np.where(np.array(photIDs_spitz)==NEDid[i])[0]

				#print(photexptime[idx])

				IRexptime[i]  = str(photexptime_spitz[idx][0])
				IRsurvey[i] = str(photsurvey_spitz[idx][0])

				IRnframes[i]  = ''
				IRcoadd[i] = ''

			elif IRcat[i] == 'my measurement (WISE)':

				# Index from measured flux database
				idx = np.where(np.array(photIDs_wise)==NEDid[i])[0]

				print(i, idx)

				IRexptime[i]  = ''
				IRsurvey[i] = ''

				IRnframes[i]  = str(photnframes_wise[idx][0])
				IRcoadd[i] = str(photcoadd_wise[idx][0])

			else:
				IRexptime[i]  = ''
				IRsurvey[i] = ''
				IRnframes[i]  = ''
				IRcoadd[i] = ''

			# Format catalogs by numbers
			IRcatalogs 	= {'my measurement' : '1', 'my measurement (WISE)' : '2', 'LVL' : '3',
						'SINGS' : '4', 'Gil de Paz+07' : '5', 'MIPS LG' : '6', 
						'IRAS BGS' : '7'}
			IRcat[i] = IRcatalogs[IRcat[i]]

		#print('My measurement (Spitzer): ', len(np.where(IRcat == '1')[0]))
		#print('My measurement (WISE): ', len(np.where(IRcat == '2')[0]))

		## List containing all columns to be put into table
		listcol = [index, NEDid, IRflag, IRfluxstr, IRerrstr, IRcat, IRexptime, IRsurvey, IRnframes, IRcoadd]
		Ncols	= len(listcol)

	elif tablename == 'tab4':

		##Import columns
		logSFR_UV_uncorr	= np.asarray(pd.to_numeric(data['Log(SFR_UVuncorr)']))[bcdcheck]
		logSFRerr_UV_uncorr	= np.asarray(pd.to_numeric(data['errLog(SFR_UVuncorr)']))[bcdcheck]

		logSFR_UV_corr		= np.asarray(pd.to_numeric(data['Log(SFR_UVcorr)']))[bcdcheck]
		logSFRerr_UV_corr	= np.asarray(pd.to_numeric(data['errLog(SFR_UVcorr)']))[bcdcheck]

		print(logSFR_UV_corr[0])

		sigHI	= np.asarray(pd.to_numeric(data['sigma(HI)']))[bcdcheck]
		sigH2	= np.asarray(pd.to_numeric(data['sigma(H2)']))[bcdcheck]
		sigH2flag	= np.asarray(data['sigma(H2)_upperlimit'])[bcdcheck]

		HIref 	= np.asarray(data['Hiref'])[bcdcheck]
		H2ref 	= np.asarray(data['H2ref'])[bcdcheck]

		print(H2ref)

		#Calculate things for surface density calculations
		SFa	= np.asarray(pd.to_numeric(data['SFa']))[bcdcheck]
		dist	= np.asarray(pd.to_numeric(data['Distance']))[bcdcheck]*1000.
		rad		= SFa*np.pi/648000.	#convert radius from arcsec to radians
		Robrad		= dist*np.tan(rad)	#convert angular diam to actual radius (in kpc)

		##Format any numerical data without errorbars
		SFa	= np.asarray(pd.to_numeric(data['SFa']))[bcdcheck]
		diam = np.asarray(np.round(SFa*2.), dtype='str')
		diam[np.where(SFa < -990)] = r''

		sigHItotal = np.asarray(["%.2f" % np.log10(number) for number in sigHI])
		sigHItotal[np.where(sigHI < -990)] = ''

		##Format data with errorbars
		SFRflaguncorr 	= np.empty(N, dtype=object)
		SFRuncorr		= np.empty(N, dtype=object)
		SFRerruncorr 	= np.empty(N, dtype=object)
		SFRflagcorr 	= np.empty(N, dtype=object)
		SFRcorr 		= np.empty(N, dtype=object)
		SFRerrcorr 		= np.empty(N, dtype=object)
		sigSFR 			= np.empty(N, dtype=object)
		sigH2total 		= np.empty(N, dtype=object)

		for i in range(N):

			# total uncorrected SFR
			if (logSFR_UV_uncorr[i] > -990):
				if logSFRerr_UV_uncorr[i] > -990:
					SFRflaguncorr[i] = '0'
					SFRuncorr[i] = "{0:.2f}".format(logSFR_UV_uncorr[i])
					SFRerruncorr[i] = "{0:.2f}".format(logSFRerr_UV_uncorr[i])
				else:
					SFRflaguncorr[i] = '1'
					SFRerruncorr[i] = ''
					SFRuncorr[i] = "{0:.2f}".format(logSFR_UV_uncorr[i])
			else:
				SFRflaguncorr[i] = ''
				SFRuncorr[i] = ''
				SFRerruncorr[i] = ''


			# total corrected SFR
			if (logSFR_UV_corr[i] > -990):
				if logSFRerr_UV_corr[i] > -990:
					SFRflagcorr[i] = '0'
					SFRcorr[i] = "{0:.2f}".format(logSFR_UV_corr[i])
					SFRerrcorr[i] = "{0:.2f}".format(logSFRerr_UV_corr[i])
				else:
					SFRflagcorr[i] = '1'
					SFRerrcorr[i] = ''
					SFRcorr[i] = "{0:.2f}".format(logSFR_UV_corr[i])
			else:
				SFRflagcorr[i] = ''
				SFRcorr[i] = ''
				SFRerrcorr[i] = ''

			# SFR surface density
			if (logSFR_UV_corr[i] > -990) and (SFa[i] > -990) and (dist[i] > -990):
				sigSFR[i]	= "{0:.2f}".format( logSFR_UV_corr[i] - np.log10(np.pi*np.power(Robrad[i],2.)) )
			else:
				sigSFR[i] = ''

			# H2 surface density and mass
			if (sigH2[i] > 0): 
				sigH2total[i] = "{0:.2f}".format(np.log10(sigH2[i]))
			else:
				sigH2total[i] = ''

		# Gas references
		#HIref = np.zeros(N, dtype='str')
		#H2ref = np.zeros(N, dtype='str')

		##List containing all columns to be put into table
		#listcol = [index, NEDid, SFRtotaluncorr, SFRtotalcorr, sigSFR, MHItotal, sigHItotal, HIref, MH2total, sigH2total, H2ref]
		listcol = [index, NEDid, SFRflaguncorr, SFRuncorr, SFRerruncorr, SFRflagcorr, SFRcorr, SFRerrcorr, sigSFR, sigHItotal, HIref, sigH2flag, sigH2total, H2ref]
		Ncols	= len(listcol)

		for i in range(Ncols):
			print(listcol[i].dtype)

	elif tablename == 'tab5':

		Mstar	= np.asarray(pd.to_numeric(data['Log(Mstar)']))[bcdcheck]
		Mstarcat	= np.asarray(data['catalogMass'])[bcdcheck]
		Z	= np.asarray(pd.to_numeric(data['12+Log(O/H)']))[bcdcheck]
		Zcat	= np.asarray(data['catalogZ'])[bcdcheck]
		conc	= np.asarray(pd.to_numeric(data['conc']))[bcdcheck]
		vcirc	= np.asarray(pd.to_numeric(data['v_circ']))[bcdcheck]

		#gasratio	= np.genfromtxt(database, delimiter=',', skip_header=1, usecols=1)
		#diam	= np.genfromtxt(database, delimiter=',', skip_header=1, usecols=1)
		#sigMstar	= np.genfromtxt(database, delimiter=',', skip_header=1, usecols=1)
		#opacity	= np.genfromtxt(database, delimiter=',', skip_header=1, usecols=1)
		#SSFR	= np.genfromtxt(database, delimiter=',', skip_header=1, usecols=1)
		#gas-stars	= np.genfromtxt(database, delimiter=',', skip_header=1, usecols=1)

		#Convert vcirc to tdyn
		SFa	= np.asarray(pd.to_numeric(data['SFa']))[bcdcheck]
		dist	= np.asarray(pd.to_numeric(data['Distance']))[bcdcheck]*1000.
		rad		= SFa*np.pi/648000.	#convert radius from arcsec to radians
		Robrad		= dist*np.tan(rad)	#convert angular diam to actual radius (in kpc)

		vcirc 	= vcirc/9.785 #convert to kpc/(10^8 yr)
		circumf = 2*np.pi*Robrad	#in units kpc
		tdyn	= 2*np.pi*Robrad/vcirc	#in units 10^8 yr
		tdyn[np.where(vcirc <= 0)] = -999
		vcirc[np.where(vcirc <= 0)] = -999
		tdyn[np.where(Robrad < 0)] = -999

		# Create final columns
		Mstar_final	= np.empty(N, dtype=object)
		Z_final	= np.empty(N, dtype=object)
		conc_final	= np.empty(N, dtype=object)
		tdyn_final	= np.empty(N, dtype=object)
		vcirc_final	= np.empty(N, dtype=object)
		tdyn_ref	= np.empty(N, dtype=object)

		# Format columns without errorbars
		for i in range(N):

			# Stellar mass
			if (Mstar[i] > -990):
				Mstar_final[i] = "{0:.2f}".format(Mstar[i])
			else:
				Mstar_final[i]	= ''

			# Metallicity
			if (Z[i] > -990):
				Z_final[i] = "{0:.2f}".format(Z[i])
			else:
				Z_final[i] = ''

			# Concentration
			if (conc[i] > -990):
				conc_final[i] = "{0:.2f}".format(conc[i])
			else:
				conc_final[i] = ''

			# Dyn time
			if (tdyn[i] > -990):
				tdyn_final[i] = "{0:.2f}".format(tdyn[i])
			else:
				tdyn_final[i] = ''

			# Dyn time
			if (vcirc[i] > -990):
				vcirc_final[i] = "{0:.2f}".format(vcirc[i])
			else:
				vcirc_final[i] = ''

			# Dyn time and rotational velocity references
			tdyn_ref[i] = ''

			##Format catalogs by numbers
			Mstarcatalogs 	= {'LVL' : '1', 'SINGS' : '2', 'S4G' : '3', 
						'Berg+12' : '4', 'Lee+06' : '5', '-999': ''}
			Mstarcat[i] = Mstarcatalogs[str(Mstarcat[i])]

			if Mstarcat[i] == '4' or Mstarcat[i] == '5':
				print('here')

			Zcatalogs 	= {'M10' : '1', 'Berg12' : '2', 'Lee06' : '3', 
						'MK06' : '4', 'Cook14' : '5', '-999': ''}
			Zcat[i] = Zcatalogs[Zcat[i]]

		##List containing all columns to be put into table
		#listcol = [index, NEDid, Mstar_final, Mstarcat, Z_final, Zcat, conc_final, tdyn_final, vcirc_final, tdyn_ref]
		listcol = [index, NEDid, Mstar_final, Mstarcat, Z_final, Zcat, conc_final, tdyn_final, vcirc_final]
		Ncols	= len(listcol)

	'''
	elif tablename == 'UVphotometryinfo':
		##Import master database file
		database = '/home/macd4/Documents/Research/Data/photometryUV/UVphotometryinfo.csv'
		##Import columns
		listcol	= np.genfromtxt(database, delimiter=',', skip_header=1, dtype='str', usecols=(0,1,2,3))
		N = len(listcol)
		Ncols = len(listcol[0,:])

	elif tablename == 'IRphotometryinfo':
		##Import master database file
		database = '/home/macd4/Documents/Research/Data/photometryIR/IRphotometryinfo.csv'
		##Import columns
		listcol	= np.genfromtxt(database, delimiter=',', skip_header=1, dtype='str', usecols=(0,1,2))
		N = len(listcol)
		Ncols = len(listcol[0,:])
	'''

	######################################
	#Make table
	######################################

	##Open text file
	workfile	= outfilepath+tablename+'.txt'
	f = open(workfile, 'w')

	for i in range(N):
		for j in range(Ncols):
			#if tablename == 'genpropsample' or tablename == 'photometryresults' or tablename == 'alldata':
			f.write(listcol[j][i])
			if j < Ncols - 1:
				f.write(' & ')
			else:
				f.write(' \n')
			#elif tablename == 'UVphotometryinfo' or tablename == 'IRphotometryinfo':
			#	f.write(listcol[i,j])
			#	if j < Ncols - 1:
			#		f.write(' & ')
			#	else:
			#		f.write(' \\\\\n')

	return

def main():
	#makeTables('tab1')
	#makeTables('tab2')
	#makeTables('tab3')
	#makeTables('tab4')
	makeTables('tab5')
	return

if __name__ == "__main__":
	main()