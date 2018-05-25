import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.interpolate
import moog_functions 
import ba_meas_functions
import astropy.io.fits as pyfits
import astropy.coordinates
from astropy import units as u
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
import pandas as pd
import re

rc('font', family='serif')
rc('font', weight='medium')
rc('mathtext', default='sf')
rc("lines", markeredgewidth=0.7)
rc('axes', labelsize=20) #24
rc("axes", linewidth=2)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('xtick.major', size=14)
rc('ytick.major', size=14)
rc('xtick.minor', size=7)
rc('ytick.minor', size=7)
rc('legend', fontsize=12, frameon=False) #16
rc('xtick.major', pad=8)
rc('ytick.major', pad=8)
rc('xtick',direction='in')
rc('ytick',direction='in')
rc('font', family='serif')
#rc('font', weight='medium')
#rc('mathtext', default='sf')
#rc("lines", markeredgewidth=0.7)
#rc('text',usetex=True)

h_lines = [4341, 4861, 6563]

xstart=np.array([4120.6, 4544.0, 4924, 5843, 6131.7, 6486.9])
xstop = np.array([4140.6, 4564.0, 4944, 5863, 6151.7, 6506.9])
xlines = np.array([4130.6, 4554.0, 4934.1, 5853.7, 6141.7, 6496.9])
wvl_radius=10

plot_path = '/raid/gduggan/analysis_code/plots/ba_fit/'
       
# systematic errors calculated by Evan on 12/28/17
fehsyserr = 0.10103081
alphasyserr = 0.084143983
mgfesyserr = 0.076933658
sifesyserr = 0.099193360
cafesyserr = 0.11088295
tifesyserr = 0.10586739
bafesyserr = 0.100

bah_solar = 2.13

accuracy_cutoff = 0.3 #dex cut for alpha/Fe error, Ba/Fe error, and Mg/Fe error

def verify_cluster_member(ra,dec,cluster_lit_tbdata,cluster_name):
    # find out if this star is a member in the correct galaxy. Might be mutliple matches because objname
    # is truncated to the first 10 characters in member file. Verify that ra and dec match.
    indices = [i for i, s in enumerate(cluster_lit_tbdata['CLUSTER']) if cluster_name in s]
    match = 0
    member_index=np.nan
    for index in indices:          
        if ((np.around(cluster_lit_tbdata['RA'][index],3)==np.around(ra,3)) & 
            (np.around(cluster_lit_tbdata['DEC'][index],3)==np.around(dec,3))):
            print "match:",index, cluster_lit_tbdata['REF'][index],cluster_lit_tbdata['NAME'][index], cluster_lit_tbdata['RA'][index],ra,cluster_lit_tbdata['DEC'][index],dec
            match = match +1
            member_index = np.copy(index)
            #print dsph_member.dtype
#                print teff[slitindex],feh[slitindex],logg[slitindex], dsph_member['Teff'][index],dsph_member['FeH'][index],dsph_member['logg'][index]
    if match ==2:
        match = 0
        member_index=np.nan
        for index in indices:          
            if ((np.around(cluster_lit_tbdata['RA'][index],3)==np.around(ra,3)) & 
                (np.around(cluster_lit_tbdata['DEC'][index],3)==np.around(dec,3))):
                if '97' in cluster_lit_tbdata["REF"][index]:
                    pass
                else:
                    print "match:",index, cluster_lit_tbdata['NAME'][index], cluster_lit_tbdata['RA'][index],ra,cluster_lit_tbdata['DEC'][index],dec
                    match = match +1
                    member_index = np.copy(index)        
    return match,member_index

def verify_dsph_member(objnames,ra,dec,feh,dsph_member,debug=False):
    #if debug == True: print objnames,ra,dec,feh
    # find out if this star is a member in the correct galaxy. Might be mutliple matches because objname
    # is truncated to the first 10 characters in member file. Verify that ra and dec match.
    #print objnames#, [str(int(s)) for i,s in enumerate(dsph_member['objid'])]
    objnames = objnames.strip(' ') #be sure to remove spaces from both names to ensure a match
    try: indices = [i for i, s in enumerate(dsph_member['objname']) if objnames[:10] in s.strip(' ')]
    except(ValueError): indices = range(len(dsph_member))
    #if debug == True: print indices, dsph_member['objname'][indices]
    match = 0
    member_index=np.nan
    for index in indices:
        try: 
            if dsph_member['Decd'][index]<0.0:
                sign = ''
            else:
                sign = '+'
            string = '%g %g %g %s%g %g %g'%(dsph_member['RAh'][index],dsph_member['RAm'][index],
                                            dsph_member['RAs'][index],sign,dsph_member['Decd'][index],
                                            dsph_member['Decm'][index],dsph_member['Decs'][index])
            c = astropy.coordinates.SkyCoord(string, unit=(u.hourangle, u.deg))
            #if debug == True: print c.ra.degree,c.dec.degree,dsph_member['FeH'][index]
            if ((np.around(c.ra.degree,2)==np.around(ra,2)) & 
                (np.around(c.dec.degree,2)==np.around(dec,2))):
                if (np.around(feh,1)!=np.around(dsph_member['FeH'][index],1)):
                    print objnames, "has a different [Fe/H] in the catalog:", feh, dsph_member['FeH'][index]
    #                print "match:",index, dsph_member['objname'][index],objnames[slitindex], c.ra.degree,ra[slitindex],c.dec.degree,dec[slitindex]
                match = match +1
                member_index = np.copy(index)
                #print dsph_member.dtype
    #                print teff[slitindex],feh[slitindex],logg[slitindex], dsph_member['Teff'][index],dsph_member['FeH'][index],dsph_member['logg'][index]
        except(ValueError):
            if ((np.around(dsph_member['ra'][index],3)==np.around(ra,3)) & 
                (np.around(dsph_member['dec'][index],3)==np.around(dec,3))): 
                #(np.around(feh,2)==np.around(dsph_member['FeH'][index],2))):
    #                print "match:",index, dsph_member['objname'][index],objnames[slitindex], c.ra.degree,ra[slitindex],c.dec.degree,dec[slitindex]
                match = match +1
                member_index = np.copy(index)
                #print dsph_member.dtype
    #                print teff[slitindex],feh[slitindex],logg[slitindex], dsph_member['Teff'][index],dsph_member['FeH'][index],dsph_member['logg'][index]
    #print match, member_index
    #if match != 0:
    #    if np.isnan(dsph_member['Alphaerr'][member_index]) == True:
    #        print np.isnan(dsph_member['Alphaerr'][member_index]), dsph_member['Alphaerr'][member_index]
    #        match=0
    #        member_index=np.nan
    #if match > 1:
    #    print 'ERROR: multiple stars matched in catalog for ',objnames
    return match, member_index
        

def cluster_literature():
    # load cluster abundances from literature from file (/raid/gduggan/gc/cluster_avg.fits)
    # created using /raid/gduggan/gc/assemble_abund.pro
    filename = '/raid/gduggan/gc/cluster_avg.fits'
    f = pyfits.open(filename)
    tbdata=f[1].data    
    print tbdata.dtype,len(tbdata)
    mask = tbdata.field('BAFE')<90 
    ba_tbdata= tbdata[mask]
    
    print np.unique(tbdata.field('CLUSTER'))
    ref = []
    for cluster in ['n2419','n4590','n7078']:
        mask2 = np.char.strip(ba_tbdata.field('CLUSTER'))==cluster
        ref_cluster = np.unique(ba_tbdata[mask2].field('REF'))
        print cluster, ref_cluster
        cluster_papers = ['sne97', 'sne00','coh12','lee05']
        #NGC7078 (M15, 18 - \citealt{Sneden97}, 31 [10 overlap] \citealt{Sneden00}, NGC2419 (1 - \citealt{Shetrone01}, 12 - \citealt{Cohen12}),
        # and NGC4590 (M68, 1 - \citealt{Gratton89}, 7 - \citealt{Lee05}, 3 - \citealt{Venn12})
        print "%s has %i duplicant measurements and %i unique stars with barium measurements"%(cluster,
                len(ba_tbdata[mask2])-len(np.unique(ba_tbdata[mask2].field('NAME'))), 
                len(np.unique(ba_tbdata[mask2].field('NAME'))))
        for data_set in ref_cluster:
            mask3 = ba_tbdata[mask2].field('REF')==data_set
            data = ba_tbdata[mask2][mask3]
            plt.errorbar(data.field('FEH'),data.field('BAFE'),fmt = '.',xerr=data.field('EFEH'),yerr=data.field('EBAFE'),label=cluster+' '+data_set)
        ### pick most recent measurement - or plot them all first
    #print ba_tbdata[mask2].field('NAME'), np.unique(ba_tbdata[mask2].field('NAME'))
    print ref
    plt.xlabel("[Fe/H]")
    plt.ylabel("[Ba/Fe]")
    plt.title("Barium Enhancement Measurements of Globular Clusters")
    plt.savefig(plot_path+"ba_GC.png")
    plt.legend()
    plt.show()
    plt.close()
    
    return ba_tbdata
    
    #for i in range(sum(mask)):
    #    if (ba_tbdata.field('CLUSTER')[i] in ['n2419','n4590','n7078'] ) == True:
    #        print "%s %s %.2f %.2f %s"%(ba_tbdata.field('CLUSTER')[i], ba_tbdata.field('NAME')[i], ba_tbdata.field('BAFE')[i], ba_tbdata.field('EBAFE')[i], ba_tbdata.field('REF')[i])               

def slitmask_to_dsph(slitmask_name):
    if slitmask_name[0]== 'b':
        slitmask_name = slitmask_name[1:]
    dsph_name = (re.sub(r'[0-9]+','',slitmask_name)).title()
    if dsph_name == 'Umi':
        dsph_name = 'UMi'
    if dsph_name == 'Cvnii':
        dsph_name = 'CVnII'
    if 'Leota' in dsph_name:
        dsph_name = 'LeoT'
    print slitmask_name, dsph_name
    return dsph_name 

def measure_ba_duggan_moogify(filename_list, carbon_filenames, overwrite=False, 
                              verify_member=False, debug=False, plot=False, 
                              fine_tune_wvl=True, simple_line_correction=False, wvl_max_shift=2):
    #measures barium from the new moogify files where all the needed values are included
    #these moogify files are created by running /raid/gduggan/analysis_code/moogify.pro
    #and are saved in /raid/gduggan/moogify/.
    #This code measures barium for all stars and saves the results in a new moogify file
    #under 'BAFE' and 'BAFEERR'.
    
    if fine_tune_wvl==False:
        wvl_max_shift=0
    else:
        wvl_max_shift=wvl_max_shift

    new_filename_list = []        
    for mask_index in range(len(filename_list)):
        filename = filename_list[mask_index]
        print 'filename',filename
        new_filename = filename_list[mask_index].split('.')[0]+'_ba.fits.gz'
        if (os.path.isfile(new_filename)==True) and (overwrite==False):
            print "overwrite=False and file exists.", new_filename, "remains unchanged."
            return new_filename
        name = (filename.split('/')[-1]).split('_moogify.fits.gz')[0]
        print 'name:', name
        # find carbon moogify file
        mask_c=[ name in file for file in carbon_filenames]
        if len(carbon_filenames[mask_c]) != 1:
            carbon_zrest = False
            print 'carbon moogify file not found for mask', name
        else:
            carbon_zrest = True
            filename_carbon = carbon_filenames[mask_c][0]
            print 'carbon filename:',filename_carbon            
            f2 = pyfits.open(filename_carbon)                 
            tbdata2 = f2[1].data                 
            objname_carbon = np.copy(tbdata2['OBJNAME'])                
            zrest_carbon = np.copy(tbdata2['ZREST'])         
        
        # load entire slitmask
        f = pyfits.open(filename)
        tbdata=f[1].data
        #print 'B_mag',tbdata.field('B')
        #print 'V_mag',tbdata.field('V')
        contdiv = np.copy(tbdata['CONTDIV']) #equals spectra/continuum
        # calculate spectrum's error
        contdivivar = np.copy(tbdata['CONTDIVIVAR'])
        contdivstd = np.zeros(contdivivar.shape)+np.inf #dummy values
        contdiv_mask = contdivivar>0
        contdivstd[contdiv_mask] = np.sqrt(np.reciprocal(contdivivar[contdiv_mask]))
        #print contdivstd.shape
            
        zrest = np.copy(tbdata['ZREST'])
        #zresterr = np.copy(tbdata.field('ZRESTERR')) #equal 0 always
        wvls = np.copy((tbdata['LAMBDA'].T/(1+zrest)).T)
        print tbdata['LAMBDA'][0].shape, zrest.shape
        dlam = np.copy(tbdata['DLAM']*0.95) #resscale fudge factor
        #print dlam
        #good = np.copy(tbdata.field('GOOD')) #unknown definition
        teff = np.copy(tbdata['TEFF'])
        #tefferr = np.copy(tbdata['TEFFERR']) #equal zero always for scl
        logg = np.copy(tbdata['LOGG'])
        print "Number of stars on slitmask with defined logg:",len(np.where(logg!=1.5)[0])
        if debug == True: print "Indices of stars with defined logg:", np.where(logg!=1.5)[0]
        #loggerr = np.copy(tbdata['LOGGERR']) #equal zero always for scl
        feh = np.copy(tbdata['FEH'])
        feherr = np.copy(tbdata['FEHERR']) #equal zero always for dra1, so use recorded error
        alphafe = np.copy(tbdata['ALPHAFE'])
        #alphafeerr = np.copy(tbdata['ALPHAFEERR']) #equal zero always for scl
        objnames = np.copy(tbdata['OBJNAME']) #returns parameter for each slit
        #print objnames
        #print np.array(tbdata.dtype) ###very helpful! Prints out all parameters
        #spectra = tbdata.field('SPEC')
        #continuum = tbdata.field('CONTINUUM') 
        #moogspectra = tbdata.field('MOOGSPEC') #btw 0 and 1
        #moogcont = tbdata.field('MOOGCONT')
        
        if verify_member == True:
            ra = np.copy(tbdata['RA'])
            dec = np.copy(tbdata['DEC'])
            dsph_name = slitmask_to_dsph(name)
                #filename_member = '/raid/m31/udwarf/leot.dat'
                #dsph_member = np.genfromtxt(filename_member,delimiter=',',names=True)
                #print members
                #[('objid', '<f8'), ('ra', '<f8'), ('dec', '<f8'), ('u', '<f8'), ('g', '<f8'), ('r', '<f8'), ('i', '<f8'), ('z', '<f8')])
                # adjusted verify_members to check this format if trying to call the other keywords failed. ra and dec are in a different format, and the names don't match
            #filename_member = '/raid/gduggan/analysis_code/dsph_catalog_copy.dat'
            filename_member = '/raid/m31/dsph/alldsph/dsph_catalog.dat'
            members = np.genfromtxt(filename_member, skip_header=1,
                                    dtype=[('dSph', 'S9'), ('objname', 'S11'), ('RAh', '<i8'), ('RAm', '<i8'), ('RAs', '<f8'), ('Decd', '<i8'), 
                                           ('Decm', '<i8'), ('Decs', '<f8'), ('v_raw', '<f8'), ('Teff', '<f8'), ('logg', '<f8'), ('vt', '<f8'), 
                                           ('FeH', '<f8'), ('Feerr', '<f8'), ('AlphaFe', '<f8'), ('Alphaerr', '<f8'), ('MgFe', '<f8'), ('Mgerr', '<f8'),
                                           ('SiFe', '<f8'), ('Sierr', '<f8'), ('CaFe', '<f8'), ('Caerr', '<f8'), ('TiFe', '<f8'), ('Tierr', '<f8')],
                                    delimiter=(9,11,3,3,6,3,3,6,7,5,5,6,6,5,6,5,6,5,6,5,6,5,6,5))
            members['dSph']=np.core.defchararray.strip(members['dSph'])
            members['objname']=np.core.defchararray.strip(members['objname'])
            #members = np.genfromtxt(filename_member, skip_header=1,delimiter=(9,11,3,3,6,3,3,6,7,5,5,6,6,5,6,5,6,5,6,5,6,5,6,5), names=True, dtype=None) #Use to get dtype list
            # AlphaFe and Alphaerr are not present in all objects, so can't read them in. 
            # Need to consider replacing spaces with negative value so error isn't thrown.
            mask = (dsph_name==members['dSph'])
            dsph_member = np.copy(members[mask])
            if debug == True: print dsph_name, np.unique(members['dSph'])
            print "Number of member stars in catalog for this galaxy:",len(dsph_member['objname'])

        total_num_matched_stars = 0
        for slitindex in range(len(wvls)):
            if debug == True: print slitindex
            if verify_member == True:
                #print objnames[slitindex]
                match,member_index = verify_dsph_member(objnames[slitindex],
                                ra[slitindex],dec[slitindex],feh[slitindex],
                                dsph_member,debug=debug)
                if match == 1:
                    total_num_matched_stars = total_num_matched_stars + 1
                    if debug == True: 
                        print 'index',slitindex, 'matched!'
                if match == 0:
                    if debug==True:
                        print "Slit at index %d (objname %s in mask %s) is not a member of %s"%(slitindex,objnames[slitindex],name, dsph_name)
                    continue
                if match >1:
                    if debug==True:
                        print "Slit at index %d (objname %s in mask %s) has more than one entry in %s"%(slitindex,objnames[slitindex],name,dsph_name)
                    continue            
        #        print member_index
            else: 
                member_index=''
                if feherr[slitindex] == 0.0:
                    continue
            if carbon_zrest == True:
                #### if it is a member, find wavelength solution (ZREST) from carbon moogify file       
                indices = [i for i, s in enumerate(objname_carbon) if objnames[slitindex] in s]
                #print 'red:', objname_red, objnames[slitindex]
                if len(indices) != 1:             
                    print "No unique carbon spectrum found (zero or more than one)"             
                else:             
                    carbon_index = indices[0]
                    zrest_carbon_slit = np.copy(zrest_carbon[carbon_index])                 
                    wvls[slitindex] = np.copy((tbdata['LAMBDA'].T/(1+zrest_carbon_slit)).T)[slitindex]                 
                                                   
            ba_result,ba_error, wvls_adj = ba_meas_functions.barium_fitting_routine(filename,objnames,
                        slitindex,tbdata,wvls,contdiv,
                        teff,logg,feh,alphafe,dlam,debug,plot,contdivstd,
                        name,name_fragment='',wvl_max_shift=wvl_max_shift,
                        simple_line_correction=simple_line_correction,
                        carbon_zrest=carbon_zrest)
            if ba_error<0:
                continue

            print slitindex, "BARIUM: ", ba_result, ba_error, "SN: ",tbdata['SN'][slitindex]
            tbdata['BAFE'][slitindex]=ba_result
            tbdata['BAFEERR'][slitindex]=ba_error
            tbdata['LAMBDA'][slitindex]=wvls_adj
            print 'Max wvl shift applied (ang):', np.max(np.absolute(wvls_adj-wvls[slitindex]))
       
        print "Saving file with barium measurements to:", new_filename
        pyfits.writeto(new_filename,tbdata,overwrite=overwrite)
        new_filename_list.append(new_filename)
    return new_filename_list

def test_impact_on_rep_stars(test_keyword, name_fragment="_compare", wvl_max_shift =2,debug = False, 
                             plot = True, verify_member=True, compare = 'grid'):
    # test_keyword = 'scat', 'iso_r', or 'iso_s', 'outliers'
    dsph_name = 'Scl'
    name_list = ['scl1','scl2','scl6']
    ####### stars that have been selected in each slitmask to span the parameter space 
    star_list = [[33,34,44,63,66],[52,57,83],[]]
    
    filename_list = ['/raid/caltech/moogify/b%s/moogify.fits.gz'%name for name in name_list]

    ## once for all dwarf spheroidals - loads all of the members
    #filename_member = '/raid/m31/dsph/alldsph/dsph_catalog.dat'

    filename_member = '/raid/gduggan/analysis_code/dsph_catalog_copy.dat'
    members = np.genfromtxt(filename_member, skip_header=2,
                            dtype=[('dSph', 'S9'), ('objname', 'S11'), ('RAh', '<i8'), ('RAm', '<i8'), ('RAs', '<f8'), ('Decd', '<i8'), 
                                   ('Decm', '<i8'), ('Decs', '<f8'), ('v_raw', '<f8'), ('Teff', '<f8'), ('logg', '<f8'), ('vt', '<f8'), 
                                   ('FeH', '<f8'), ('Feerr', '<f8'), ('AlphaFe', '<f8'), ('Alphaerr', '<f8'), ('MgFe', '<f8'), ('Mgerr', '<f8'),
                                   ('SiFe', '<f8'), ('Sierr', '<f8'), ('CaFe', '<f8'), ('Caerr', '<f8'), ('TiFe', '<f8'), ('Tierr', '<f8')],
                            delimiter=(9,11,3,3,6,3,3,6,7,5,5,6,6,5,6,5,6,5,6,5,6,5,6,5))
    members['dSph']=np.core.defchararray.strip(members['dSph'])
    members['objname']=np.core.defchararray.strip(members['objname'])
    #members = np.genfromtxt(filename_member, skip_header=1,delimiter=(9,11,3,3,6,3,3,6,7,5,5,6,6,5,6,5,6,5,6,5,6,5,6,5), names=True, dtype=None) #Use to get dtype list
    # AlphaFe and Alphaerr are not present in all objects, so can't read them in. 
    # Need to consider replacing spaces with negative value so error isn't thrown.
    mask = (dsph_name==members['dSph'])
    dsph_member = np.copy(members[mask])
    #print len(dsph_member['objname']), len(np.unique(dsph_member['objname'])) 
    #print members['objname'], members['objname'][0]
#    print len(members['objname']), len(np.unique(members['objname'])) # not unique
    print np.unique(members['dSph'])
#    print dSph_names
    #['Aqr' 'CVnI' 'CVnII' 'ComBer' 'Dra' 'For' 'Herc' 'IC1613' 'LeoA' 'LeoI'
    #'LeoII' 'LeoIV' 'LeoT' 'NGC6822' 'Peg' 'SagDIG' 'Scl' 'Seg2' 'Sex' 'UMaI'
    #'UMaII' 'UMi' 'VV124']

#    for name in dSph_names:
#        mask = (members['dSph']==name)
        #if len(members['objname'][mask]) != len(np.unique(members['objname'][mask])):
             # not unique, truncated if objname > 10 characters, find all that include objname, and then match RA and DEC
             #print name, members['objname'][mask], np.unique(members['objname'][mask])
    ba_results = []
    ba_results_test = []
    for mask_index in range(len(filename_list)):
        filename = filename_list[mask_index]
        name = name_list[mask_index]
        # load entire slitmask
        #print filename
        f = pyfits.open(filename)
        tbdata=f[1].data
        contdiv = np.copy(tbdata.field('CONTDIV')) #equals spectra/continuum
        contdivivar = np.copy(tbdata.field('CONTDIVIVAR'))
        contdivstd = np.sqrt(np.reciprocal(contdivivar))
        zrest = np.copy(tbdata.field('ZREST'))
        #zresterr = np.copy(tbdata.field('ZRESTERR')) #equal 0 always
        wvls = np.copy((tbdata.field('LAMBDA').T/(1+zrest)).T)
        dlam = np.copy(tbdata.field('DLAM')*0.95) #resscale fudge factor
        #print dlam
    #    good = np.copy(tbdata.field('GOOD'))
        ra = np.copy(tbdata.field('RA'))
        dec = np.copy(tbdata.field('DEC'))
        teff = np.copy(tbdata.field('TEFF'))
#        tefferr = np.copy(tbdata.field('TEFFERR')) #equal zero always for scl
        logg = np.copy(tbdata.field('LOGG'))
#        loggerr = np.copy(tbdata.field('LOGGERR')) #equal zero always for scl
        feh = np.copy(tbdata.field('FEH'))
#        feherr = np.copy(tbdata.field('FEHERR')) #equal zero always for dra1, so use recorded error
        alphafe = np.copy(tbdata.field('ALPHAFE'))
#        alphafeerr = np.copy(tbdata.field('ALPHAFEERR')) #equal zero always for scl
        objnames = np.copy(tbdata.field('OBJNAME')) #returns parameter for each slit
        #print np.array(tbdata.dtype) ###very helpful! Prints out all parameters
        spectra = tbdata.field('SPEC')
    #    continuum = tbdata.field('CONTINUUM') 
        moogspectra = tbdata.field('MOOGSPEC') #btw 0 and 1
        moogcont = tbdata.field('MOOGCONT')
        
        for slitindex in star_list[mask_index]:#range(len(wvls)):
    #        ################# skip stars that don't conver all barium lines - can alter later
    #        if (wvls[slitindex][0]>xlines[0]-wvl_radius) or (wvls[slitindex][-1]<xlines[-1]+wvl_radius):
    #            print "Slit at index %d (objname %s) does not cover all 6 lines"%(slitindex,objnames[slitindex])
    #            continue
            #print mask_index, name, slitindex, star_list[mask_index]
            if verify_member == True:
                match,member_index = verify_dsph_member(objnames[slitindex],ra[slitindex],dec[slitindex],feh[slitindex],dsph_member)
                if match == 0:
                    if debug==True:
                        print "Slit at index %d (objname %s in mask %s) is not a member of %s"%(slitindex,objnames[slitindex],name, dsph_name)
                    continue
                if match >1:
                    if debug==True:
                        print "Slit at index %d (objname %s in mask %s) has more than one entry in %s"%(slitindex,objnames[slitindex],name,dsph_name)
                    continue            
        #        print member_index
            else: member_index=''

            ############### FIX THIS #################
            print "ERROR: fix code to not use zrest from carbon moogify file if available"

            if compare == 'grid':
                ba_result,ba_error, wvls_adj = ba_meas_functions.barium_fitting_routine(filename,objnames,
                            slitindex,tbdata,wvls,contdiv,
                            teff,logg,feh,alphafe,dlam,debug,plot,contdivstd,
                            name,name_fragment,wvl_max_shift)
            if compare == '2014':
                ba_result,ba_error, wvls_adj = ba_meas_functions.barium_fitting_routine(filename,objnames,
                            slitindex,tbdata,wvls,contdiv,
                            teff,logg,feh,alphafe,dlam,debug,plot,contdivstd,
                            name,name_fragment,wvl_max_shift,grid=False,test='outliers')                
            ba_result_test,ba_error_test, wvls_adj_test = ba_meas_functions.barium_fitting_routine(filename,objnames,
                            slitindex,tbdata,wvls,contdiv,
                            teff,logg,feh,alphafe,dlam,debug,plot=plot,contdivstd=contdivstd,
                            name=name,name_fragment=name_fragment+'_'+test_keyword,
                            wvl_max_shift=wvl_max_shift,grid=False,test=test_keyword)

            if ba_error<0:
                continue
            #ba_result,ba_error = 0.0,0.1
            #if debug==True:
            print "BARIUM: ", ba_result, ba_error, "S/N: ",tbdata.field('SN')[slitindex]
            print "BARIUM TEST: ", ba_result_test, ba_error_test
            ba_results.append((name,slitindex,objnames[slitindex],ra[slitindex],dec[slitindex],dsph_member['v_raw'][member_index],
                                   teff[slitindex],tbdata2.field('TEFFERR')[red_index],logg[slitindex],tbdata2.field('LOGGERR')[red_index],
                                   dsph_member['vt'][member_index],alphafe[slitindex],dsph_member['Alphaerr'][member_index],
                                   feh[slitindex],dsph_member['Feerr'][member_index],ba_result,ba_error, dsph_member['MgFe'][member_index],
                                   dsph_member['Mgerr'][member_index],dsph_member['SiFe'][member_index],dsph_member['Sierr'][member_index],
                                   dsph_member['CaFe'][member_index],dsph_member['Caerr'][member_index],dsph_member['TiFe'][member_index],
                                   dsph_member['Tierr'][member_index],tbdata2.field('SN')[red_index],tbdata.field('SN')[slitindex]))#feherr[slitindex],dsph_member['FeH'][member_index],
            ba_results_test.append((name,slitindex,objnames[slitindex],ra[slitindex],dec[slitindex],dsph_member['v_raw'][member_index],
                                   teff[slitindex],tbdata2.field('TEFFERR')[red_index],logg[slitindex],tbdata2.field('LOGGERR')[red_index],
                                   dsph_member['vt'][member_index],alphafe[slitindex],dsph_member['Alphaerr'][member_index],
                                   feh[slitindex],dsph_member['Feerr'][member_index],ba_result_test,ba_error_test, dsph_member['MgFe'][member_index],
                                   dsph_member['Mgerr'][member_index],dsph_member['SiFe'][member_index],dsph_member['Sierr'][member_index],
                                   dsph_member['CaFe'][member_index],dsph_member['Caerr'][member_index],dsph_member['TiFe'][member_index],
                                   dsph_member['Tierr'][member_index],tbdata2.field('SN')[red_index],tbdata.field('SN')[slitindex]))#feherr[slitindex],dsph_member['FeH'][member_index],
    ba_results = np.array(ba_results,dtype=[('dSph', '|S7'), ('slitindex', '>i4'),('objname', '|S25'), ('ra','<f8'),
                            ('dec','<f8'),('v_raw', '<f8'), ('Teff', '<f8'), ('Tefferr','<f8'),
                            ('logg', '<f8'), ('loggerr','<f8'),('vt', '<f8'), ('AlphaFe', '<f8'),('Alphaerr','<f8'),('FeH', '<f8'),
                            ('Feerr', '<f8'),('BaFe', '<f8'),('Baerr', '<f8'), ('MgFe', '<f8'), ('Mgerr', '<f8'),
                            ('SiFe', '<f8'), ('Sierr', '<f8'), ('CaFe', '<f8'), ('Caerr', '<f8'), ('TiFe', '<f8'), 
                            ('Tierr', '<f8'),('SN_red','<f8'),('SN_blue','<f8')])
    ba_results_test = np.array(ba_results_test,dtype=[('dSph', '|S7'), ('slitindex', '>i4'),('objname', '|S25'), ('ra','<f8'),
                            ('dec','<f8'),('v_raw', '<f8'), ('Teff', '<f8'), ('Tefferr','<f8'),
                            ('logg', '<f8'), ('loggerr','<f8'),('vt', '<f8'), ('AlphaFe', '<f8'),('Alphaerr','<f8'),('FeH', '<f8'),
                            ('Feerr', '<f8'),('BaFe', '<f8'),('Baerr', '<f8'), ('MgFe', '<f8'), ('Mgerr', '<f8'),
                            ('SiFe', '<f8'), ('Sierr', '<f8'), ('CaFe', '<f8'), ('Caerr', '<f8'), ('TiFe', '<f8'), 
                            ('Tierr', '<f8'),('SN_red','<f8'),('SN_blue','<f8')])
    with open(dsph_name+'_baresults'+name_fragment+'_test.txt','w') as f:
        f.write('dSph, slitindex, objname, ra, dec, v_raw, Teff, Tefferr, logg, loggerr, vt, AlphaFe, Alphaerr, FeH, Feerr, BaFe, Baerr, MgFe, Mgerr, SiFe, Sierr, CaFe, Caerr, TiFe, Tierr, SN_red, SN_blue\n')
        for line in ba_results:
            f.write(','.join([str(item) for item in line])+'\n')
        f.close()    
    with open(dsph_name+'_baresults'+name_fragment+'_'+test_keyword+'.txt','w') as f:
        f.write('dSph, slitindex, objname, ra, dec, v_raw, Teff, Tefferr, logg, loggerr, vt, AlphaFe, Alphaerr, FeH, Feerr, BaFe, Baerr, MgFe, Mgerr, SiFe, Sierr, CaFe, Caerr, TiFe, Tierr, SN_red, SN_blue\n')
        for line in ba_results_test:
            f.write(','.join([str(item) for item in line])+'\n')
        f.close()
    print ba_results.dtype
    if len(ba_results)==0:
        print "No barium results to show"
        return []
    if debug==True:
        print ba_results
    
    fig3 = plt.figure()
    fig3.subplots_adjust(bottom=0.18,left=0.19)
    plt.errorbar(ba_results['FeH'],ba_results['BaFe'],fmt='.',xerr=ba_results['Feerr'],yerr=ba_results['Baerr'],label='grid')
    plt.errorbar(ba_results_test['FeH'],ba_results_test['BaFe'],fmt='.',xerr=ba_results_test['Feerr'],yerr=ba_results_test['Baerr'],label=test_keyword)
    plt.xlabel("[Fe/H]")
    plt.ylabel("[Ba/Fe]")
    plt.title(name+" Barium Enhancement Measurements "+test_keyword)
    plt.legend()
    plt.savefig(plot_path+dsph_name+name_fragment+'_'+test_keyword+"_ba.png")
    plt.show(fig3)
    plt.close()

    ba_diff = ba_results['BaFe']-ba_results_test['BaFe']
    fig4 = plt.figure()
    fig4.subplots_adjust(bottom=0.18,left=0.19)
    plt.errorbar(ba_results['FeH'],ba_diff,fmt='.',xerr=ba_results['Feerr'],yerr=ba_results['Baerr'])
    #plt.errorbar(ba_results_test['FeH'],ba_results_test['BaFe'],fmt='.',xerr=ba_results_test['Feerr'],yerr=ba_results_test['Baerr'])
    plt.xlabel("[Fe/H]")
    plt.ylabel("[Ba/Fe] grid - "+test_keyword)
    plt.title(name+" Barium Enhancement Measurements: grid - "+test_keyword)
    plt.savefig(plot_path+dsph_name+name_fragment+'_'+test_keyword+"_ba_diff.png")
    plt.show(fig4)
    plt.close()
    
    print 'grid - '+test_keyword + ':', ba_diff, np.mean(np.abs(ba_diff)), np.std(np.abs(ba_diff)), np.max(np.abs(ba_diff))

    #np.savetxt('baresults'+name_fragment+'.txt', ba_results, delimiter=',',fmt="%s",header='dSph objname ra dec v_raw Teff logg vt Feerr BaFe Baerr')
    return ba_results    

def read_cluster_observation(cluster_names, filename_list, carbon_filenames,
                             name_fragment="", overwrite=False, simple_line_correction=True, 
                             debug = False, plot = True, wvl_max_shift =2):

    cluster_lit_tbdata = cluster_literature()
    print cluster_lit_tbdata.dtype
    cluster_papers = ['sne97', 'sne00','coh12','lee05']
    #cluster_papers = ['coh12']
    #cluster_papers = ['sne97','coh12']
    mask_clust = np.array([cluster_lit_tbdata.field('REF')[i] in cluster_papers for i in range(len(cluster_lit_tbdata.field('REF')))])
    cluster_lit_tbdata = cluster_lit_tbdata[mask_clust]
    print cluster_lit_tbdata['NAME']
    print cluster_lit_tbdata['CLUSTER']
    
    #print cluster_lit_tbdata['EALPHAFE']
    
    new_filename_list = []
    
    for mask_index in range(len(filename_list)):
        filename = filename_list[mask_index]
        new_filename = filename_list[mask_index].split('.')[0]+'_ba.fits.gz'
        new_filename_list.append(new_filename)
        if (os.path.isfile(new_filename)==True) and (overwrite==False):
            print "overwrite=False and file exists.", new_filename, "remains unchanged."
            continue
 
        name = (filename.split('/')[-1]).split('_moogify.fits.gz')[0]
        print 'name:', name
        mask_c=[ name in file for file in carbon_filenames]
        if len(carbon_filenames[mask_c]) != 1:
            carbon_zrest = False
        else:
            carbon_zrest = True
            filename_carbon = carbon_filenames[mask_c][0]
            print 'carbon filename:',filename_carbon            
            f2 = pyfits.open(filename_carbon)                 
            tbdata2 = f2[1].data                 
            objname_carbon = np.copy(tbdata2['OBJNAME'])                
            zrest_carbon = np.copy(tbdata2['ZREST'])
        
        cluster_name = cluster_names[mask_index]
        # load entire slitmask
        print filename
        f = pyfits.open(filename)
        tbdata=f[1].data
        contdiv = np.copy(tbdata.field('CONTDIV')) #equals spectra/continuum
        # calculate spectrum's error
        contdivivar = np.copy(tbdata['CONTDIVIVAR'])
        contdivstd = np.zeros(contdivivar.shape)+np.inf #dummy values
        contdiv_mask = contdivivar>0
        contdivstd[contdiv_mask] = np.sqrt(np.reciprocal(contdivivar[contdiv_mask]))

        zrest = np.copy(tbdata.field('ZREST'))
        wvls = np.copy((tbdata.field('LAMBDA').T/(1+zrest)).T)
        dlam = np.copy(tbdata.field('DLAM')*0.95) #resscale fudge factor
        ra = np.copy(tbdata.field('RA'))
        dec = np.copy(tbdata.field('DEC'))
        teff = np.copy(tbdata.field('TEFF'))
#        tefferr = np.copy(tbdata.field('TEFFERR'))
        logg = np.copy(tbdata.field('LOGG'))
#        loggerr = np.copy(tbdata.field('LOGGERR'))
        feh = np.copy(tbdata.field('FEH'))
        feherr = np.copy(tbdata.field('FEHERR')) #equal zero always for dra1, so use recorded error
        alphafe = np.copy(tbdata.field('ALPHAFE'))
        alphafeerr = np.copy(tbdata.field('ALPHAFEERR'))
        objnames = np.copy(tbdata.field('OBJNAME')) #returns parameter for each slit
        #print np.array(tbdata.dtype) ###very helpful! Prints out all parameters
        #spectra = tbdata.field('SPEC')
    #    continuum = tbdata.field('CONTINUUM') 
        #moogspectra = tbdata.field('MOOGSPEC') #btw 0 and 1
        #moogcont = tbdata.field('MOOGCONT')                                  
        
        for slitindex in range(len(wvls)):
            match,member_index = verify_cluster_member(ra[slitindex],dec[slitindex],
                                                    cluster_lit_tbdata,cluster_name)
            if match == 0:
                if debug==True:
                    print "Slit at index %d (objname %s in mask %s) is not a member of %s"%(slitindex,objnames[slitindex],name, cluster_name)
                continue
            if match >1:
                if debug==True:
                    print "Slit at index %d (objname %s in mask %s) has more than one entry in %s"%(slitindex,objnames[slitindex],name,cluster_name)
                continue            
    #        print member_index
        
            if carbon_zrest == True:
                #### if it is a member, find wavelength solution (ZREST) from carbon moogify file       
                indices = [i for i, s in enumerate(objname_carbon) if objnames[slitindex] in s]
                #print 'red:', objname_red, objnames[slitindex]
                if len(indices) != 1:             
                    print "No unique carbon spectrum found (zero or more than one)"             
                else:             
                    carbon_index = indices[0]
                    zrest_carbon_slit = np.copy(zrest_carbon[carbon_index])                 
                    wvls[slitindex] = np.copy((tbdata['LAMBDA'].T/(1+zrest_carbon_slit)).T)[slitindex]                 
                                                   
            ba_result,ba_error, wvls_adj = ba_meas_functions.barium_fitting_routine(filename,objnames,
                        slitindex,tbdata,wvls,contdiv,
                        teff,logg,feh,alphafe,dlam,debug,plot,contdivstd,
                        name,name_fragment,wvl_max_shift,simple_line_correction=simple_line_correction,
                        carbon_zrest=carbon_zrest)
            if ba_error<0:
                continue
            #ba_result,ba_error = 0.0,0.1
            #if debug==True:
            print "BARIUM: ", ba_result, ba_error

            tbdata['BAFE'][slitindex] = ba_result
            tbdata['BAFEERR'][slitindex] = ba_error
            tbdata['LAMBDA'][slitindex] = wvls_adj
            tbdata['OBJNAME_LIT'][slitindex] = cluster_lit_tbdata['NAME'][member_index]
            print 'Max wvl shift applied (ang):', np.max(np.absolute(wvls_adj-wvls[slitindex]))
        
            tbdata['TEFF_LIT'][slitindex] = cluster_lit_tbdata['TEFF'][member_index]
            tbdata['TEFFERR_LIT'][slitindex]=0
            tbdata['LOGG_LIT'][slitindex] = cluster_lit_tbdata['LOGG'][member_index]
            tbdata['LOGGERR_LIT'][slitindex]=0
            tbdata['VT_LIT'][slitindex]=cluster_lit_tbdata['VT'][member_index]
            tbdata['VTERR_LIT'][slitindex]=0
            tbdata['FEH_LIT'][slitindex] = cluster_lit_tbdata['FEH'][member_index]
            tbdata['FEHERR_LIT'][slitindex]=cluster_lit_tbdata['EFEH'][member_index]   
            tbdata['ALPHAFE_LIT'][slitindex] = cluster_lit_tbdata['ALPHAFE'][member_index]
            tbdata['ALPHAFEERR_LIT'][slitindex]=cluster_lit_tbdata['EALPHAFE'][member_index]
            tbdata['BAFE_LIT'][slitindex]=cluster_lit_tbdata['BAFE'][member_index]
            tbdata['BAFEERR_LIT'][slitindex]=cluster_lit_tbdata['EBAFE'][member_index]                            
            tbdata['LIT_COMMENT'][slitindex]=cluster_lit_tbdata['REF'][member_index]
            tbdata['OBJNAME_LIT'][slitindex]=cluster_lit_tbdata['NAME'][member_index]


            print "Saving file with barium measurements to:", new_filename
            pyfits.writeto(new_filename,tbdata,overwrite=overwrite)
            mask = (tbdata['TEFF_LIT'] != 0)
            print name, np.sum(mask)
            print tbdata[mask]['LIT_COMMENT']
            print tbdata[mask]['OBJNAME_LIT']

    print new_filename_list
    return new_filename_list
      
def make_HRS_comp_table(filename_list,name_fragment="", overwrite=False):
    ba_results = []
    for mask_index in range(len(filename_list)):
        filename = filename_list[mask_index]    
        f = pyfits.open(filename)
        tbdata=f[1].data
        #print tbdata.dtype
        mask = (tbdata['TEFF_LIT'] != 0)
        for slitindex in np.arange(len(mask))[mask]:
            ba_results.append((tbdata['MASK'][slitindex],tbdata['SLIT'][slitindex],
                tbdata['OBJNAME'][slitindex],tbdata['RA'][slitindex],tbdata['DEC'][slitindex],
                tbdata['TEFF'][slitindex],tbdata['LOGG'][slitindex],tbdata['FEH'][slitindex],
                np.sqrt(tbdata['FEHERR'][slitindex]**2+fehsyserr**2),tbdata['ALPHAFE'][slitindex],np.sqrt(tbdata['ALPHAFEERR'][slitindex]**2+alphasyserr**2),
                tbdata['BAFE'][slitindex],tbdata['BAFEERR'][slitindex],tbdata['OBJNAME_LIT'][slitindex],
                tbdata['FEH_LIT'][slitindex],tbdata['FEHERR_LIT'][slitindex],tbdata['ALPHAFE_LIT'][slitindex],
                tbdata['ALPHAFEERR_LIT'][slitindex],tbdata['BAFE_LIT'][slitindex],tbdata['BAFEERR_LIT'][slitindex],
                tbdata['LIT_COMMENT'][slitindex]
                ))
    ba_results = np.array(ba_results,dtype=[('MASK', '|S7'),('SLIT','i'),
                                        ('OBJNAME', '|S25'),('RA','<f8'),('DEC','<f8'),
                                        ('TEFF', '<f8'),('LOGG', '<f8'),  ('FEH', '<f8'),
                                        ('FEHERR', '<f8'),('ALPHAFE','<f8'),('ALPHAFEERR','<f8'),
                                        ('BAFE', '<f8'),('BAFEERR', '<f8'),('OBJNAME_LIT', '|S25'),
                                        ('FEH_LIT', '<f8'),('FEHERR_LIT', '<f8'),('ALPHAFE_LIT','<f8'),
                                        ('ALPHAFEERR_LIT','<f8'),('BAFE_LIT', '<f8'),('BAFEERR_LIT', '<f8'),
                                        ('LIT_COMMENT', '|S7')])
    table_filename = 'HRS_comp_table'+name_fragment+'.txt'
    with open(table_filename,'w') as f:
        f.write('MASK,SLITINDEX,OBJNAME,RA,DEC,TEFF,LOGG,FEH,FEHERR,ALPHAFE,'\
        'ALPHAFEERR,BAFE,BAFEERR,OBJANME_LIT,FEH_LIT,FEHERR_LIT,ALPHAFE_LIT,'\
        'ALPHAFEERR_LIT,BAFE_LIT,BAFEERR_LIT,LIT_COMMENT \n')
        for line in ba_results:
            f.write(','.join([str(item) for item in line])+'\n')
        f.close()
    #print ba_results,ba_results.dtype
    if len(ba_results)==0:
        print "No barium results to show"
        return ''
    #if debug==True:
    return table_filename
  
def plot_HRS_comparison(moogify_filename_ba=[''], name_fragment=''):
    HRS = np.array([])
    MRS = np.array([])
    HRS_err = np.array([])
    MRS_err = np.array([])
    REF = np.array([])
    for i in range(len(moogify_filename_ba)):
        filename = moogify_filename_ba[i]
        f = pyfits.open(filename)
        data = f[1].data
        mask = (data['BAFEERR']<accuracy_cutoff)&(data['BAFEERR']>0)&(data['BAFE']<0.9)
        ############## DELETE last condition if you expand the sample. Removed duplicate star.
        HRS= np.concatenate((HRS,data['BAFE_LIT'][mask]))
        MRS = np.concatenate((MRS,data['BAFE'][mask]))
        HRS_err = np.concatenate((HRS_err,data['BAFEERR_LIT'][mask]))
        MRS_err = np.concatenate((MRS_err,data['BAFEERR'][mask]))
        ref=' '.join([string.split('#')[0] for string in np.unique(data[mask]['LIT_COMMENT'])])
        REF = np.concatenate((REF,np.array([ref]*np.sum(mask))))

    print 'mean(MRS-HRS)',np.mean(MRS-HRS)
    print 'std with sigma=0.14',np.std((MRS-HRS)/np.sqrt(HRS_err**2+MRS_err**2+0.14**2))
    mask_halo = (REF == 'ful00')
    mask_other = (REF != 'ful00')
    print 'mean(MRS-HRS) halo',np.mean(MRS[mask_halo]-HRS[mask_halo])
    print 'mean(MRS-HRS) other',np.mean(MRS[mask_other]-HRS[mask_other])
    print 'std with sigma=0.17 halo',np.std((MRS[mask_halo]-HRS[mask_halo])/
                                       np.sqrt(HRS_err[mask_halo]**2+MRS_err[mask_halo]**2+0.17**2))
    print 'std with sigma=0 other',np.std((MRS[mask_other]-HRS[mask_other])/
                                       np.sqrt(HRS_err[mask_other]**2+MRS_err[mask_other]**2+0**2))

    fig3, axs = plt.subplots(2, figsize=(8,12),gridspec_kw = {'height_ratios':[2, 1]})
    fig3.subplots_adjust(bottom=0.11,left=0.18,top=0.95,hspace=0.24)
    plt.setp([a.minorticks_on() for a in fig3.axes[:]])
    axs = axs.ravel()
    REFS = np.unique(REF)
    #print REFS
    for ref in REFS:
        #print ref
        mask = (REF==ref)
        if ref == 'ful00':
            fmt = 's'
        else:
            fmt='o'
        if ref == 'coh12':
            ref = 'J.G.Cohen+ 2012'
        elif ref == 'ful00':
            ref = 'J.P.Fulbright 2000'
        elif ref == 'lee05':
            ref = 'J.Lee+ 2005'
        elif 'sne' in ref:
            ref = 'C.Sneden+ 1997,2000'

        axs[0].errorbar(HRS[mask],MRS[mask],fmt=fmt,
             xerr=HRS_err[mask],yerr=MRS_err[mask],
             label=ref) 
        axs[1].errorbar((HRS[mask]+MRS[mask])/2.,MRS[mask]-HRS[mask],fmt=fmt,
             yerr = np.sqrt(HRS_err[mask]**2+MRS_err[mask]**2),
             label=ref)         
     
    axs[0].set_xlabel("[Ba/Fe]$_{HRS}$")
    axs[0].set_ylabel("[Ba/Fe]$_{MRS}$")
    axs[0].plot([-0.6,1],[-0.6,1],':k')
    axs[0].axis([-0.6,1,-0.6,1])
    axs[0].set_xticks(np.arange(-0.6, 1.2, step=0.2))
    #axs[0].set_ylim([-0.6,1])
    axs[0].legend(loc=(0.55,0.05))
    #plt.title("Cluster Comparison Measurements", y=1.05)
    #plt.savefig(plot_path+name_fragment+"baHRS_moogify.png")
    #if plot==True:
    #plt.show(fig3)
    #plt.close(fig3)

    #fig2 = plt.figure(figsize=(8,4))
    #fig2.subplots_adjust(bottom=0.16,left=0.18)
    axs[1].set_xlabel("([Ba/Fe]$_{MRS}$+[Ba/Fe]$_{HRS})/2$")
    axs[1].set_ylabel("[Ba/Fe]$_{MRS}$-[Ba/Fe]$_{HRS}$")
    axs[1].plot([-0.6,1],[0,0],':k')
    axs[1].set_xlim([-0.6,1])
    axs[1].set_ylim([-0.5,1])
    axs[1].set_xticks(np.arange(-0.6, 1.2, step=0.2))
    #plt.title("Cluster Comparison Measurements", y=1.05)
    plt.savefig(plot_path+name_fragment+"baHRS_full_moogify.png")
    #if plot==True:
    plt.show()
    plt.close()              

def plot_nlte_comparison(name_fragment='',show_plot=True):
    stars = [['bdra1',12],['bfor6',84,88,89,9],['bscl1',53,63,7],['bscl2',72],['bscl6',52,69],['bumi1',23]]
    bah_nonsolar = [[-1.48],[1.44,1.24,1.54,1.09],[0.17,-1.00,-0.46],[1.93],[0.56,0.36],[1.04]]
    bah_nlte = []
    bah_lte = []
    baherr_lte = []
    teff = []
    bafe_nlte = []
    bafe_lte = []
    bafeerr_lte = []
    for mask_ind in range(len(stars)):
        #print stars[mask_ind][0], stars[mask_ind]
        filename = '/raid/gduggan/moogify/%s_moogify_ba.fits.gz'%stars[mask_ind][0]   
        f = pyfits.open(filename)
        data = f[1].data
        for list_ind in range(1,len(stars[mask_ind])):
            star_ind = stars[mask_ind][list_ind]
            #print star_ind, bah_nonsolar[mask_ind][list_ind-1],data[star_ind]['BAFE']+data[star_ind]['FEH'],data[star_ind]['TEFF']
            bah_nlte.append(bah_nonsolar[mask_ind][list_ind-1]-bah_solar)
            bah_lte.append(data[star_ind]['BAFE']+data[star_ind]['FEH'])
            teff.append(data[star_ind]['TEFF'])
            bafe_nlte.append(bah_nonsolar[mask_ind][list_ind-1]-bah_solar-data[star_ind]['FEH'])
            bafe_lte.append(data[star_ind]['BAFE'])
            bafeerr_lte.append(data[star_ind]['BAFEERR'])
            baherr_lte.append(np.sqrt(data[star_ind]['FEHERR']**2+fehsyserr**2+data[star_ind]['BAFEERR']**2))

    fig1 = plt.figure(figsize=(8,8))
    fig1.subplots_adjust(bottom=0.16,left=0.18)            
    plt.errorbar(bah_nlte,bah_lte,yerr=baherr_lte,fmt='o')
    plt.plot([min(bah_nlte+bah_lte),max(bah_nlte+bah_lte)],
             [min(bah_nlte+bah_lte),max(bah_nlte+bah_lte)],'k:')
    plt.xlabel('[Ba/H]$_{NLTE}$')
    plt.ylabel('[Ba/H]$_{LTE}$')
    plt.savefig(plot_path+name_fragment+"bah_nlte_comp.png")
    if show_plot == True:
        plt.show()
    plt.close(fig1)
    
    fig2 = plt.figure(figsize=(8,4))
    fig2.subplots_adjust(bottom=0.16,left=0.18)    
    plt.errorbar(bah_lte,np.array(bah_nlte)-np.array(bah_lte),yerr=baherr_lte,fmt='o')
    plt.plot([min(bah_lte),max(bah_lte)],[0,0],'k:')
    plt.xlabel('[Ba/H]$_{LTE}$')
    plt.ylabel('[Ba/H]$_{NLTE}$-[Ba/H]$_{LTE}$')
    plt.savefig(plot_path+name_fragment+"bah_nlte_diff.png")
    if show_plot == True:
        plt.show()
    plt.close(fig2)

    fig3 = plt.figure(figsize=(8,4))
    fig3.subplots_adjust(bottom=0.16,left=0.18)        
    plt.errorbar(teff,np.array(bah_nlte)-np.array(bah_lte),yerr=baherr_lte,fmt='o')
    plt.plot([min(teff),max(teff)],[0,0],'k:')
    plt.xlabel('T$_{eff}$')
    plt.ylabel('[Ba/H]$_{NLTE}$-[Ba/H]$_{LTE}$')
    plt.savefig(plot_path+name_fragment+"bah_nlte_diff_teff.png")
    if show_plot == True:
        plt.show()
    plt.close(fig3)

    fig1 = plt.figure(figsize=(8,8))
    fig1.subplots_adjust(bottom=0.16,left=0.18)            
    plt.errorbar(bafe_nlte,bafe_lte,yerr=bafeerr_lte,fmt='o')
    plt.plot([min(bafe_nlte+bafe_lte),max(bafe_nlte+bafe_lte)],
             [min(bafe_nlte+bafe_lte),max(bafe_nlte+bafe_lte)],'k:')
    plt.xlabel('[Ba/Fe]$_{NLTE}$')
    plt.ylabel('[Ba/Fe]$_{LTE}$')
    plt.savefig(plot_path+name_fragment+"bafe_nlte_comp.png")
    if show_plot == True:
        plt.show()
    plt.close(fig1)
    
    fig2 = plt.figure(figsize=(8,4))
    fig2.subplots_adjust(bottom=0.16,left=0.18)    
    plt.errorbar(bah_lte,np.array(bafe_nlte)-np.array(bafe_lte),yerr=bafeerr_lte,fmt='o')
    plt.plot([min(bah_lte),max(bah_lte)],[0,0],'k:')
    plt.xlabel('[Ba/H]$_{LTE}$')
    plt.ylabel('[Ba/Fe]$_{NLTE}$-[Ba/Fe]$_{LTE}$')
    plt.savefig(plot_path+name_fragment+"bafe_nlte_diff_bah.png")
    if show_plot == True:
        plt.show()
    plt.close(fig2)

    fig3 = plt.figure(figsize=(8,4))
    fig3.subplots_adjust(bottom=0.16,left=0.18)        
    plt.errorbar(teff,np.array(bafe_nlte)-np.array(bafe_lte),yerr=bafeerr_lte,fmt='o')
    plt.plot([min(teff),max(teff)],[0,0],'k:')
    plt.xlabel('T$_{eff}$')
    plt.ylabel('[Ba/Fe]$_{NLTE}$-[Ba/Fe]$_{LTE}$')
    plt.savefig(plot_path+name_fragment+"bafe_nlte_diff_teff.png")
    if show_plot == True:
        plt.show()
    plt.close(fig3)
        
    ##### compare four fornax stars to lte and nlte literature values. Teff v. [Ba/H] and [Ba/Fe] 
    stars = ['bfor6',84,88,89,9]
    letarte_names = ['BL304','BL260','BL210','BL203']
    bah_nonsolar = [1.44,1.24,1.54,1.09]
    bah_nonsolar_And17 = [1.27,1.17,1.67,1.37]
    feh_letarte = [-0.97,-0.87,-0.76,-0.83]
    feherr_letarte = [0.06,0.06,0.07,0.07]
    bafe_lte_letarte = [0.55,0.07,0.90,0.58]
    bafeerr_lte_letarte = [0.14,0.13,0.14,0.18]
    teff_letarte = [3950,4009,4062,4037]
    filename = '/raid/gduggan/moogify/%s_moogify_ba.fits.gz'%stars[0]   
    f = pyfits.open(filename)
    data = f[1].data
    teff = []
    feh = []
    feherr = []
    bah_lte = []
    baherr_lte = []
    bafe_lte = []
    bafeerr_lte = []
    for star_ind in stars[1:]:
        bafe_lte.append(data[star_ind]['BAFE'])
        bafeerr_lte.append(data[star_ind]['BAFEERR'])
        teff.append(data[star_ind]['TEFF'])
        feh.append(data[star_ind]['FEH'])
        feherr.append(np.sqrt(data[star_ind]['FEHERR']**2+fehsyserr**2))
        bah_lte.append(data[star_ind]['BAFE']+data[star_ind]['FEH'])
        baherr_lte.append(np.sqrt(data[star_ind]['FEHERR']**2+fehsyserr**2+data[star_ind]['BAFEERR']**2))

    bah_lte_letarte = bafe_lte_letarte + feh_letarte
    bafe_nlte = np.array(bah_nonsolar)-bah_solar - np.array(feh)
    bafe_nlte_And17 = np.array(bah_nonsolar_And17)-bah_solar-np.array(feh_letarte)
    
    fig4 = plt.figure(figsize=(8,4))
    fig4.subplots_adjust(bottom=0.16,left=0.18,right=0.75)        
    plt.errorbar(range(1,5),bafe_lte,yerr=bafeerr_lte,fmt='g^',label='LTE')
    plt.errorbar(range(1,5),bafe_nlte,fmt='go',label='non-LTE')
    plt.errorbar(range(1,5),bafe_lte_letarte,yerr=bafeerr_lte_letarte,fmt='^',color='darkgoldenrod',label='LTE Let10')
    plt.errorbar(range(1,5),bafe_nlte_And17,fmt='o',color='darkgoldenrod',label='non-LTE And17')
    plt.plot([1,4],[0,0],'k:')
    plt.xticks(range(1,5),letarte_names)
    plt.ylabel('[Ba/Fe]')
    #plt.legend(loc=1)
    #plt.legend(loc = 'lower left',bbox_to_anchor=(1, -1),frameon=True)
    plt.legend(bbox_to_anchor=(0.98, 1))
    plt.savefig(plot_path+name_fragment+"bafe_nlte_fornax.png")
    if show_plot == True:
        plt.show()
    plt.close(fig4)        
        
    fig5 = plt.figure(figsize=(8,4))
    fig5.subplots_adjust(bottom=0.16,left=0.18,right=0.75)        
    for star_ind in range(len(stars)-1): 
        color_list = ['g','darkgoldenrod','m','b']     
        plt.errorbar(feh[star_ind],bafe_lte[star_ind],yerr=bafeerr_lte[star_ind],
                     xerr=feherr[star_ind],fmt='^',color=color_list[star_ind],label='LTE')
        plt.errorbar(feh[star_ind],bafe_nlte[star_ind],fmt='o',xerr=feherr[star_ind],
                     color=color_list[star_ind],label='non-LTE')
        plt.errorbar(feh_letarte[star_ind],bafe_lte_letarte[star_ind],yerr=bafeerr_lte_letarte[star_ind],
                     xerr=feherr_letarte[star_ind],fmt='>',color=color_list[star_ind],label='LTE Let10')
        plt.errorbar(feh_letarte[star_ind],bafe_nlte_And17[star_ind],
                     xerr=feherr_letarte[star_ind],fmt='s',color=color_list[star_ind],label='non-LTE And17')
        if star_ind==0:
            plt.legend(bbox_to_anchor=(0.98, 1))
    plt.plot([min(feh+feh_letarte),max(feh+feh_letarte)],[0,0],'k:')
    plt.xlabel('[Fe/H]')
    plt.ylabel('[Ba/Fe]')
    plt.savefig(plot_path+name_fragment+"bafe_nlte_feh_fornax.png")
    if show_plot == True:
        plt.show()
    plt.close(fig5)           

    fig7 = plt.figure(figsize=(8,4))
    fig7.subplots_adjust(bottom=0.16,left=0.18,right=0.75)        
    for star_ind in range(len(stars)-1): 
        color_list = ['g','darkgoldenrod','m','b']     
        plt.errorbar(teff[star_ind],bafe_lte[star_ind],yerr=bafeerr_lte[star_ind],
                     fmt='^',color=color_list[star_ind],label='LTE')
        plt.errorbar(teff[star_ind],bafe_nlte[star_ind],fmt='o',
                     color=color_list[star_ind],label='non-LTE')
        plt.errorbar(teff_letarte[star_ind],bafe_lte_letarte[star_ind],yerr=bafeerr_lte_letarte[star_ind],
                     fmt='>',color=color_list[star_ind],label='LTE Let10')
        plt.errorbar(teff_letarte[star_ind],bafe_nlte_And17[star_ind],
                     fmt='s',color=color_list[star_ind],label='non-LTE And17')
        if star_ind==0:
            plt.legend(bbox_to_anchor=(0.98, 1))
    plt.plot([min(teff+teff_letarte),max(teff+teff_letarte)],[0,0],'k:')
    plt.xlabel('T$_{eff}$')
    plt.ylabel('[Ba/Fe]')
    plt.savefig(plot_path+name_fragment+"bafe_nlte_teff_fornax.png")
    if show_plot == True:
        plt.show()
    plt.close(fig7)     

    fig8 = plt.figure(figsize=(8,4))
    fig8.subplots_adjust(bottom=0.16,left=0.18,right=0.75)        
    for star_ind in range(len(stars)-1): 
        color_list = ['g','darkgoldenrod','m','b']     
        plt.errorbar(bah_lte[star_ind],bafe_lte[star_ind],yerr=bafeerr_lte[star_ind],
                     fmt='^',color=color_list[star_ind],label='LTE')
        plt.errorbar(bah_lte[star_ind],bafe_nlte[star_ind],fmt='o',
                     color=color_list[star_ind],label='non-LTE')
        plt.errorbar(bah_lte_letarte[star_ind],bafe_lte_letarte[star_ind],yerr=bafeerr_lte_letarte[star_ind],
                     fmt='>',color=color_list[star_ind],label='LTE Let10')
        plt.errorbar(bah_lte_letarte[star_ind],bafe_nlte_And17[star_ind],
                     fmt='s',color=color_list[star_ind],label='non-LTE And17')
        if star_ind==0:
            plt.legend(bbox_to_anchor=(0.98, 1))
    plt.plot([min(bah_lte+bah_lte_letarte),max(bah_lte+bah_lte_letarte)],[0,0],'k:')
    plt.xlabel('[Ba/H]$_{LTE}$')
    plt.ylabel('[Ba/Fe]')
    plt.savefig(plot_path+name_fragment+"bafe_nlte_bah_fornax.png")
    if show_plot == True:
        plt.show()
    plt.close(fig8)    
         
def make_barium_results_table(name_fragment="", overwrite=False):
    name_list = ['dra1','dra2','dra3','umi1','umi2','umi3','scl1','scl2','scl6','for6','sex2','sex3']
    filename_list = ['/raid/gduggan/moogify/b%s_moogify_ba.fits.gz'%name for name in name_list]   

    ba_results = []
    for mask_index in range(len(filename_list)):
        dsph_name = slitmask_to_dsph(name_list[mask_index])
        filename = filename_list[mask_index]    
        f = pyfits.open(filename)
        tbdata=f[1].data
        #print tbdata.dtype
        mask = (tbdata['BAFEERR']<accuracy_cutoff)&(np.around(tbdata['BAFEERR'],3)>0.0)& \
                (tbdata['ALPHAFEERR']<accuracy_cutoff)&(np.around(tbdata['ALPHAFEERR'],3)>0.0)
        for slitindex in np.arange(len(mask))[mask]:
            ba_results.append((dsph_name,tbdata['MASK'][slitindex],tbdata['SLIT'][slitindex],
                tbdata['OBJNAME'][slitindex],tbdata['RA'][slitindex],tbdata['DEC'][slitindex],
                tbdata['TEFF'][slitindex],tbdata['LOGG'][slitindex],tbdata['FEH'][slitindex],
                np.sqrt(tbdata['FEHERR'][slitindex]**2+fehsyserr**2),
                tbdata['ALPHAFE'][slitindex],np.sqrt(tbdata['ALPHAFEERR'][slitindex]**2+alphasyserr**2),
                tbdata['MGFE'][slitindex],tbdata['MGFEERR'][slitindex],
                tbdata['BAFE'][slitindex],tbdata['BAFEERR'][slitindex]
                ))
    ba_results = np.array(ba_results,dtype=[('DSPH', '|S7'),('MASK', '|S7'),('SLIT','i'),
                                        ('OBJNAME', '|S25'),('RA','<f8'),('DEC','<f8'),
                                        ('TEFF', '<f8'),('LOGG', '<f8'),  ('FEH', '<f8'),
                                        ('FEHERR', '<f8'),('ALPHAFE','<f8'),('ALPHAFEERR','<f8'),
                                        ('MGFE', '<f8'),('MGFEERR', '<f8'),
                                        ('BAFE', '<f8'),('BAFEERR', '<f8')])
    table_filename = 'barium_results_table'+name_fragment+'.txt'
    print ba_results.dtype.names
    with open(table_filename,'w') as f:
        f.write(','.join(['{:>10}'.format(str(item))[:10] for item in ba_results.dtype.names])+'\n')
        for line in ba_results:
            f.write(','.join(['{:>10}'.format(str(item))[:10] for item in line])+'\n')
        f.close()
    #print ba_results,ba_results.dtype
    if len(ba_results)==0:
        print "No barium results to show"
        return ''
    #if debug==True:
    return table_filename

def plot_bafe_feh(dsph_name, moogify_filename_ba=[],use_catalog=False, plot_full=False, name_fragment='', outliers=True, show_plot=False):
    if len(moogify_filename_ba) == 0 and use_catalog==False:
        print "ERROR: Declare what moogify files to use"
        return
    
    fig3 = plt.figure(figsize=(8,6))
    fig3.subplots_adjust(bottom=0.16,left=0.18)
    if outliers == True:
        name_fragment = '_outliers' + name_fragment
    if use_catalog == True:
        name_fragment = '_catalog' + name_fragment  
        moogify_filename_ba = ['filler']      
    if plot_full == True:
        #plt.ylim([-1.7,1.1])
        #plt.xlim([-3.1,-0.4])
        plt.ylim([-2.3,1.7])
        plt.xlim([-3.2,-0.4])
        print plot_path, dsph_name, name_fragment
        save_name = plot_path+dsph_name+name_fragment+"_ba4.png"  
    else:
        if (dsph_name not in ["For","Scl"]):
            #plt.ylim([-1.6,1.1])
            #plt.xlim([-3.1,-0.4])
            save_name = plot_path+dsph_name+name_fragment+"_ba5.png"        
        elif dsph_name == "For":
            #plt.ylim([-1.3,1.3])
            #plt.xlim([-2.8,0.2])
            save_name = plot_path+dsph_name+name_fragment+"_ba6.png"           
        elif dsph_name == "Scl":
            #plt.ylim([-3.5,1.1])
            #plt.xlim([-4.2,-.8])    
            save_name = plot_path+dsph_name+name_fragment+"_ba2.png"     
        else:
            print "should not go here :)"
            #plt.ylim([-2.1,1.1])
            #plt.xlim([-3.1,0.1])        
            save_name = plot_path+dsph_name+name_fragment+"_ba2.png"    
    #################### include systematic error! Use load_ba_results! Change >0.0 to systematic error
    Total_stars = 0
    for filename in moogify_filename_ba:
        if use_catalog == True:
            ba_results, mask, ba_final = load_ba_results(dsph_name)
        else:
            slitmask = pyfits.open(filename)
            ba_results = slitmask[1].data
            print filename
            print 'BAFEERR: ', np.mean(ba_results['BAFEERR']),np.min(ba_results['BAFEERR']),np.max(ba_results['BAFEERR'])
            print 'BAFEERR:',ba_results['BAFEERR'], ba_results['BAFE']
            print 'FEHERR:',ba_results['FEHERR'], ba_results['FEH']
            print 'ALPHAFEERR:',ba_results['ALPHAFEERR'], ba_results['ALPHAFE']
    
            ba_results['FEHERR'] = np.sqrt(ba_results['FEHERR']**2+fehsyserr**2)
            ba_results['ALPHAFEERR'] = np.sqrt(ba_results['ALPHAFEERR']**2+alphasyserr**2)
            ba_results['BAFEERR'] = np.sqrt(ba_results['BAFEERR']**2+bafesyserr**2)
            ba_results['MGFEERR'] = np.sqrt(ba_results['MGFEERR']**2+mgfesyserr**2)
            
        #if outliers == True:
        #    mask = (ba_results['BAFEERR']<accuracy_cutoff)&(np.around(ba_results['BAFEERR'],3)>0.0)& \
        #        (np.sqrt(ba_results['ALPHAFEERR']**2+alphasyserr**2)<accuracy_cutoff)&\
        #        (np.around(np.sqrt(ba_results['ALPHAFEERR']**2+alphasyserr**2),3)>0.0)&\
        #        (-2.0<np.around(ba_results['BAFE'],3)) & (np.around(ba_results['BAFE'],3)<1.0) 
        #    maskoutlier1 = (np.around(ba_results['BAFE'],3)>=1.0) 
        #    maskoutlier2 = (np.around(ba_results['BAFE'],3)<=-2.0)
        #    maskoutlier3 = (np.around(ba_results['BAFEERR'],3)==0.0)    
        #    maskoutlier = np.sum([maskoutlier1,maskoutlier2,maskoutlier3],axis=0,dtype=bool)
            mask = (ba_results['BAFEERR']<accuracy_cutoff)&(np.around(ba_results['BAFEERR'],3)>bafesyserr)& \
                (ba_results['ALPHAFEERR']<accuracy_cutoff)&(np.around(ba_results['ALPHAFEERR'],3)>alphasyserr)& \
                (ba_results['FEHERR']<accuracy_cutoff)&(np.around(ba_results['FEHERR'],3)>fehsyserr)
        print ba_results.shape[0], np.sum(mask), (ba_results['BAFEERR']<accuracy_cutoff)&(np.around(ba_results['BAFEERR'],3)>bafesyserr)
        print 'BAFEERR: ', np.mean(ba_results['BAFEERR']),np.min(ba_results['BAFEERR']),np.max(ba_results['BAFEERR'])
        print (ba_results['ALPHAFEERR']<accuracy_cutoff)&(np.around(ba_results['ALPHAFEERR'],3)>alphasyserr)
        print (ba_results['FEHERR']<accuracy_cutoff)&(np.around(ba_results['FEHERR'],3)>fehsyserr)
        ba_results_clean = ba_results[mask]
        print (filename.split('/')[-1]).split('_')[0], ba_results_clean.shape[0]
        Total_stars = Total_stars + ba_results_clean.shape[0]
        if ba_results_clean.shape[0]>0: #if the array isn't empty
            print 'BAFE: ', np.nanmean(ba_results_clean['BAFE']),np.nanmin(ba_results_clean['BAFE']),np.nanmax(ba_results_clean['BAFE'])
            print 'BAFEERR: ', np.mean(ba_results_clean['BAFEERR']),np.min(ba_results_clean['BAFEERR']),np.max(ba_results_clean['BAFEERR'])
            print 'FEH: ', np.mean(ba_results_clean['FEH']),np.min(ba_results_clean['FEH']),np.max(ba_results_clean['FEH'])
            print 'FEHERR: ', np.mean(ba_results_clean['FEHERR']),np.min(ba_results_clean['FEHERR']),np.max(ba_results_clean['FEHERR'])
            print 'ALPHAFE: ', np.mean(ba_results_clean['ALPHAFE']),np.min(ba_results_clean['ALPHAFE']),np.max(ba_results_clean['ALPHAFE'])
            print 'ALPHAFEERR: ', np.mean(ba_results_clean['ALPHAFEERR']),np.min(ba_results_clean['ALPHAFEERR']),np.max(ba_results_clean['ALPHAFEERR'])

            #if (plot_full == True) or (dsph_name not in ["For","Scl"]):
            #    maskupper2 = np.around(ba_results_clean['BAFE'],3)<=-1.5 
            #    plt.plot(ba_results_clean['FEH'][maskupper2],[-1.5]*sum(maskupper2),'vk')
            plt.errorbar(ba_results_clean['FEH'],ba_results_clean['BAFE'],
                         fmt='ok',xerr=ba_results_clean['FEHERR'],
                         yerr=ba_results_clean['BAFEERR'],ecolor='0.6')
            if outliers == True:
                good_outliers = [['dra2','27'],['dra3','0'],['for6','78'],['for6','27'],['for6','31'],['umi2','0'],['umi2','41'],['scl2','72'],['scl6','20']]
                results = np.array([ba_results_clean['MASK'],ba_results_clean['SLIT']]).T
                masklower = np.array([results.tolist()[i] in good_outliers for i in range(len(results))])
                maskupper = np.around(ba_results_clean['BaFe'],3)<-2.0
        
                print 'outliers\n',[[ba_results['MASK'][i], ba_results['SLIT'][i]] for i in np.arange(len(maskoutlier))[maskoutlier]]   
                print 'lower limits\n',[[ba_results['MASK'][i], ba_results['SLIT'][i]] for i in np.arange(len(masklower))[masklower]]   

                plt.plot(ba_results_clean['FEH'][masklower],ba_results_clean['BAFE'][masklower],'^k')
                plt.plot(ba_results_clean['FEH'][maskupper],ba_results_clean['BAFE'][maskupper],'vk')
    plt.xlabel("[Fe/H]")
    plt.ylabel(r"[Ba/Fe]")
    plt.title(dsph_name+" Barium Enhancement Measurements of %i stars"%(Total_stars),y=1.05)
    plt.plot([-5,1],[-0,0],'k:')
    plt.minorticks_on()   
    plt.savefig(save_name)
    if show_plot == True:
        plt.show(fig3)
    plt.close(fig3)

def plot_scatter(dsph_name, moogify_filename_ba, x_keyword, y_keyword, plot_full=False, name_fragment='', show_plot=False):
    save_name = plot_path+dsph_name+x_keyword+y_keyword+name_fragment+".png"        
    fig3 = plt.figure(figsize=(8,6))
    fig3.subplots_adjust(bottom=0.16,left=0.18)
    Total_stars = 0
    x_data=np.array([])
    x_error=np.array([])
    y_data=np.array([])
    y_error=np.array([])
    for filename in moogify_filename_ba:
        slitmask = pyfits.open(filename)
        data = slitmask[1].data
        mask = (data['BAFEERR']<accuracy_cutoff)&(np.around(data['BAFEERR'],3)>0.0)& \
                (data['ALPHAFEERR']<accuracy_cutoff)&(np.around(data['ALPHAFEERR'],3)>0.0)
               
        ba_results_clean = data[mask]
        Total_stars = Total_stars + ba_results_clean.shape[0]
        print (filename.split('/')[-1]).split('_')[0], ba_results_clean.shape[0]
                                
        if ba_results_clean.shape[0]>0: #if the array isn't empty
            # add in systematic uncertainty
            if 'FEH' in [x_keyword,y_keyword]:
                ba_results_clean['FEHERR'] = np.sqrt(ba_results_clean['FEHERR']**2+fehsyserr**2)
            if 'ALPHAFE' in [x_keyword,y_keyword]:
                ba_results_clean['ALPHAFEERR'] = np.sqrt(ba_results_clean['ALPHAFEERR']**2+alphasyserr**2)
            # account for exceptions
            if x_keyword=='BAH':
                x_data_mask=ba_results_clean['FEH']+ba_results_clean['BAFE']
                x_error_mask=np.sqrt(ba_results_clean['FEHERR']**2+ba_results_clean['BAFEERR']**2)
            elif x_keyword == 'SN':
                x_data_mask = ba_results_clean[x_keyword]
                x_error_mask = np.zeros(len(x_data_mask))
            elif 'ERR' in x_keyword:
                x_data_mask = ba_results_clean[x_keyword]
                x_error_mask = np.zeros(len(x_data_mask))                
            else:
                x_data_mask = ba_results_clean[x_keyword]
                x_error_mask = ba_results_clean[x_keyword+'ERR']                
            if y_keyword=='BAH':
                y_data_mask=ba_results_clean['FEH']+ba_results_clean['BAFE']
                y_error_mask=np.sqrt(ba_results_clean['FEHERR']**2+ba_results_clean['BAFEERR']**2)
            elif y_keyword == 'SN':
                y_data_mask = ba_results_clean[y_keyword]
                y_error_mask = np.ones(len(y_data_mask))
            elif 'ERR' in y_keyword:
                y_data_mask = ba_results_clean[y_keyword]
                y_error_mask = np.zeros(len(y_data_mask))                
            else:                
                y_data_mask = ba_results_clean[y_keyword]
                y_error_mask = ba_results_clean[y_keyword+'ERR']

            x_data = np.concatenate((x_data,x_data_mask))
            x_error = np.concatenate((x_error,x_error_mask))
            y_data = np.concatenate((y_data,y_data_mask))
            y_error = np.concatenate((y_error,y_error_mask))
                
    print x_keyword, np.nanmean(x_data),np.nanmin(x_data),np.nanmax(x_data)
    print x_keyword+'ERR', np.nanmean(x_error),np.nanmin(x_error),np.nanmax(x_error)
    print y_keyword, np.nanmean(y_data),np.nanmin(y_data),np.nanmax(y_data)
    print y_keyword+'ERR', np.nanmean(y_error),np.nanmin(y_error),np.nanmax(y_error)

    plt.errorbar(x_data,y_data,fmt='ok',xerr=x_error,yerr=y_error,ecolor='0.6')
            
    plt.xlabel(x_keyword)
    plt.ylabel(y_keyword)
    plt.title(dsph_name+" %i stars"%(Total_stars),y=1.05)
    plt.minorticks_on()   
    plt.savefig(save_name)
    if show_plot == True:
        plt.show(fig3)
    plt.close(fig3)
        
def plot_hist(dsph_name, moogify_filename_ba, x_keyword, plot_full=False, name_fragment='', show_plot=False):
    save_name = plot_path+dsph_name+x_keyword+'_hist'+name_fragment+".png"        
    fig3 = plt.figure(figsize=(8,6))
    fig3.subplots_adjust(bottom=0.16,left=0.18)
    Total_stars = 0
    x_data=np.array([])
    for filename in moogify_filename_ba:
        slitmask = pyfits.open(filename)
        data = slitmask[1].data
        mask = (data['BAFEERR']<accuracy_cutoff)&(np.around(data['BAFEERR'],3)>0.0)& \
                (data['ALPHAFEERR']<accuracy_cutoff)&(np.around(data['ALPHAFEERR'],3)>0.0)
               
        ba_results_clean = data[mask]
        Total_stars = Total_stars + ba_results_clean.shape[0]
        print (filename.split('/')[-1]).split('_')[0], ba_results_clean.shape[0]
                                
        if ba_results_clean.shape[0]>0: #if the array isn't empty
            if x_keyword=='BAH':
                x_data=np.concatenate((x_data,ba_results_clean['FEH']+ba_results_clean['BAFE']))
            else:
                x_data = np.concatenate((x_data,ba_results_clean[x_keyword]))
                
    print x_keyword, np.nanmean(x_data),np.nanmin(x_data),np.nanmax(x_data)
            
    plt.hist(x_data)
            
    plt.xlabel(x_keyword)
    plt.title(dsph_name+" %i stars"%(Total_stars),y=1.05)
    plt.minorticks_on()   
    plt.savefig(save_name)
    if show_plot == True:
        plt.show(fig3)
    plt.close(fig3)
    
def plot_all(plot_full=True, copy_files = False,name_fragment='',show_plot=False):
    for dsph_name, name_list in [['Dra',['dra1','dra2','dra3']],
                                 ['UMi',['umi1','umi2','umi3']],
                                 ['Scl',['scl1','scl2','scl6']],
                                 ['For',['for6']],['Sex',['sex2','sex3']]]:
        filename_ba = ['/raid/gduggan/moogify/b%s_moogify_ba.fits.gz'%name for name in name_list]   
        plot_bafe_feh(dsph_name, filename_ba, plot_full=plot_full, name_fragment=name_fragment, 
                      outliers=False, show_plot=show_plot)
        plot_bafe_feh(dsph_name, filename_ba, plot_full=plot_full, name_fragment=name_fragment, 
                      outliers=True, show_plot=show_plot)
        plot_bafe_feh(dsph_name, filename_ba, name_fragment=name_fragment, 
                      plot_full=True, show_plot=show_plot)
        plot_scatter(dsph_name,filename_ba,x_keyword='FEH',y_keyword='BAFE',show_plot=show_plot)
        plot_scatter(dsph_name,filename_ba,x_keyword='FEH',y_keyword='ALPHAFE',show_plot=show_plot)
        plot_scatter(dsph_name,filename_ba,x_keyword='FEH',y_keyword='BAH',show_plot=show_plot)
        plot_scatter(dsph_name,filename_ba,x_keyword='TEFF',y_keyword='LOGG',show_plot=show_plot)
        plot_scatter(dsph_name,filename_ba,x_keyword='SN',y_keyword='BAFEERR',show_plot=show_plot)
        plot_hist(dsph_name,filename_ba,x_keyword='BAH',show_plot=show_plot)
        plot_hist(dsph_name,filename_ba,x_keyword='SN',show_plot=show_plot)

def plot_all_bafe(plot_full=True, use_catalog=True, copy_files = False,name_fragment='',show_plot=False):
    for dsph_name, name_list in [['Dra',['dra1','dra2','dra3']],
                                 ['UMi',['umi1','umi2','umi3']],
                                 ['Scl',['scl1','scl2','scl6']],
                                 ['For',['for6']],['Sex',['sex2','sex3']]]:
        filename_ba = ['/raid/gduggan/moogify/b%s_moogify_ba.fits.gz'%name for name in name_list]   
        plot_bafe_feh(dsph_name, filename_ba, use_catalog=use_catalog,plot_full=plot_full, name_fragment=name_fragment, 
                      outliers=False, show_plot=show_plot)
                
def measure_all():
    carbon_mask_list = ['7078d_blue','7078e_blue','bdra1','bdra2','bdra3','bfor6',
                        'bpal13','bscl1','bscl2','bscl6','bumi1','bumi2','bumi3',
                        'n2419b_blue','n4590a_blue'] #masks that have had carbon measured.
    carbon_filenames = np.array(['/raid/caltech/moogify/%s/moogifych.fits.gz'%
                                 name for name in carbon_mask_list])
    for dsph_name, name_list in [['Dra',['dra1','dra2','dra3']],
                                 ['UMi',['umi1','umi2','umi3']],
                                 ['Scl',['scl1','scl2','scl6']],
                                 ['For',['for6']],['Sex',['sex2','sex3']]]:
        filename = ['/raid/gduggan/moogify/b%s_moogify.fits.gz'%name for name in name_list]   
        measure_ba_duggan_moogify(filename,carbon_filenames,overwrite=True,
                                  verify_member=True,fine_tune_wvl=True,
                                  simple_line_correction=True,plot=False,
                                  debug=False, wvl_max_shift=20)   

def plot_single_spectrum_ba(slitmask,slitindex,save_ascii=False):
    filename = '/raid/gduggan/moogify/%s_moogify_ba.fits.gz'%slitmask
    f = pyfits.open(filename)
    # moogify file with '_ba' already has the corrected wavelength saved
    tbdata=f[1].data
    print tbdata.field('OBJNAME')[slitindex]
    print tbdata.field('SN')[slitindex]
    objnames = np.copy(tbdata.field('OBJNAME'))
    contdiv = np.copy(tbdata.field('CONTDIV')) #equals spectra/continuum
    contdivivar = np.copy(tbdata.field('CONTDIVIVAR'))
    contdivstd = np.sqrt(np.reciprocal(contdivivar))
    zrest = np.copy(tbdata.field('ZREST'))
    dlam = np.copy(tbdata.field('DLAM')*0.95) #resscale fudge factor
    teff = np.copy(tbdata.field('TEFF'))
    logg = np.copy(tbdata.field('LOGG'))
    feh = np.copy(tbdata.field('FEH'))
    alphafe = np.copy(tbdata.field('ALPHAFE'))
    wvls = tbdata['LAMBDA']         
    
    plt.plot(wvls[slitindex],contdiv[slitindex])
    
    #cycle through five barium lines and save wvl and flux to ascii
    if save_ascii == True:
        ascii_name = plot_path+(filename.split('/')[-1]).split('_')[0]+'_%i_ba.txt'%slitindex
        print ascii_name
        for i in range(1,6):
            mask = np.abs(wvls[slitindex]-xlines[i])<15
            array = np.column_stack((wvls[slitindex][mask],contdiv[slitindex][mask]))
            if i==1:
                array_full = np.copy(array)
            else:
                array_full = np.vstack((array_full,array))
            print i, array.shape, array_full.shape
        np.savetxt(ascii_name,array_full,delimiter=' ',fmt='%8g', header = 'Wvl (ang)    Flux')
        plt.plot(array_full[:,0],array_full[:,1])
        plt.show()     

def average_two_slitmasks(slitmask1,slitmask2):
    filename = '/raid/gduggan/moogify/%s_moogify_ba.fits.gz'%slitmask1
    f = pyfits.open(filename)
    # moogify file with '_ba' already has the corrected wavelength saved
    tbdata=f[1].data
    print tbdata.field('OBJNAME')
    print tbdata.field('SN')
    objnames = np.copy(tbdata.field('OBJNAME'))
    contdiv = np.copy(tbdata.field('CONTDIV')) #equals spectra/continuum
    contdivivar = np.copy(tbdata.field('CONTDIVIVAR'))
    contdivstd = np.sqrt(np.reciprocal(contdivivar))
    wvls = tbdata['LAMBDA']         

    filename2 = '/raid/gduggan/moogify/%s_moogify_ba.fits.gz'%slitmask2
    f2 = pyfits.open(filename2)
    # moogify file with '_ba' already has the corrected wavelength saved
    tbdata2=f2[1].data
    print tbdata2.field('OBJNAME')
    print tbdata2.field('SN')
    objnames2 = np.copy(tbdata2.field('OBJNAME'))
    contdiv2 = np.copy(tbdata2.field('CONTDIV')) #equals spectra/continuum
    contdivivar2 = np.copy(tbdata2.field('CONTDIVIVAR'))
    contdivstd2 = np.sqrt(np.reciprocal(contdivivar2))
    wvls2 = tbdata2['LAMBDA'] 
    
    obj_mask = [True]*len(wvls)
    print len(wvls)
    for slitindex in range(len(objnames)):
        slitindex2 = np.where(objnames2==objnames[slitindex])[0]
        if len(slitindex2)==0:
            obj_mask[slitindex]=False
            continue
        else: #multiple observations means the same star can be listed multiple times
            slitindex2=slitindex2[0]
        print slitindex,slitindex2,objnames[slitindex],objnames2[slitindex2]
        #plt.plot(wvls[slitindex],contdiv[slitindex])
        #plt.plot(wvls2[slitindex],contdiv2[slitindex])
        
        #cycle through five barium lines
        for i in range(1,6):
            mask = np.abs(wvls[slitindex]-xlines[i])<15
            min_index = np.argmin(np.abs(wvls2[slitindex2]-np.min(wvls[slitindex][mask])))
            mask2 = (min_index <= range(len(wvls2[slitindex2]))) & (min_index+np.sum(mask) > range(len(wvls2[slitindex2])))
            
            array = np.column_stack((wvls[slitindex][mask],contdiv[slitindex][mask],contdivstd[slitindex][mask]))
            if i==1:
                array_full = np.copy(array)
            else:
                array_full = np.vstack((array_full,array))
            array2 = np.column_stack((wvls2[slitindex2][mask2],contdiv2[slitindex2][mask2],contdivstd2[slitindex2][mask2]))
            if i==1:
                array_full2 = np.copy(array2)
            else:
                array_full2 = np.vstack((array_full2,array2))
            wvl_av = (array[:,0]+array2[:,0])/2.
            # weighted average
            contdiv_av = (array[:,1]/array[:,2]**2+array2[:,1]/array2[:,2]**2)/(1./array[:,2]**2+1./array2[:,2]**2)
            contdivstd_av = np.sqrt(1./(1./array[:,2]**2+1./array2[:,2]**2))
            contdivivar_av = 1./array[:,2]**2+1./array2[:,2]**2
            tbdata['LAMBDA'][slitindex][mask] = wvl_av
            tbdata['CONTDIV'][slitindex][mask] = contdiv_av
            tbdata['CONTDIVIVAR'][slitindex][mask] = contdivivar_av
            plt.errorbar(wvl_av,contdiv_av,yerr=contdivstd_av,fmt='o')        
            plt.errorbar(tbdata['LAMBDA'][slitindex][mask],tbdata['CONTDIV'][slitindex][mask],yerr=contdivstd_av,fmt='o')        
        plt.errorbar(array_full[:,0],array_full[:,1],yerr=array_full[:,2],fmt='o')
        plt.errorbar(array_full2[:,0],array_full2[:,1],yerr=array_full2[:,2],fmt='o')
        if slitindex > 70:
            plt.show()   
    print np.sum(obj_mask), len(wvls),len(wvls2)
    filename_new = '/raid/gduggan/moogify/%s_moogify.fits.gz'%slitmask1.split('_')[0]
    print 'Saving to: ',filename_new
    tbdata['BAFE'] = 0.0
    tbdata['BAFEERR'] = 0.0
    pyfits.writeto(filename_new,tbdata[obj_mask],overwrite=True)

    #check that it worked
    filename = '/raid/gduggan/moogify/%s_moogify_ba.fits.gz'%slitmask1
    f = pyfits.open(filename)
    # moogify file with '_ba' already has the corrected wavelength saved
    tbdata=f[1].data
    print tbdata.field('OBJNAME')
    print tbdata.field('SN')
    objnames = np.copy(tbdata.field('OBJNAME'))
    contdiv = np.copy(tbdata.field('CONTDIV')) #equals spectra/continuum
    contdivivar = np.copy(tbdata.field('CONTDIVIVAR'))
    contdivstd = np.sqrt(np.reciprocal(contdivivar))
    wvls = tbdata['LAMBDA']   
    
    filename2 = '/raid/gduggan/moogify/%s_moogify_ba.fits.gz'%slitmask2
    f2 = pyfits.open(filename2)
    # moogify file with '_ba' already has the corrected wavelength saved
    tbdata2=f2[1].data
    print tbdata2.field('OBJNAME')
    print tbdata2.field('SN')
    objnames2 = np.copy(tbdata2.field('OBJNAME'))
    contdiv2 = np.copy(tbdata2.field('CONTDIV')) #equals spectra/continuum
    contdivivar2 = np.copy(tbdata2.field('CONTDIVIVAR'))
    contdivstd2 = np.sqrt(np.reciprocal(contdivivar2))
    wvls2 = tbdata2['LAMBDA'] 
      
    f3 = pyfits.open(filename_new)
    # moogify file with '_ba' already has the corrected wavelength saved
    tbdata3=f3[1].data
    print tbdata3.field('OBJNAME')
    print tbdata3.field('SN')
    objnames3 = np.copy(tbdata3.field('OBJNAME'))
    contdiv3 = np.copy(tbdata3.field('CONTDIV')) #equals spectra/continuum
    contdivivar3 = np.copy(tbdata3.field('CONTDIVIVAR'))
    contdivstd3 = np.sqrt(np.reciprocal(contdivivar3))
    wvls3 = tbdata3['LAMBDA']         
    
    for slitindex3 in range(len(objnames3)):
        slitindex2 = np.where(objnames2==objnames3[slitindex3])[0]
        slitindex = np.where(objnames==objnames3[slitindex3])[0]
        if len(slitindex2)==0:
            print 'Ahhh, should not happpen',objnames3[slitindex3]
            continue
        else: #multiple observations means the same star can be listed multiple times
            slitindex2=slitindex2[0]
        if len(slitindex)==0:
            print 'Ahhh, should not happpen',objnames3[slitindex3]
            continue
        else: #multiple observations means the same star can be listed multiple times
            slitindex=slitindex[0]
        print slitindex3,slitindex,slitindex2,objnames3[slitindex3],objnames[slitindex],objnames2[slitindex2]
        #plt.plot(wvls[slitindex],contdiv[slitindex])
        #plt.plot(wvls2[slitindex],contdiv2[slitindex])
        
        #cycle through five barium lines
        for i in range(1,6):
            mask = np.abs(wvls[slitindex]-xlines[i])<15
            min_index = np.argmin(np.abs(wvls2[slitindex2]-np.min(wvls[slitindex][mask])))
            mask2 = (min_index <= range(len(wvls2[slitindex2]))) & (min_index+np.sum(mask) > range(len(wvls2[slitindex2])))
            
            array = np.column_stack((wvls[slitindex][mask],contdiv[slitindex][mask],contdivstd[slitindex][mask]))
            array3 = np.column_stack((wvls3[slitindex3][mask],contdiv3[slitindex3][mask],contdivstd3[slitindex3][mask]))
            if i==1:
                array_full = np.copy(array)
            else:
                array_full = np.vstack((array_full,array))
            array2 = np.column_stack((wvls2[slitindex2][mask2],contdiv2[slitindex2][mask2],contdivstd2[slitindex2][mask2]))
            if i==1:
                array_full2 = np.copy(array2)
            else:
                array_full2 = np.vstack((array_full2,array2))
            if i==1:
                array_full3 = np.copy(array3)
            else:
                array_full3 = np.vstack((array_full3,array3))
        plt.errorbar(array_full[:,0],array_full[:,1],yerr=array_full[:,2],fmt='o')
        plt.errorbar(array_full2[:,0],array_full2[:,1],yerr=array_full2[:,2],fmt='o')
        plt.errorbar(array_full3[:,0],array_full3[:,1],yerr=array_full3[:,2],fmt='o')
        if slitindex3 >65:
            plt.show()   
                        
def plot_single_spectrum(filename,filename_red,slitindex,save_ascii=False):
    f = pyfits.open(filename)
    tbdata=f[1].data
    #print tbdata.dtype
    print tbdata.field('OBJNAME')[slitindex]
    #print tbdata.field('TEFF')[slitindex], tbdata.field('TEFFERR')[slitindex]
    #print tbdata.field('LOGG')[slitindex], tbdata.field('LOGGERR')[slitindex]
    #print tbdata.field('FEH')[slitindex], tbdata.field('FEHERR')[slitindex]
    #print tbdata.field('ALPHAFE')[slitindex], tbdata.field('ALPHAFEERR')[slitindex]
    print tbdata.field('SN')[slitindex]
    #print tbdata.field('VT')[slitindex], tbdata.field('VTERR')[slitindex]
    #print tbdata.field('VT')
    objnames = np.copy(tbdata.field('OBJNAME'))
    contdiv = np.copy(tbdata.field('CONTDIV')) #equals spectra/continuum
    contdivivar = np.copy(tbdata.field('CONTDIVIVAR'))
    contdivstd = np.sqrt(np.reciprocal(contdivivar))
    zrest = np.copy(tbdata.field('ZREST'))
    dlam = np.copy(tbdata.field('DLAM')*0.95) #resscale fudge factor
    teff = np.copy(tbdata.field('TEFF'))
    logg = np.copy(tbdata.field('LOGG'))
    feh = np.copy(tbdata.field('FEH'))
    alphafe = np.copy(tbdata.field('ALPHAFE'))
    wvls = np.copy((tbdata.field('LAMBDA').T/(1+zrest)).T)            
#filename = ['/raid/gduggan/moogify/bdra1_moogify.fits.gz']
#filename_red = ['/raid/m31/dsph/dra/dra1/moogify7_flexteff.fits.gz']
    
    # correct wavelength using three h lines (add 5170 line to improve fit - need to synthesize new spec?)
    #find wavelength solution from red spectrum
    f2 = pyfits.open(filename_red)
    print 'filename_red', filename_red
    tbdata2 = f2[1].data
    objname_red = np.copy(tbdata2.field('OBJNAME'))
    zrest_red = np.copy(tbdata2.field('ZREST'))
    ra_red = np.copy(tbdata2.field('RA'))
    dec_red = np.copy(tbdata2.field('DEC'))
    indices = [i for i, s in enumerate(objname_red) if objnames[slitindex] in s]
    #print 'red:', objname_red, objnames[slitindex]
    if len(indices) != 1:
        print "No unique red spectrum found (zero or more than one)"
        wvls_red = np.array([])
        #continue
    else:
        red_index = indices[0]
        zrest_red_slit = np.copy(zrest_red[red_index])
        wvls_red = np.copy((tbdata.field('LAMBDA').T/(1+zrest_red_slit)).T)[slitindex]
    f_data = scipy.interpolate.interp1d(wvls[slitindex], contdiv[slitindex]) 
    wvls_adj = ba_meas_functions.fit_wvl(filename, wvls[slitindex], contdiv[slitindex], zrest[slitindex], 
                       dlam[slitindex][0], teff[slitindex], logg[slitindex], feh[slitindex], 
                       alphafe[slitindex],f_data,name=filename.split('/')[-2]+'_%i'%slitindex,
                       name_fragment='', slitindex=slitindex, 
                       wvls_red=wvls_red, wvl_max_shift=2, debug=False,plot=False)

    plt.plot(wvls[slitindex],contdiv[slitindex])
    plt.plot(wvls_adj,contdiv[slitindex])
    plt.plot(h_lines,[1,1,1],'go')
    plt.show()
    
    #cycle through five barium lines and save wvl and flux to ascii
    if save_ascii == True:
        ascii_name = plot_path+filename.split('/')[-2]+'_%i.txt'%slitindex
        print ascii_name
        for i in range(1,6):
            mask = np.abs(wvls_adj-xlines[i])<15
            array = np.column_stack((wvls_adj[mask],contdiv[slitindex][mask]))
            if i==1:
                array_full = np.copy(array)
            else:
                array_full = np.vstack((array_full,array))
            print i, array.shape, array_full.shape
        np.savetxt(ascii_name,array_full,delimiter=' ',fmt='%8g', header = 'Wvl (ang)    Flux')
        plt.plot(array_full[:,0],array_full[:,1])
        plt.show()
        
def sn_hist_slitmask(filename,filename_red,name,plot=False):

    f = pyfits.open(filename)
    tbdata=f[1].data
    sn = tbdata.field('SN')
    objnames = np.copy(tbdata.field('OBJNAME')) #returns parameter for each slit
    f2 = pyfits.open(filename_red)
    tbdata2=f2[1].data
    sn2 = tbdata2.field('SN')    
    objname_red = np.copy(tbdata2.field('OBJNAME'))
    step = 1

    mask = []
    for slitindex in range(len(objnames)):
        #### create mask to see if it is a member
        indices = [i for i, s in enumerate(objname_red) if objnames[slitindex] in s]
        if len(indices) == 1:
            mask.append(True)
            print indices[0],objnames[slitindex],objname_red[indices[0]],sn[slitindex]
        else:
            mask.append(False)
    mask = np.array(mask)                    
    plt.figure()
    bins = np.arange(0,max(sn)+step,step)
    plt.hist(sn,bins,alpha=0.5)
    plt.hist(sn[mask],bins,alpha=0.5,label="Present in red slitmask")
    plt.xlabel('Signal to Noise')
    title = name+' blue: Mean=%g Median=%g Std=%g'%(np.mean(sn),np.median(sn),np.std(sn))
    print title
    plt.title(title)
    plt.legend()
    plt.savefig(plot_path+name+'_blue_red_sn_hist.png')
    if plot == True: plt.show()
    plt.close()

    plt.figure()
    plt.hist(sn)
    plt.xlabel('Signal to Noise')
    title = name+' blue: Mean=%g Median=%g Std=%g'%(np.mean(sn),np.median(sn),np.std(sn))
    print title
    plt.title(title)
    plt.savefig(plot_path+name+'_blue_sn_hist.png')
    if plot == True: plt.show()
    plt.close()
    
    plt.figure()
    plt.hist(sn2)
    plt.xlabel('Signal to Noise')
    title = name+' red: Mean=%g Median=%g Std=%g'%(np.mean(sn2),np.median(sn2),np.std(sn2))
    print title
    plt.title(title)
    plt.savefig(plot_path+name+'_red_sn_hist.png')
    if plot == True: plt.show()
    plt.close()
        
def sn_hist_all(plot=False):
    for dsph_name, name_list in [['Dra',['dra1','dra2','dra3']],['UMi',['umi1','umi2','umi3']],['Scl',['scl1','scl2','scl6']],['For',['for6']]]:
        for name in name_list:
            filename_red = '/raid/m31/dsph/%s/%s/moogify7_flexteff.fits.gz'%(name_list[0][:-1],name)
            filename = '/raid/caltech/moogify/b%s/moogify.fits.gz'%name
            sn_hist_slitmask(filename,filename_red,name,plot=plot)
    ufd_name,name_list = ['LeoT', ['LeoTa']]
    filename_red = '/raid/m31/udwarf/%s/moogify7_flexteff.fits.gz'%(ufd_name)
    filename = '/raid/caltech/moogify/%s/moogify.fits.gz'%name_list[0]    
    sn_hist_slitmask(filename,filename_red,name_list[0],plot=plot)

def simulate_data(name_fragment='',plot=True):
    ############ Used in M15 Keck proposal for 2017_B. Creates simulated data 
    # that matches 1200 blue grating specs for two main sequence stars: one that
    # matches the barium abundance measured a RGB stars and one that is enhanced. 
    #It showed that we will be able to detect if the main sequence star is enhanced.
    Teff,logg,feh,alphafe,bafe = 6450,4,-2.4, 0, 0.5
    bafe2 = 1.7
    delta = 5
    #dlam_sigma = 0.7276596
    dlam_sigma = 0.5 #spectral resolution, FWHM_lam = R*lam = 2.355*dlam_sigma, R=4200 at 5000ang for new grating
    dlam_pix = 0.33 #angstrom/pixel pixel scale for 1200 lines/mm grating
    #S_to_N_low = 25 # for lambda less than 5000
    S_to_N_high = 45 # for lambda more than 5500 
    S_to_N_low = 25 # for lambda less than 5000
    #S_to_N_high = 100 # for lambda more than 5500 
    S_to_N = [25,30,49,49,55]    
    
    wvl0, relflux0, title0, outfilename0 = moog_functions.interp_moog_ba(Teff,logg,feh,alphafe,bafe,dlam_sigma)
    wvl1, relflux1, title1, outfilename1 = moog_functions.interp_moog_ba(Teff,logg,feh,alphafe,bafe2,dlam_sigma,out_of_grid=True)
    data_wvl = np.concatenate(([np.arange(xstart[i],xstop[i],dlam_pix) for i in range(len(xlines))]))
    f_synth0 = scipy.interpolate.interp1d(wvl0, relflux0)
    f_synth1 = scipy.interpolate.interp1d(wvl1, relflux1)

    xminorLocator = plt. MultipleLocator (1)
    xmajorLocator = plt. MultipleLocator (3)

        # plot adjusted synthetic spectra with the wvl corrected data
    fig, axs = plt.subplots(1,5, figsize=(12,4))
    fig.subplots_adjust(bottom=0.10,top=0.91,hspace=.18, right=0.95, left=0.11,wspace=0)
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    #plt.setp([a.xaxis.set_major_locator(xmajorLocator) for a in fig.axes[:]])
    #plt.setp([a.xaxis.set_minor_locator(xminorLocator) for a in fig.axes[:]])
    plt.setp([a.xaxis.set_major_locator(MaxNLocator(nbins=3,prune=None)) for a in fig.axes[:]])

    #plt.suptitle(title0+'$\pm$ %.2f'%ba_error, y = 0.95)

    #ymajorLocator = plt. MultipleLocator (0.25)
    

    axs = axs.ravel()
        
    for i in np.arange(0,len(xstart)-1):
        data_flux0 = f_synth0(data_wvl)*(1+1./S_to_N[i]*np.random.normal(loc=0,scale=1,size=len(data_wvl)))
        #print np.mean(data_flux0)/np.std(data_flux0)

        data_flux1 = f_synth1(data_wvl)*(1+1./S_to_N[i]*np.random.normal(loc=0,scale=1,size=len(data_wvl)))
        print np.mean(data_flux1)/np.std(data_flux1)
        
        if i==0:
            axs[i].set_ylabel('Normalized Flux',labelpad=10)#,fontsize=14, labelpad=10)
        if i==2:
            axs[i].set_title(title0 + " and [Ba/Fe] = %g"%bafe2,y=1.02)
        #if i == 2:
        #    axs[i].set_xlabel('Wavelength ($\AA$)',labelpad=10)#,fontsize=14, labelpad=10)
        single_line_mask = (wvl0>=xlines[i+1]-delta) & (wvl0<=xlines[i+1]+delta)

        if i<2:
            #print np.std(data_flux0)
            #axs[i].plot(data_wvl,data_flux0_low,'o',color='darkgoldenrod')#,yerr=np.array([np.std(data_flux0_low)]*len(data_wvl)))
            axs[i].errorbar(data_wvl,data_flux1,yerr=np.array([np.std(data_flux1)]*len(data_wvl)),fmt='og')            
        else:
            #axs[i].plot(data_wvl,data_flux0_high,'o',color='darkgoldenrod')#,yerr=[np.std(data_flux0_high)]*len(data_wvl))
            axs[i].errorbar(data_wvl,data_flux1,yerr=np.array([np.std(data_flux1)]*len(data_wvl)),fmt='og')            
        axs[i].plot(wvl0[single_line_mask],relflux0[single_line_mask],'-',color="darkgoldenrod",linewidth=2)#relflux0/f_moog_cont(wvl0))
        #axs[i].errorbar(data_wvl_adapted[single_line_mask],data_flux_adapted[single_line_mask],yerr=contdivstd[slitindex][data_mask][single_line_mask],fmt='.b') #add errorbar 
        axs[i].plot(wvl1[single_line_mask],relflux1[single_line_mask],'-g',linewidth=2)#relflux0/f_moog_cont(wvl0))
        axs[i].set_xlim([xlines[i+1]-delta,xlines[i+1]+delta])
        #axs[i].set_ylim([min(data_flux_adapted)-0.1,max(data_flux_adapted)+0.1])
        axs[i].set_ylim([0.8,1.05])
        #axs[i].xaxis.set_minor_locator(xminorLocator)
        #axs[i].xaxis.set_major_locator(xmajorLocator)
    plt.savefig(plot_path+'simulated_data_ba_%d_%d'%(bafe,bafe2)+name_fragment+'.png')
    #print plot_path+outfilename+name_fragment+'.png'
    if plot == True:
        plt.show()
    plt.close()

def find_star(dsph_name, name_list, target_objname):
    filename_list = ['/raid/caltech/moogify/b%s/moogify.fits.gz'%name for name in name_list]
    for mask_index in range(len(filename_list)):
        filename = filename_list[mask_index]
        name = name_list[mask_index]
        # load entire slitmask
        f = pyfits.open(filename)
        tbdata=f[1].data
        objnames = np.copy(tbdata.field('OBJNAME')) #returns parameter for each slit
        #print objnames
        if target_objname in objnames:
            print 'SUCCESS! %s was found in mask: %s, slit:'%(target_objname,name),np.where(objnames==target_objname)
        else: 
            print target_objname, "was not found in mask:",name

def write_lit_paramaters_halo_stars(overwrite=False):
    filename = '/raid/gduggan/moogify/halo_900ZD_moogify.fits.gz'
    new_filename = '/raid/gduggan/moogify/lit_halo_900ZD_moogify.fits.gz'
    if (os.path.isfile(new_filename)==True) and (overwrite==False):
        print "overwrite=False and file exists.", new_filename, "remains unchanged."
        return new_filename
    print 'Setting parameter values of:',filename
    f = pyfits.open(filename)
    tbdata=f[1].data
    mask = np.where(tbdata['OBJNAME']!='slit0') #this moogify file has extra empty slit files where the object name is equal to 'slit0'
    # set errors from Fulbright et al. 2000
    #tbdata['VTERR'][:]=0.11
    comment = 'ful00# All from Fulbright et al. 2000 except BAFE/BAFEERR and VT/VTERR (calc from logg).'
    tbdata['LIT_COMMENT'][:]=comment
    print comment
    
    # load star-specific parameters from tables. Many of the stars don't match! How was fulbright200.for.python generated?
    table2='/raid/gduggan/gc/Fulbright2000/datafile2.txt'
    d2=pd.read_csv(table2,skiprows=18,names=['HIP','Obs','HD','BD/CD','Other','Vmag','Ex','S/N'])
    d2['BD/CD']=d2['BD/CD'].str.replace(' ','') #remove internal spaces from name
    d2['HIP']=d2['HIP'].str.replace(' ','') #remove internal spaces from name
    table5='/raid/gduggan/gc/Fulbright2000/datafile5.txt'
    d5=pd.read_csv(table5,skiprows=22,skipfooter=11,engine='python',names=['HIP','HD/BD','IntTeff','Intlogg','TEFF','LOGG','FEH','VT'])
    d5['HD/BD']=d5['HD/BD'].str.replace(' ','') #remove internal spaces from name
    d5['HIP']=d5['HIP'].str.replace(' ','') #remove internal spaces from name
    table6='/raid/gduggan/gc/Fulbright2000/datafile6.txt'
    d6=pd.read_csv(table6,skiprows=31,skipfooter=11,engine='python',skipinitialspace=True,names=['HIP','FEH','logLi','NAFE','MGFE','ALFE','SIFE','CAFE','TIFE','VFE','CRFE','NIFE','YFE','ZRFE','BAFE','EUFE'])
    d6['HIP']=d6['HIP'].str.replace(' ','') #remove internal spaces from name
    table1='/raid/gduggan/gc/Fulbright2000/Fulbright2002datafile1.txt'
    d1=pd.read_csv(table1,skiprows=50,engine='python',skipinitialspace=True,names=['HIP','RVel','r_RVel','U-LSR','V-LSR','W-LSR','RFVel','ROTVel','AngMom','RMin','RMax','Eccen','ZMax'])
    d1['HIP']=d1['HIP'].str.replace(' ','') #remove internal spaces from name
    for star_ind in mask[0]:
        #print star_ind
        obj = tbdata['OBJNAME'][star_ind].replace(' ','')
        if 'BD' in obj: #find corresponding HIP name
            d2_ind = np.where(d2['BD/CD']==obj)[0]
            if len(d2_ind)==0:
                print "BD did not match!", star_ind, obj, obj_short
                continue
            else: #multiple observations means the same star can be listed multiple times
                d2_ind=d2_ind[0]
                #print "BD matched to HIP:", star_ind, obj, d2['BD/CD'][d2_ind], d2_ind, 'HIP:',d2['HIP'][d2_ind]
                obj_hip = d2['HIP'][d2_ind]
        elif 'HIP' in obj: 
            obj_hip=obj[3:]
        #find HIP name in d5 and save Teff, logg, vt, and Fe/H
        d5_ind = np.where(d5['HIP']==obj_hip)[0]
        if len(d5_ind)==0:
            print "HIP not found in table 5", star_ind, obj, obj_hip
            continue
        elif len(d5_ind) ==1:
            d5_ind = d5_ind[0]
            #print "Table 5 MATCH:", star_ind, obj, obj_hip, d5['HIP'][d5_ind], d5_ind
            tbdata['TEFF'][star_ind] = d5['TEFF'][d5_ind]
            tbdata['TEFFERR'][star_ind]=40
            tbdata['LOGG'][star_ind] = d5['LOGG'][d5_ind]
            tbdata['LOGGERR'][star_ind]=0.06
            tbdata['VT'][star_ind]=2.13-0.23*d5['LOGG'][d5_ind] #from equation described in Kirby et al. 2009
            tbdata['VTERR'][star_ind]=np.sqrt(0.05**2+(d5['LOGG'][d5_ind]*0.03)**2+(0.23*0.06)**2) #from equation with loggerr = 0.06   
            tbdata['FEH'][star_ind] = d5['FEH'][d5_ind]
            tbdata['FEHERR'][star_ind]=0.079    
            
            tbdata['TEFF_LIT'][star_ind] = d5['TEFF'][d5_ind]
            tbdata['TEFFERR_LIT'][star_ind]=40
            tbdata['LOGG_LIT'][star_ind] = d5['LOGG'][d5_ind]
            tbdata['LOGGERR_LIT'][star_ind]=0.06
            tbdata['VT_LIT'][star_ind]=d5['VT'][d5_ind]
            tbdata['VTERR_LIT'][star_ind]=0.11 #km/s
            tbdata['FEH_LIT'][star_ind] = d5['FEH'][d5_ind]
            tbdata['FEHERR_LIT'][star_ind]=0.079                      
        else:
            print "HIP had too many matches in table 5", star_ind, obj, obj_hip, d5['HIP'][d5_ind], d5_ind    ### convert to structured array, match BD or HIP name to the correct row, check its the same one for data6, and set parameters
            continue
        #find HIP name in d6 and save alphafe and bafe_lit
        d6_ind = np.where(d6['HIP']==obj_hip)[0]
        if len(d6_ind)==0:
            print "HIP not found in table 6", star_ind, obj, obj_hip
            continue
        elif len(d6_ind) ==1:
            d6_ind = d6_ind[0]
            #print "Table 6 MATCH:", star_ind, obj, obj_hip, d6['HIP'][d6_ind], d6_ind
            #check [Fe/H] is the same in both tables. It is different. Use atmosphere value in table 5.
            #if tbdata['FEH'][star_ind] != d6['FEH'][d6_ind]:
            #    print "ERROR: [Fe/H] is different in table 5 and 6.",tbdata['FEH'][star_ind],d6['FEH'][d6_ind]
            alphafe_array = np.array([d6['MGFE'][d6_ind],d6['SIFE'][d6_ind],d6['CAFE'][d6_ind],d6['TIFE'][d6_ind]])
            alphafe=np.nanmean(alphafe_array)
            #print 'alphafe',alphafe, d6['MGFE'][d6_ind],d6['SIFE'][d6_ind],d6['CAFE'][d6_ind],d6['TIFE'][d6_ind]
            tbdata['ALPHAFE'][star_ind] = alphafe
            tbdata['ALPHAFEERR'][star_ind]=0.119
            tbdata['ALPHAFE_LIT'][star_ind] = alphafe
            tbdata['ALPHAFEERR_LIT'][star_ind]=0.119
            tbdata['BAFE_LIT'][star_ind]=d6['BAFE'][d6_ind]
            tbdata['BAFEERR_LIT'][star_ind]=0.117
        else:
            print "HIP had too many matches in table 1", star_ind, obj, obj_hip, d6['HIP'][d6_ind], d6_ind    ### convert to structured array, match BD or HIP name to the correct row, check its the same one for data6, and set parameters
            continue
        #find HIP name in d1 and save radial velocity as z_obs
        d1_ind = np.where(d1['HIP']==obj_hip)[0]
        if len(d1_ind)==0:
            print "HIP not found in table 1", star_ind, obj, obj_hip
            continue
        elif len(d1_ind) ==1:
            d1_ind = d1_ind[0]
            c_kms = 299792. #km/s
            RV = d1['RVel'][d1_ind]
            z_RV = ((1+RV/c_kms)/(1-RV/c_kms))**0.5 - 1.0
            tbdata['ZREST'][star_ind]=z_RV
        else:
            print "HIP had too many matches in table 1", star_ind, obj, obj_hip, d6['HIP'][d6_ind], d6_ind    ### convert to structured array, match BD or HIP name to the correct row, check its the same one for data6, and set parameters
            continue
        
        #science.zobs = moogifyred[w].zobs
        #science.zobsmc = moogifyred[w].zobsmc
        #science.zobserr = 0.0
        #science.vr = moogifyred[w].vr - (helio_deimos(moogifyred[w].ra, moogifyred[w].dec, 2000.0, jd=moogifyred[w].jdobs) + helio_deimos(science.ra, science.dec, 2000.0, jd=science.jdobs))
        #science.vrerr = moogifyred[w].vrerr
        #science.vhelio = moogifyred[w].vhelio
        
    print "Saving updated file to:", new_filename
    pyfits.writeto(new_filename,tbdata[mask],overwrite=True)
    return new_filename
    #to check in command line: star_ind=5, string='TEFF'
    #tbdata3['OBJNAME'][star_ind],tbdata3[string][star_ind],tbdata3[string+'_LIT'][star_ind],tbdata3[string+'ERR'][star_ind],tbdata3[string+'ERR_LIT'][star_ind]    

def load_ba_results(dsph_name):
    ba_results=np.genfromtxt('/raid/gduggan/analysis_code/barium_results_table.txt',names=True,dtype=None,delimiter=',')
    ba_results = ba_results[ba_results['DSPH']==dsph_name]
    print np.unique(ba_results['DSPH'])
    ba_results['FEHERR'] = np.sqrt(ba_results['FEHERR']**2+fehsyserr**2)
    ba_results['ALPHAFEERR'] = np.sqrt(ba_results['ALPHAFEERR']**2+alphasyserr**2)
    ba_results['BAFEERR'] = np.sqrt(ba_results['BAFEERR']**2+bafesyserr**2)
    ba_results['MGFEERR'] = np.sqrt(ba_results['MGFEERR']**2+mgfesyserr**2)
    mask_alpha_err = (ba_results['ALPHAFEERR']<=accuracy_cutoff)&(ba_results['ALPHAFEERR']>alphasyserr)& \
                     (ba_results['FEHERR']<=accuracy_cutoff)&(ba_results['FEHERR']>fehsyserr)
    ba_results_clean = ba_results[mask_alpha_err]
    mask = (ba_results_clean['BAFEERR']<accuracy_cutoff)&(ba_results_clean['BAFEERR']>bafesyserr)
    ba_final = ba_results_clean['BAFEERR'][mask]
    return ba_results_clean, mask, ba_final   

def plot_moving_av(element_Z=56,ylim=True,average=True,data=True,name_fragment='',plot=True):
#def plot_compare_element_mult(num_models_to_compare = 2, name_fragment ='',title = [''],
#                              r_process_keyword = ['typical_SN_only'], IMF = ['Kroupa93'], 
#                              AGB_yield_fudge = [1],AGB_source=['kar16'],SNII_yield_fudge =[1],
#                              NSM_yield_fudge =[1] ,element_Z = 56, data=True, 
#                              plot=True, ylim=True,model=True,Keck_proposal=False):

    labels = ['Fornax', 'Sculptor', 'Ursa Minor', 'Draco']
    dsph = ['For', 'Scl', 'UMi', 'Dra']
    
    if element_Z == 56:
        label = '[Ba/Fe]'
        data_element = 'BAFE'
        data_err = 'BAFEERR'
        dot_size = 0
    elif element_Z == 12:
        label = '[Mg/Fe]'
        data_element = 'MGFE'
        data_err = 'MGFEERR'
        dot_size = 0
    else: print "No label specified"
    filename = name_fragment + label
          
    rc('axes', labelsize=14) #24
    #rc("axes", linewidth=2)
    rc('xtick', labelsize=12)
    rc('ytick', labelsize=12)
    rc('xtick.major', size=12)
    rc('ytick.major', size=12)
    #rc('xtick.minor', size=7)
    #rc('ytick.minor', size=7)
    # plot Ba abundances vs Fe
    fig, axs = plt.subplots(4, figsize=(5,11),sharex=True)#, sharey=True)
    fig.subplots_adjust(bottom=0.06,top=0.97, left = 0.2,wspace=0.29,hspace=0)
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    axs = axs.ravel()
    #axs[0].set_title(title1 + ' vs. ' +'\textcolor{red}{%s}'%title2)

    for i in np.arange(4):

        if data==True:        
            if i==0: filename = filename + '_data'
            ba_results, mask, ba_final = load_ba_results(dsph[i])

            if element_Z == 12:
                mask = (ba_results['MGFEERR']<=accuracy_cutoff)&(ba_results['MGFEERR']>mgfesyserr)&(ba_results['BAFEERR']<accuracy_cutoff)&(ba_results['BAFEERR']>bafesyserr)
                print "don't be zero",np.min(ba_results['MGFEERR'][mask]), np.min(ba_results['MGFE'][mask])
            #axs[i].errorbar(ba_results['FeH'][mask][maskdetect],ba_results['BaFe'][mask][maskdetect],fmt='o',xerr=ba_results['Feerr'][mask][maskdetect],yerr=ba_results['Baerr'][mask][maskdetect],color='0.6',markeredgewidth=0.5,markeredgecolor='k')#,capsize=3))
            err = np.reciprocal(ba_results['FEHERR'][mask]**2+ba_results[data_err][mask]**2)
            axs[i].scatter(ba_results['FEH'][mask],ba_results[data_element][mask],
                           marker='o',c='k',s=err*1.3+dot_size, alpha=0.7,linewidth=0)#,capsize=3))

        if average==True:
            if i==0: filename = filename + '_average'            
            step_size = 0.6
            begin = np.around(np.nanmin(ba_results['FEH'][mask]),1)-0.1
            end = np.around(np.nanmax(ba_results['FEH'][mask]),1)
            results = []
            for x_value in np.arange(begin,end+0.1,0.1):
                mask_bin = (ba_results['FEH'][mask]>x_value-step_size/2.0)&(ba_results['FEH'][mask]<x_value+step_size/2.0)
                if sum(mask_bin)>=3:
                    #print x_value, ba_results['FEH'][mask][mask_bin],ba_results[data_element][mask][mask_bin]
                    #weighted average
                    weight = np.reciprocal(ba_results[data_err][mask][mask_bin]**2)
                    y_result = np.sum(ba_results[data_element][mask][mask_bin]*weight)/np.sum(weight)
                    y_result_err = np.reciprocal(np.sqrt(np.sum(weight)))
                    results.append([x_value,y_result,y_result_err])
            results = np.array(results)
            axs[i].errorbar(results[:,0],results[:,1],yerr = results[:,2],
                            fmt='.-',label="Weighted Average")

        axs[0].set_title(filename)
        axs[i].set_ylabel(labels[i]+' (%i)'%np.sum(mask)+'\n'+label)
        axs[i].plot([-6,0],[0,0],':k')
        print label,'range',np.nanmin(ba_results[data_element][mask]),np.nanmax(ba_results[data_element][mask])  
        print '[Fe/H] range',np.nanmin(ba_results['FEH'][mask]),np.nanmax(ba_results['FEH'][mask])  
        if ylim==True:
            y_errorbar_loc = -1
            axs[i].set_ylim([-2.1,1.6])
        else:
            min_data = np.nanmin(ba_results[data_element][mask])
            y_errorbar_loc = min_data+0.3 
        x_errorbar_loc = -0.8
        x_errorbar_err = np.nanmean(ba_results['FEHERR'][mask])
        y_errorbar_err = np.nanmean(ba_results[data_err][mask])
        axs[i].errorbar([x_errorbar_loc], [y_errorbar_loc],color='k',xerr=[x_errorbar_err],yerr=[y_errorbar_err])
        print 'average error x and y:', x_errorbar_err, y_errorbar_err
    plt.xlim([-3.1,-0.5])
    plt.xlabel('[Fe/H]')
    filename = (((((filename.replace('[','')).replace('/','')).replace(']','')).replace("\\",'')).replace('$','')).replace(' ','')
    plt.savefig(plot_path+filename+'.eps')
    if plot==True: plt.show()
    
def plot_ba_results_w_lit(element_Z=56,plot_lit=True,ylim=False,errorbars=False,
                          average=True,data=True,name_fragment='',plot=True):
    # element_Z can equal 12 (Mg), 56 (Ba), or 63 (Eu). Barium can be plotted 
    # with or without our measurements and the literature measurements. Mg will 
    # be plotted from Kirby et al. 2010. If Eu is selected, [Ba/Eu] will be 
    # plotted for the literature. All literature measurements plotted exclude 
    # upper/lower limits.

    labels = ['Fornax', 'Sculptor', 'Draco','Sextans', 'Ursa Minor']
    dsph = ['For', 'Scl', 'Dra','Sex', 'UMi']
        
    if element_Z == 56:
        label = '[Ba/Fe]'
        data_element = 'BAFE'
        data_err = 'BAFEERR'
        data_feh = 'FEH'
        data_feh_err = 'FEHERR'
        dot_size = 0.4
    elif element_Z == 12:
        label = '[Mg/Fe]'
        data_element = 'MgFe'
        data_err = 'Mgerr'
        data_feh = 'FeH'
        data_feh_err = 'Feerr'
        dot_size = 0.4
        plot_lit=False
    elif element_Z == 63:
        label = '[Ba/Eu]'
        dot_size=0
        data=False
        plot_lit=True
    else: print "No label specified"
    filename = name_fragment + label
    if errorbars == True:
        filename = filename + '_errorbars'
        
    x_dim = 5
    r_space=0.9
    #load literature values      
    if plot_lit==True: 
        filename_w_errors = '/raid/gduggan/dsph/sagadatabase_asplund2009_fixederr.tsv'
        #filename_w_radec = '/raid/gduggan/dsph/sagadatabase_asplund2009_radec.tsv'
        data_w_errors = np.genfromtxt(filename_w_errors,delimiter='\t',dtype=None,comments='#',names=True)
        #data_w_errors.dtype = [('Object', 'S15'), ('Reference', 'S33'), ('Teff', '<i8'), 
        #    ('log_g', '<f8'), ('FeH1', '<f8'), ('FeH', 'S11'), ('BaFe', 'S11')]

        objname = data_w_errors['Object']
        REF = np.array([str.split(',')[0]+str.split(',')[-1] for str in data_w_errors['Reference']])
        HRS_FEH_str = data_w_errors['FeH']
        HRS_FEH = np.array([float(str.split('+-')[0]) for str in HRS_FEH_str])
        HRS_FEH_err = np.array([float(str.split('+-')[1]) for str in HRS_FEH_str])
        HRS_BAFE_str = data_w_errors['BaFe']
        HRS_BAFE = np.array([float(str.split('+-')[0]) for str in HRS_BAFE_str])
        HRS_BAFE_err = np.array([float(str.split('+-')[1]) for str in HRS_BAFE_str])  
        
        # load Tsujimoto et al. 2017 results for Draco. Tsujimoto et al. 2015 are remeasured. Asplund et al 2009 solar def.
        #filename_tsuj17_radec = '/raid/gduggan/dsph/Tsu17_tab1.txt'
        #data_radec = np.genfromtxt(filename_tsuj17_radec,skip_header=4,names=True,skip_footer=6,delimiter='\t',dtype=None,comments='#')
        #data_radec.type = [('Object_Name', 'S23'), ('RA', 'S11'), ('Decl', 'S11'), 
        #('Exposure', 'S4'), ('SNa', '<i8'), ('Vb', '<f8'), ('K_sc', '<f8'), 
        #('EBV_d', '<f8'), ('v_mathrmhel', 'S9'), ('f0', '?')]
        #objname_tsuj17 = data_radec['Object_Name']
        filename_tsuj17_data = '/raid/gduggan/dsph/Tsu17_tab2.txt'
        data_tsuj17 = np.genfromtxt(filename_tsuj17_data,skip_header=4,names=True,delimiter='\t',dtype=None,comments='#')
        # data_tsuj17.dtype = [('Object_Name', 'S11'), ('T_mathrmeff', '<i8'), 
        #('mathrmlogg', '<f8'), ('v_t', '<f8'), ('mathrmFermH', 'S15'), 
        #('rmYrmH', 'S15'), ('mathrmBarmH', 'S15'), ('mathrmEurmH', 'S15'), 
        #('BaEu', 'S15'), ('f0', '?')]
        objname_tsuj17 = np.array([str+'Dra' for str in data_tsuj17['Object_Name']])
        REF_tsuj17 = ['T.Tsujimoto+ 2017']*len(objname_tsuj17)
        HRS_FEH_str_tsuj17 = data_tsuj17['mathrmFermH']
        HRS_FEH_tsuj17 = np.array([float(str.split(' +or- ')[0]) for str in HRS_FEH_str_tsuj17])
        HRS_FEH_err_tsuj17 = np.array([float(str.split(' +or- ')[1]) for str in HRS_FEH_str_tsuj17])
        HRS_BAH_str_tsuj17 = data_tsuj17['mathrmBarmH']
        HRS_BAH_tsuj17 = np.array([float(str.split(' +or- ')[0]) for str in HRS_BAH_str_tsuj17])
        HRS_BAH_err_tsuj17 = np.array([float(str.split(' +or- ')[1]) for str in HRS_BAH_str_tsuj17])
        HRS_BAFE_tsuj17 = HRS_BAH_tsuj17-HRS_FEH_tsuj17
        HRS_BAFE_err_tsuj17 = np.sqrt(HRS_BAH_err_tsuj17**2+HRS_FEH_err_tsuj17**2)
        
        objname = np.concatenate((objname,objname_tsuj17))
        REF = np.concatenate((REF,REF_tsuj17))
        HRS_FEH = np.concatenate((HRS_FEH,HRS_FEH_tsuj17))
        HRS_FEH_err = np.concatenate((HRS_FEH_err,HRS_FEH_err_tsuj17))
        HRS_BAFE = np.concatenate((HRS_BAFE,HRS_BAFE_tsuj17))
        HRS_BAFE_err = np.concatenate((HRS_BAFE_err,HRS_BAFE_err_tsuj17))
        
        #### convert solar definition from Asplund et al. 2009 to our def (Anders & Grevesse 1989 with (Fe/H)=7.52)
        HRS_FEH = HRS_FEH + 7.50 - 7.52
        HRS_BAFE = HRS_BAFE + 2.18 - 2.13 - (7.50 - 7.52)
        
        if element_Z == 63: 
            filename_eufe = '/raid/gduggan/dsph/sagadatabase_eufe_asplund2009_fixederr.tsv'
            data_eufe = np.genfromtxt(filename_eufe,delimiter='\t',dtype=None,comments='#',names=True)
            #data_eufe.dtype = [('Object', 'S15'), ('Reference', 'S35'), 
            #('RA', 'S11'), ('Decl', 'S12'), ('Teff', '<i8'), ('log_g', '<f8'), 
            #('FeH', '<f8'), ('FeH_1', 'S11'), ('EuFe', 'S11')]

            objname_eu = data_eufe['Object']
            REF_eu = np.array([str.split(',')[0]+str.split(',')[-1] for str in data_eufe['Reference']])
            HRS_FEH_eu_str = data_eufe['FeH_1']
            HRS_FEH_eu = np.array([float(str.split('+-')[0]) for str in HRS_FEH_eu_str])
            HRS_FEH_eu_err = np.array([float(str.split('+-')[1]) for str in HRS_FEH_eu_str])
            HRS_EUFE_str = data_eufe['EuFe']
            HRS_EUFE = np.array([float(str.split('+-')[0]) for str in HRS_EUFE_str])
            HRS_EUFE_err = np.array([float(str.split('+-')[1]) for str in HRS_EUFE_str])

            # find [Ba/Fe] measurement for that star, mention if it's from a different source
            HRS_BAEU =  np.array([np.nan]*len(objname_eu))
            HRS_BAEU_err =  np.array([np.nan]*len(objname_eu))
            for eu_index in range(len(objname_eu)):
                ba_index = np.where(objname_eu[eu_index]==objname)[0]
                if len(ba_index)==1:
                    HRS_BAEU[eu_index] = HRS_BAFE[ba_index]-HRS_EUFE[eu_index]
                    HRS_BAEU_err[eu_index] = np.sqrt(HRS_BAFE_err[ba_index]**2+HRS_EUFE_err[eu_index]**2)
                    if REF[ba_index] != REF_eu[eu_index]:
                        print 'Ah!', objname_eu[eu_index], ba_index, objname[ba_index], REF_eu[eu_index], REF[ba_index]
                #else:
                    #print objname_eu[eu_index], ba_index, objname[ba_index], REF_eu[eu_index]      

            # Add Tsuj17 observations
            HRS_BAEU_str_tsuj17 = data_tsuj17['BaEu']
            baeu_mask = [' +or- ' in str for str in HRS_BAEU_str_tsuj17]
            HRS_BAEU_tsuj17 =  np.array([np.nan]*len(HRS_BAEU_str_tsuj17))
            HRS_BAEU_err_tsuj17 =  np.array([np.nan]*len(HRS_BAEU_str_tsuj17))
            HRS_BAEU_tsuj17[baeu_mask] = np.array([float(str.split(' +or- ')[0]) for str in HRS_BAEU_str_tsuj17[baeu_mask]])
            HRS_BAEU_err_tsuj17[baeu_mask] = np.array([float(str.split(' +or- ')[1]) for str in HRS_BAEU_str_tsuj17[baeu_mask]])

            objname_eu = np.concatenate((objname_eu,objname_tsuj17))
            REF_eu = np.concatenate((REF_eu,REF_tsuj17))
            HRS_FEH_eu = np.concatenate((HRS_FEH_eu,HRS_FEH_tsuj17))
            HRS_FEH_eu_err = np.concatenate((HRS_FEH_eu_err,HRS_FEH_err_tsuj17))
            HRS_BAEU = np.concatenate((HRS_BAEU,HRS_BAEU_tsuj17))
            HRS_BAEU_err = np.concatenate((HRS_BAEU_err,HRS_BAEU_err_tsuj17))

            #### convert solar definition from Asplund et al. 2009 to our def (Anders & Grevesse 1989 with (Fe/H)=7.52)
            #HRS_EUFE = HRS_EUFE + 0.52 - 0.51 - (7.50 - 7.52)
            HRS_BAEU = HRS_BAEU + 2.18 - 2.13 - (0.52 - 0.51)
            
        x_dim = 8   
        r_space=0.64
        print '[Fe/H] literature range',np.nanmin(HRS_FEH),np.nanmax(HRS_FEH)  

    
    rc('axes', labelsize=14) #24
    #rc("axes", linewidth=2)
    rc('xtick', labelsize=12)
    rc('ytick', labelsize=12)
    rc('xtick.major', size=12)
    rc('ytick.major', size=12)
    #rc('xtick.minor', size=7)
    #rc('ytick.minor', size=7)
    # plot Ba abundances vs Fe
    fig, axs = plt.subplots(5, figsize=(x_dim,11),sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.06,top=0.97, left = 0.2,wspace=0.29,hspace=0,right=r_space)
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    axs = axs.ravel()
    #axs[0].set_title(title1 + ' vs. ' +'\textcolor{red}{%s}'%title2)

    marker_shapes = ['v','s','D','>','*']
    for i in np.arange(5):

        if plot_lit==True:
            if i==0: filename = filename + '_lit'
            if element_Z == 56:
                dsph_mask = [dsph[i] in name for name in objname]
                REFS = np.unique(REF[dsph_mask])
                for j,ref in enumerate(REFS):
                    if 'Letarte' in ref:
                        continue
                    ref_mask = (REF[dsph_mask]==ref)
                    #plt.errorbar(HRS[mask],MRS[mask],fmt=fmt,
                    #     xerr=HRS_err[mask],yerr=MRS_err[mask],
                    #     label=ref) 
                    #err = np.reciprocal(HRS_FEH_err[dsph_mask][ref_mask]**2+HRS_BAFE_err[dsph_mask][ref_mask]**2)
                    axs[i].scatter(HRS_FEH[dsph_mask][ref_mask],HRS_BAFE[dsph_mask][ref_mask],
                                   marker=marker_shapes[j],s=20,c='b', alpha=0.4,linewidth=0,label=ref)#,s=err*0.9+dot_size,label=ref)#,capsize=3))    
                    #
                    #axs[i].scatter([np.min(HRS_FEH)-1],[0],marker=marker_shapes[j],
                    #               c='orange',alpha=0.7,linewidth=0,label=ref)#,capsize=3))    
         
                    #axs[i].legend(bbox_to_anchor=(0.98, 1))
                    axs[i].legend(bbox_to_anchor=(0.98, 0.57))
            if element_Z == 63:
                dsph_mask = [dsph[i] in objname_eu[k] and 'Letarte' not in REF_eu[k] for k in range(len(objname_eu))]
                REFS = np.unique(REF_eu[dsph_mask])
                for j,ref in enumerate(REFS):
                    if 'Letarte' in ref:
                        continue
                    ref_mask = (REF_eu[dsph_mask]==ref)
                    #plt.errorbar(HRS[mask],MRS[mask],fmt=fmt,
                    #     xerr=HRS_err[mask],yerr=MRS_err[mask],
                    #     label=ref) 
                    err = np.reciprocal(HRS_FEH_eu_err[dsph_mask][ref_mask]**2+HRS_BAEU_err[dsph_mask][ref_mask]**2)
                    axs[i].scatter(HRS_FEH_eu[dsph_mask][ref_mask],HRS_BAEU[dsph_mask][ref_mask],
                                   marker=marker_shapes[j],c='grey', alpha=0.7,linewidth=0)#,s=err*0.9+dot_size,label=ref)#,capsize=3))    
                    axs[i].scatter([np.min(HRS_FEH_eu)-1],[0],marker=marker_shapes[j],
                                   c='grey',alpha=0.7,linewidth=0,label=ref)#,capsize=3))    
         
                    #axs[i].legend(bbox_to_anchor=(0.98, 1))
                    axs[i].legend(bbox_to_anchor=(0.98, 0.57))
                print dsph[i], label, "median=",np.nanmedian(HRS_BAEU[dsph_mask]),"mean=",np.nanmean(HRS_BAEU[dsph_mask])
                mask = dsph_mask
                x_errorbar_err = 0#np.nanmean(HRS_FEH_eu_err[dsph_mask])
                y_errorbar_err = 0#np.nanmean(HRS_BAEU_err[dsph_mask])
        if data==True:        
            if i==0: filename = filename + '_data'
            if element_Z == 56:
                results, mask, ba_final = load_ba_results(dsph[i])
            elif element_Z == 12:
                # load Kirby 2010 Mg measurements
                filename_member = '/raid/m31/dsph/alldsph/dsph_catalog.dat'
                members = np.genfromtxt(filename_member, skip_header=1,
                                        dtype=[('dSph', 'S9'), ('objname', 'S11'), ('RAh', '<i8'), ('RAm', '<i8'), ('RAs', '<f8'), ('Decd', '<i8'), 
                                               ('Decm', '<i8'), ('Decs', '<f8'), ('v_raw', '<f8'), ('Teff', '<f8'), ('logg', '<f8'), ('vt', '<f8'), 
                                               ('FeH', '<f8'), ('Feerr', '<f8'), ('AlphaFe', '<f8'), ('Alphaerr', '<f8'), ('MgFe', '<f8'), ('Mgerr', '<f8'),
                                               ('SiFe', '<f8'), ('Sierr', '<f8'), ('CaFe', '<f8'), ('Caerr', '<f8'), ('TiFe', '<f8'), ('Tierr', '<f8')],
                                        delimiter=(9,11,3,3,6,3,3,6,7,5,5,6,6,5,6,5,6,5,6,5,6,5,6,5))
                members['dSph']=np.core.defchararray.strip(members['dSph'])
                members['objname']=np.core.defchararray.strip(members['objname'])
                #members = np.genfromtxt(filename_member, skip_header=1,delimiter=(9,11,3,3,6,3,3,6,7,5,5,6,6,5,6,5,6,5,6,5,6,5,6,5), names=True, dtype=None) #Use to get dtype list
                # AlphaFe and Alphaerr are not present in all objects, so can't read them in. 
                # Need to consider replacing spaces with negative value so error isn't thrown.
                mask_dsph = (dsph[i]==members['dSph'])
                results = np.copy(members[mask_dsph])
                results['Feerr'] = np.sqrt(results['Feerr']**2+fehsyserr**2)
                results['Alphaerr'] = np.sqrt(results['Alphaerr']**2+alphasyserr**2)
                results['Mgerr'] = np.sqrt(results['Mgerr']**2+mgfesyserr**2)
                    
                mask = (results['Mgerr']<=accuracy_cutoff)&(results['Mgerr']>mgfesyserr
                        )&(results['Alphaerr']<=accuracy_cutoff)&(results['Alphaerr']>alphasyserr
                        )&(results['Feerr']<=accuracy_cutoff)&(results['Feerr']>fehsyserr)
            print label,'range',np.nanmin(results[data_element][mask]),np.nanmax(results[data_element][mask])  
            print '[Fe/H] range',np.nanmin(results[data_feh][mask]),np.nanmax(results[data_feh][mask])  
            weight = np.reciprocal(results[data_err][mask]**2)
            weighted_av = np.sum(results[data_element][mask]*weight)/np.sum(weight)
            weighted_av_err = np.reciprocal(np.sqrt(np.sum(weight)))        
            print 'Weighted Average for',label,weighted_av,'+-',weighted_av_err,len(results[data_element][mask])
            print 'Average for',label,np.mean(results[data_element][mask]),'+-',np.mean(results[data_err][mask]),len(results[data_element][mask])
            x_errorbar_err = np.nanmean(results[data_feh_err][mask])
            y_errorbar_err = np.nanmean(results[data_err][mask])
            print 'average error x and y:', x_errorbar_err, y_errorbar_err

            if errorbars == True:
                #axs[i].errorbar(results[data_feh][mask],results[data_element][mask],
                #                   marker='o',c='k',xerr=results[data_feh_err][mask],
                #                   yerr=results[data_err][mask], alpha=0.7,linewidth=0)#,capsize=3)) 
                axs[i].errorbar(results[data_feh][mask],results[data_element][mask],
                         fmt='.k',xerr=results[data_feh_err][mask],
                         yerr=results[data_err][mask],ecolor='0.6')

            else:
                err = np.reciprocal(results[data_feh_err][mask]**2+results[data_err][mask]**2)
                axs[i].scatter(results[data_feh][mask],results[data_element][mask],
                                   marker='o',c='k',s=err*1.3+dot_size, alpha=0.7,linewidth=0)#,capsize=3))   
                #print 'use this size', np.nanmedian(err*1.3+dot_size) 
                y_errorbar_loc = -1
                #    axs[i].set_ylim([-2.1,1.6])
                #else:
                #    min_data = np.nanmin(results[data_element][mask])
                #    y_errorbar_loc = min_data+0.3 
                x_errorbar_loc = -0.8
        
                axs[i].errorbar([x_errorbar_loc], [y_errorbar_loc],color='k',xerr=[x_errorbar_err],yerr=[y_errorbar_err])
        else:
            mask=[]
        #axs[0].set_title(filename)
        axs[i].set_ylabel(labels[i]+' (%i)'%np.sum(mask)+'\n'+label)
        axs[i].plot([-6,0],[0,0],':k')
    if ylim==True:
        plt.ylim([-2.3,1.7])

    if plot_lit == True:
        if element_Z == 56:
            plt.xlim([np.min(HRS_FEH)-0.2,np.max(HRS_FEH)+0.2])
        elif element_Z == 63:
            plt.xlim([np.min(HRS_FEH_eu)-0.2,np.max(HRS_FEH_eu)+0.2])
    else:
        plt.xlim([-3.1,-0.5])
    plt.xlabel('[Fe/H]')
    filename = (((((filename.replace('[','')).replace('/','')).replace(']','')).replace("\\",'')).replace('$','')).replace(' ','')
    plt.savefig(plot_path+filename+'.png')
    print "Plot saved here:",filename
    if plot==True: plt.show()
