import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.interpolate
import scipy.optimize
import moog_functions 
import astropy.io.fits as pyfits
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
from boto.beanstalk.exception import simple
from scipy.signal import find_peaks_cwt

# search 'CHECK' to find hard-coded decisions

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
rc('legend', fontsize=15) #16
rc('xtick.major', pad=8)
rc('ytick.major', pad=8)
rc('xtick',direction='in')
rc('ytick',direction='in')
    
#c_cgs = 2.998e10 #cm/s

xstart=np.array([4120.6, 4544.0, 4924, 5843, 6131.7, 6486.9])
xstop = np.array([4140.6, 4564.0, 4944, 5863, 6151.7, 6506.9])
xlines = np.array([4130.6, 4554.0, 4934.1, 5853.7, 6141.7, 6496.9])
wvl_radius=10

#data_path = '/raid/gduggan/gridba/'
plot_path = '/raid/gduggan/analysis_code/plots/ba_fit/'
test_data_path = '/raid/gduggan/analysis_code/test_results/'
ha_data_path = '/raid/grid7/'
hb_data_path = '/raid/gduggan/gridhbeta/'
hy_data_path = '/raid/gridch/'
    
#hteffarr=[3500, 4000, 4500, 5000, 5500, 6000, 7000] 
# Even though synthetic spectra was created for temperatures lower than 4000K, 
# these lower temperatures resulted in very weak Hbeta, which messed up the 
# fit between the synthetic and observed spectra.
hteffarr=[4500, 5000, 5500, 6000]
hloggarr=[0.5, 1.0, 1.5, 2.0, 2.5]
hfeharr=[-2.0, -1.0]
halphaarr=[0.0]

h_lines = [4341, 4861, 6563]

def reduced_chi_square(f_obs,f_exp,f_err,dof=1):
    chi_sqr = np.sum(np.square((f_obs-f_exp)/f_err))
    reduced_chi_sqr = chi_sqr/(len(f_obs)-dof)
    return chi_sqr, reduced_chi_sqr
        
def return_hydrogen_synth(teff,logg,feh,alphafe,dlam,wvl_radius):
    #find and return closest hydrogen synth spec, truncating the returned spectra to h_lines +/- wvl_radius in angstroms
    hteff, ind = moog_functions.find_nearest(hteffarr,teff)
    hlogg, ind = moog_functions.find_nearest(hloggarr,logg)
    hfeh, ind = moog_functions.find_nearest(hfeharr,feh)
    halpha, ind = moog_functions.find_nearest(halphaarr,alphafe)
    subtitle = r'Synth: T$_{eff}$=%i, log(g)=%.2f, [Fe/H]=%.2f, [$\alpha$/Fe]=%.2f'%(hteff, hlogg, hfeh, halpha)
    wvly, relfluxy, title0, outfilename0 = moog_functions.read_moog(hteff, hlogg, hfeh, halpha,data_path_name=hy_data_path,lambda_sigma=dlam)
    #### CH synthetic spectra is stacked to include all C abundances in one file - want C/Fe = 0.0: 
    #cfearr = [-2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2,  0.4, 0.6,  0.8,  1.0,  1.4,  1.8, 2.2,  2.6,  3.0,  3.5]
    #print cfearr[12]
    n = len(relfluxy)/24
    relfluxy = relfluxy[n*12:n*13]
#    wvlb, relfluxb, title0, outfilename0 = read_moog(hteff, hlogg, hfeh, halpha,data_path_name=hb_data_path,lambda_sigma=dlam)
    wvlb, relfluxb, title0, outfilename0 = moog_functions.read_moog(hteff, hlogg, hfeh, halpha,data_path_name=hb_data_path,lambda_sigma=dlam,start_wvl=4851,stop_wvl=4871)
    wvla, relfluxa, title0, outfilename0 = moog_functions.read_moog(hteff, hlogg, hfeh, halpha,data_path_name=ha_data_path,lambda_sigma=dlam)
#    print "hgamma", wvly[0], wvly[-1]
#    print "hbeta", wvlb[0], wvlb[-1]
#    print "halpha", wvla[0], wvla[-1]
    masky = (wvly > h_lines[0]-wvl_radius) & (wvly < h_lines[0] + wvl_radius)
    maskb = (wvlb > h_lines[1]-wvl_radius) & (wvlb < h_lines[1] + wvl_radius)
    maska = (wvla > h_lines[2]-wvl_radius) & (wvla < h_lines[2] + wvl_radius)
    wvlh = np.array([wvly[masky],wvlb[maskb],wvla[maska]])
#    relfluxb = 1-(1-relfluxb)/2.
    relfluxh = np.array([relfluxy[masky],relfluxb[maskb],relfluxa[maska]])
    ############# DANGER DANGER DANGER - hard coded reducing flux from hbeta synthesis
    return wvlh, relfluxh, subtitle

def h_continuum(data_wvl,data_flux,data_flux_std,spec_wvl,spec_flux,title,f_data,debug=False,plot=False):

    line_data_std = []
    wvl_begin_gap = data_wvl[4089]
    wvl_end_gap = data_wvl[4101]
    #check that spectrum isn't awful in these regions and covers the h lines
    for i in range(len(h_lines)):
        data_mask = (data_wvl>h_lines[i]-2*wvl_radius) & (data_wvl<h_lines[i]+2*wvl_radius)
        line_data_std.append(np.median(data_flux_std[data_mask]))
        if ((len(data_flux[data_mask]) != sum(np.isfinite(data_flux[data_mask]))) or 
            (h_lines[i]-2*wvl_radius<data_wvl[0]) or (h_lines[i]+2*wvl_radius>data_wvl[-1]) or 
            ((wvl_end_gap>h_lines[i]+2*wvl_radius>wvl_begin_gap) or (wvl_end_gap>h_lines[i]-2*wvl_radius>wvl_begin_gap))):
            return [],[],[]
       
    divided = np.array([f_data(spec_wvl[i])/spec_flux[i] for i in range(len(h_lines))])
    fit_param = np.array([np.polyfit(spec_wvl[i],divided[i],0) for i in range(len(h_lines))])
    f_synth_adj = []
    h_continuum_array = []
    for i in range(len(h_lines)):
        continuum = np.poly1d(fit_param[i])
        h_continuum_array.append(continuum(spec_wvl[i]))
        f_synth_adj.append(spec_flux[i]*continuum(spec_wvl[i]))
    f_synth_adj = np.array(f_synth_adj)  
    h_continuum_array = np.array(h_continuum_array)
    
    if debug==True and plot==True:
        fig, axs = plt.subplots(1,3, figsize=(11,5))
        fig.subplots_adjust(bottom=0.17,wspace=0.29)
        #plt.suptitle(subtitle, y = 0.955)
        #xminorLocator = plt. MultipleLocator (1)
        #xmajorLocator = plt. MultipleLocator (10)
        #ymajorLocator = plt. MultipleLocator (0.25)
    
        axs = axs.ravel()
        for i in range(len(h_lines)):
            if (i==0):
                axs[i].set_ylabel('Normalized Flux')#,fontsize=14, labelpad=10)
            if i==1:
                axs[i].set_title(title,y=1.03)
                axs[i].set_xlabel('Wavelength ($\AA$)',labelpad=10)#,fontsize=14, labelpad=10)
            xmajorLocator = plt. MultipleLocator (10)
            axs[i].xaxis.set_major_locator(xmajorLocator)    
            axs[i].plot(spec_wvl[i],divided[i])
            axs[i].plot(spec_wvl[i],f_synth_adj[i],'r')
            p = np.poly1d(fit_param[i])
            axs[i].plot(spec_wvl[i],p(spec_wvl[i]))
        plt.show()    
        plt.close()
    return f_synth_adj, h_continuum_array, np.array(line_data_std) # spec_wvl[i]*h_continuum_array[i] = f_synth_adj[i]

def data_continuum(data_wvl,data_flux,data_flux_err,spec_wvl,spec_flux,title,debug=False,plot=False): 
            
    num = int(len(spec_wvl)/6.0)
    spec_wvl_array = np.array([spec_wvl[i*num:(i+1)*num] for i in range(len(xlines))])
    # select barium lines used in the barium abundance measurement
    #check that spectrum isn't awful in these regions
    wvl_begin_gap = data_wvl[4089]
    wvl_end_gap = data_wvl[4101]
    covered_lines = []
    data_mask_array=[]
    spec_mask_array=[]
    for i in range(len(xlines)):
        good=False
        spec_mask_line = (spec_wvl>=min(spec_wvl_array[i])) & (spec_wvl<=max(spec_wvl_array[i]))
        data_mask_line = (data_wvl>min(spec_wvl_array[i])) & (data_wvl<max(spec_wvl_array[i]))
        if i>0: ########## skip 4130 barium line for all spectra!!! CHECK to see if this is something you want
            if (spec_wvl_array[i][0]>data_wvl[0]) & (spec_wvl_array[i][-1]<data_wvl[-1]):
                if ((spec_wvl_array[i][0]>wvl_end_gap) or (spec_wvl_array[i][-1]<wvl_begin_gap)):
                    if (len(data_flux[data_mask_line]) == sum(np.isfinite(data_flux[data_mask_line]))) & (len(data_flux[data_mask_line]) == sum(data_flux[data_mask_line]>0.0)):
                        good=True
        covered_lines.append(good)
        if good==True:
            data_mask_array.append(data_mask_line)
            spec_mask_array.append(spec_mask_line)
    covered_lines = np.array(covered_lines)
    if sum(covered_lines)<4:
        return [],[],[],[],[]
    data_mask_array = np.array(data_mask_array)
    spec_mask_array = np.array(spec_mask_array)
    data_mask = np.sum(data_mask_array,axis=0,dtype=bool)
    spec_mask = np.sum(spec_mask_array,axis=0,dtype=bool) 
    # create an array of masks to form disjoint data sets to match synthetic spectra range
    data_wvl_array = [data_wvl[data_mask_array[i]] for i in np.arange(sum(covered_lines))] #lists because they aren't all the same size
    data_wvl_adapted = data_wvl[data_mask]
    data_flux_adapted = data_flux[data_mask]
    spec_flux_adapted = spec_flux[spec_mask]
    spec_wvl_adapted=spec_wvl[spec_mask]
    f_synth = scipy.interpolate.interp1d(spec_wvl_adapted, spec_flux_adapted)
    spec_flux_interp = f_synth(data_wvl_adapted)
    
    divided = np.array([data_flux[data_mask_array[i]]/f_synth(data_wvl_array[i]) for i in range(sum(covered_lines))])
    divided_1d = data_flux_adapted/spec_flux_interp
    # do not use 1 angstrom on either side of a barium line with finding the continuum
    mask = np.array([np.absolute(data_wvl_array[i]-xlines[i])>1.0 for i in range(sum(covered_lines))])

    fit_param = np.array([np.polyfit(data_wvl_array[i][mask[i]],divided[i][mask[i]],1,
                                     w=np.reciprocal(data_flux_err[data_mask_array[i]][mask[i]]/f_synth(data_wvl_array[i][mask[i]]))) 
                          for i in range(sum(covered_lines))])
    ba_continuum = np.array([])
    for i in range(sum(covered_lines)):
        continuum = np.poly1d(fit_param[i]) #include errors
        ba_continuum = np.concatenate((ba_continuum,continuum(data_wvl_array[i])))
    f_synth_adj = ba_continuum*spec_flux_interp
    #print data_wvl_adapted.shape, data_flux_adapted.shape, spec_flux_interp.shape,divided_1d.shape,ba_continuum.shape,f_synth_adj.shape 
    
    if debug==True and plot==True:
        fig, axs = plt.subplots(2,3, figsize=(11,8.5))
        fig.subplots_adjust(bottom=0.17,wspace=0.29)
        #plt.suptitle(subtitle, y = 0.955)
        #xminorLocator = plt. MultipleLocator (1)
        #xmajorLocator = plt. MultipleLocator (10)
        #ymajorLocator = plt. MultipleLocator (0.25)
    
        axs = axs.ravel()
        for i in np.arange(len(xlines))[covered_lines]:
            if (i==0 or i==3):
                axs[i].set_ylabel('Normalized Flux')#,fontsize=14, labelpad=10)
            if i==1:
                axs[i].set_title(title,y=1.03)
            if i==5:
                axs[i].set_xlabel('Wavelength ($\AA$)',labelpad=10)#,fontsize=14, labelpad=10)
            xmajorLocator = plt. MultipleLocator (10)
            axs[i].xaxis.set_major_locator(xmajorLocator) 
            axs[i].errorbar(data_wvl_adapted,data_flux_adapted,yerr=data_flux_err[data_mask],fmt='o') #add errorbar 
            #axs[i].plot(data_wvl_adapted,spec_flux_interp) 
            axs[i].plot(data_wvl_adapted,divided_1d,'k')
            axs[i].plot(data_wvl_adapted,f_synth_adj,'r')
            j= sum(covered_lines[:i+1])-1
            p = np.poly1d(fit_param[j])
            axs[i].plot(data_wvl_array[j],p(data_wvl_array[j]),'k')
            axs[i].set_xlim([xstart[i],xstop[i]])
        plt.show() 
        plt.close()
    return f_synth_adj,ba_continuum, covered_lines, data_mask,spec_mask# spec_wvl_array, f_synth_adj_array  
    
def find_wvl_offset(wvlh,wvl_radius,f_data,f_synth_adj,title,debug=False,plot=False):
    # convolve data with synthetic spectra to find wavelength offset
    synth_wvls = []
    data_interp_wvls = []
    wvl_data_conv = []
    conv_array = []
    if debug==True:
        print "Result of convolution: wvl difference, synthetic, data"
    for i in range(len(h_lines)):
        # create wvl grid for the large data wvl interval that matches synthetic spectra -> wvl_data
        wvl_step = np.around(wvlh[i][1]-wvlh[i][0],2)
        wvl_data = np.arange(h_lines[i]-1.5*wvl_radius, h_lines[i]+1.5*wvl_radius, wvl_step)
        f_data(wvl_data)
        # convolve it. Need to move spectra so the absorption is positive and is centered at 0
        conv = np.correlate((1-f_data(wvl_data)),(1-f_synth_adj[i]),'same')
        true_synth_wvl = np.median(wvlh[i])#wvlh[i][np.argmin(f_synth_adj[i])]
        # if many corrections are within 10% of the best match, pick the closest one
        conv_normalized = conv/np.max(conv)
        peakind = find_peaks_cwt(conv,np.array([120]))
        peak_mask = conv[peakind]/np.max(conv[peakind])>0.9
        #print 'peaks!!',peakind[peak_mask], conv[peakind[peak_mask]]/np.max(conv[peakind]),wvl_data[peakind[peak_mask]]
        data_interp_wvl = wvl_data[peakind[peak_mask]][np.argmin(np.abs(wvl_data[peakind[peak_mask]]-true_synth_wvl))]
        #print 'selected wvl!!!',data_interp_wvl
        #data_interp_wvl = wvl_data[np.argmax(conv)]
        diff = true_synth_wvl - data_interp_wvl
        if debug==True:
            print "H line %i: "%i, diff, true_synth_wvl, data_interp_wvl
        synth_wvls.append(true_synth_wvl)
        data_interp_wvls.append(data_interp_wvl)
        wvl_data_conv.append(wvl_data)
        conv_array.append(conv)
    synth_wvls = np.array(synth_wvls)
    data_interp_wvls = np.array(data_interp_wvls)
    wvl_data_conv = np.array(wvl_data_conv)
    conv_array = np.array(conv_array)
    
    if debug==True and plot==True:
        fig, axs = plt.subplots(1,3, figsize=(11,5))
        fig.subplots_adjust(bottom=0.17,wspace=0.29)
        axs = axs.ravel()
        for i in range(len(h_lines)):
            if (i==0):
                axs[i].set_ylabel('Normalized Flux')#,fontsize=14, labelpad=10)
            if i==1:
                axs[i].set_title(title,y=1.03)
                axs[i].set_xlabel('Wavelength ($\AA$)',labelpad=10)#,fontsize=14, labelpad=10)
            xmajorLocator = plt. MultipleLocator (25)
            axs[i].xaxis.set_major_locator(xmajorLocator)    
            axs[i].plot(wvlh[i],(1-f_synth_adj[i]),'b-',label="continuum corrected")
            axs[i].plot(wvl_data_conv[i],(1-f_data(wvl_data)),'k-',label="data")
            axs[i].plot(wvl_data_conv[i],conv_array[i]/max(conv_array[i]),'m',label="convolution results")
            axs[i].plot([synth_wvls[i]]*2,[0,1],'g:',label="synthetic h-line wvl")
            axs[i].plot([data_interp_wvls[i]]*2,[0,1],'m:',label="uncorrected h-line wvl")
            axs[i].set_xlim([h_lines[i]-2*wvl_radius,h_lines[i]+2*wvl_radius])
        plt.show()    
        plt.close()
    return synth_wvls, data_interp_wvls, wvl_data_conv, conv_array
        
def fit_wvl(filename, wvls_slit, data_flux_orig, data_flux_std, dlam, teff, logg, feh, 
            alphafe, f_data,name,name_fragment="",slitindex = 0, 
            wvl_max_shift = 2, simple_line_correction=True,debug = False,plot=False, 
            carbon_zrest=False):
    # goal: fine-tune wavelength by matching to Halpha, beta, and gamma
    # simple_line_correction = False means...
    # The Hgamma correction is set equal to 0 
    # (because this wavelength was matched using a larger wavelength range when 
    # Evan measured carbon). Then a line is fit to the blue chip (Hgamma and Hbeta).
    # A flat offset is calculated for the red chip.
    # simple_line_correction = True means...
    # a simple line should be fit based on Halpha, beta and gamma.
        
    wvl_begin_gap = wvls_slit[4089]
    wvl_end_gap = wvls_slit[4101]
    mask_red_data = wvls_slit>wvl_end_gap

    #check that all three hydrogen lines are included in the data
    h_lines_blue_chip = (h_lines>wvls_slit[0]) & (h_lines<wvl_begin_gap)
    h_lines_red_chip = (h_lines<wvls_slit[-1]) & (h_lines>wvl_end_gap)
    if np.sum(h_lines_blue_chip)+np.sum(h_lines_red_chip) != 3:
        print h_lines_blue_chip,h_lines_red_chip, "Halpha, beta, or gamma is missing. Wavelength is not fine-tuned."
        return wvls_slit

    f_data = scipy.interpolate.interp1d(wvls_slit, data_flux_orig) 
                
#    # for a given slit in the slitmask, set up slit info
#    if good[slitindex] != 1.0:
#        return np.array([])    
    title1 = r'T$_{eff}$=%i, log(g)=%.2f, [Fe/H]=%.2f, [$\alpha$/Fe]=%.2f'%(teff, logg, feh, alphafe)
    if debug==True:
        print title1
    #name = (filename.split('/')[-1]).split('_')[0]
    outfilename = name+'_%i_'%slitindex+'wvlfit'
    title = name+'_%i '%slitindex+' '+title1    


    # find corresponding hydrogen line synthetic spectra
    wvlh, relfluxh, subtitle = return_hydrogen_synth(teff,logg,feh,alphafe,dlam,wvl_radius)

    # find continuum 
    f_synth_adj, h_continuum_array, h_data_std = h_continuum(wvls_slit,data_flux_orig,
                                                    data_flux_std,wvlh,relfluxh,
                                                    title,f_data,debug,plot) # wvlh[i]*h_continuum_array[i] = f_synth_adj[i]
    print 'h_data_std',h_data_std
    if len(f_synth_adj) == 0:
        print "Skipping %s slitindex %d - non-finite flux or incomplete coverage near H lines"%(name,slitindex)
        return []
        
    # convolve data with synthetic spectra to find wavelength offset
    synth_wvls, data_interp_wvls, wvl_data_conv, conv_array = find_wvl_offset(wvlh,
                                        wvl_radius,f_data,f_synth_adj,title,debug,plot)
    diff_wvls = synth_wvls - data_interp_wvls
              
    hgamma_data_interp = data_interp_wvls[0]
    if carbon_zrest == True:           
        # reset hgamma to use default wvl (original wvl was the result of a 
        # much larger convolution when measured carbon abundance)
        if debug==True:
            print "Setting Hgamma synthesis difference at 0"
        diff_wvls[0] = 0
        data_interp_wvls[0]=synth_wvls[0] 
    
    if np.max(np.abs(diff_wvls))>wvl_max_shift:
        print "Wavelength shift larger than %d ang requested - fail"%wvl_max_shift
        return []

    # test out fits   
    if debug == True: 
        wvl_old_to_new_interp= scipy.interpolate.interp1d(data_interp_wvls,synth_wvls)
        param2 = np.polyfit(data_interp_wvls,synth_wvls,2,w=np.reciprocal(h_data_std))
        wvl_old_to_new2 = np.poly1d(param2) 

    if simple_line_correction == False:
        ## only fit blue side
        h_line_blue_mask = synth_wvls<wvl_end_gap
        param = np.polyfit(data_interp_wvls[h_line_blue_mask],synth_wvls[h_line_blue_mask],1,w=np.reciprocal(h_data_std[h_line_blue_mask]))
        wvl_old_to_new = np.poly1d(param)       
    else:
        ## fit all three balmer lines
        param = np.polyfit(data_interp_wvls,synth_wvls,1,w=np.reciprocal(h_data_std))
        wvl_old_to_new = np.poly1d(param) 
                
    # based on the three corrected h-lines, create a new wavelength solution 
    interp_mask = (wvls_slit>min(data_interp_wvls)) & (wvls_slit<max(data_interp_wvls)) 
    wvl_slit_new = wvl_old_to_new(wvls_slit)
    
    if simple_line_correction == False:
        ## will apply an offset to the red chip
        wvl_slit_new[mask_red_data] = wvls_slit[mask_red_data]+diff_wvls[-1]
    
    fig, axs = plt.subplots(2,3, figsize=(11,8.5))
    fig.subplots_adjust(bottom=0.10,top=0.92,hspace=.18, right=0.95, left=0.11)
    plt.suptitle(subtitle, y = 0.955)
    ymajorLocator = plt. MultipleLocator (0.25)
    axs = axs.ravel()   
    axs[1].set_title(title, y=1.1)
    axs[4].set_xlabel('Wavelength ($\AA$)',labelpad=10)#,fontsize=14, labelpad=10) 
    axs[0].set_ylabel('Normalized Flux')#,fontsize=14, labelpad=10)
    axs[3].set_ylabel('Normalized Flux')#,fontsize=14, labelpad=10)

    for i in range(len(h_lines)):
        axs[i].plot(wvls_slit,data_flux_orig,'k.', label="data")
#        print "Wvl range of synth: ", wvlh[i][0],wvlh[i][-1]
        axs[i].plot(wvlh[i],relfluxh[i],'g-',label="un-adjusted synth")
        axs[i].plot(wvlh[i],f_synth_adj[i],'b',label="continuum corrected")       
        axs[i].plot(wvlh[i]-diff_wvls[i],f_synth_adj[i],'r',label="wavelength corrected")
    
        axs[i+3].plot(wvlh[i],f_synth_adj[i],'b-',label="continuum corrected")
        axs[i+3].plot(wvl_data_conv[i],f_data(wvl_data_conv[i]),'k-',label="data")
        axs[i+3].plot(wvl_data_conv[i],conv_array[i]/max(conv_array[i]),'m',label="convolution results")
        axs[i+3].plot([synth_wvls[i]]*2,[0,1],'g:',label="synthetic h-line wvl")
        if i == 0:
            axs[i+3].plot([hgamma_data_interp]*2,[0,1],'m:',label="uncorrected h-line wvl")
        else:    
            axs[i+3].plot([data_interp_wvls[i]]*2,[0,1],'m:',label="uncorrected h-line wvl")
   
        axs[i].set_ylim([0,1.2])
        axs[i+3].set_ylim([0,1.2])
        axs[i+3].set_xlim([h_lines[i]-2*wvl_radius,h_lines[i]+2*wvl_radius])
        axs[i].set_xlim([h_lines[i]-wvl_radius,h_lines[i]+wvl_radius])
        #xmajorLocator = plt. MultipleLocator (10)
        #axs[i].xaxis.set_major_locator(xmajorLocator)
        xmajorLocator = plt. MultipleLocator (10)
        axs[i+3].xaxis.set_major_locator(xmajorLocator)
    #plt.legend()
    plt.savefig(plot_path+outfilename+name_fragment+'.png')
    if debug==True and plot == True:
        plt.show(fig)
    plt.close(fig)

    # plot new wavelength solution results    
    fig2, axs2 = plt.subplots(1,1, figsize=(11,6.0))
    #fig2.subplots_adjust(bottom=0.15,top=0.88,hspace=.21, right=0.93, left=0.17, wspace=0.3)
    plt.title(title,y=1.07)
    plt.suptitle(subtitle, y = 0.93)
    #axs2 = axs2.ravel()
   
    #axs2[0].plot(data_interp_wvls,data_interp_wvls-synth_wvls,'x',label='Synth')
    #if debug == True:
    #    axs2[0].plot(wvls_slit,wvls_slit-wvl_old_to_new(wvls_slit),':')#, label='1d')
    #    axs2[0].plot(wvls_slit,wvls_slit-wvl_old_to_new2(wvls_slit),'--')#, label='2d')
    #    axs2[0].plot(wvls_slit[interp_mask],
    #                 wvls_slit[interp_mask]-wvl_old_to_new_interp(wvls_slit[interp_mask]),'-.')#, label='Interp')
    #axs2[0].plot(wvls_slit,wvls_slit-wvl_slit_new,'-', label='final')
    #axs2[0].plot(np.log10([hgamma_data_interp]), np.log10([hgamma_data_interp])-synth_wvls[0],'rx')
    #axs2[0].plot(np.log10(xlines),np.array([0,0,0,0,0,0]), 'ro',label='Ba lines')  
    #axs2[0].plot(np.log10(np.array([4000,7000])),np.array([0,0]),'--')
    #axs2[0].set_xlabel('Log Current Data Wavelength')
    #axs2[0].set_ylabel('Log Difference (Current Data - X)')    
#    plt.legend(loc=4)
    
    axs2.errorbar(data_interp_wvls,data_interp_wvls-synth_wvls,yerr=h_data_std,fmt='x',label='Synth')
    axs2.plot(wvls_slit,wvls_slit-wvl_slit_new,'-', label='final')
    axs2.plot([hgamma_data_interp], [hgamma_data_interp]-synth_wvls[0],'rx')
    axs2.plot(xlines,[0,0,0,0,0,0], 'ro',label='Ba lines')  
    axs2.plot([4000,7000],[0,0],'--')
    xmajorLocator = plt. MultipleLocator (1000)
    axs2.xaxis.set_major_locator(xmajorLocator)
#    axs2[1].set_ylim(bottom=-0.1)
    axs2.set_xlabel('Current Data Wavelength')
    axs2.set_ylabel('Difference (Current Data - X)')

    plt.savefig(plot_path+outfilename+'2'+name_fragment+'.png') 
    if debug==True and plot == True:
        plt.show(fig2) 
    plt.close(fig2)
    
    return wvl_slit_new

def barium_fitting_routine(filename,objnames,slitindex,
                           tbdata,wvls,contdiv,teff,logg,feh,alphafe,dlam,debug,
                           plot,contdivstd,name,name_fragment,wvl_max_shift,
                           grid=True,test='',simple_line_correction=True,carbon_zrest=False):
    # if grid==False, then use individually calculated synthetic spectra to 
    # measure barium abundances. test='scat' uses moog17scat and test='iso_r' and 'iso_s'
    # uses moog14 with a different isotope ratio.
    
    f_data = scipy.interpolate.interp1d(wvls[slitindex], contdiv[slitindex]) 
    
    bafe = 0.0
    
    #### create spectrum that matches data
    wvl0, relflux0, title0, outfilename0=moog_functions.interp_moog_ba(teff[slitindex], 
                                            logg[slitindex],feh[slitindex], 
                                            alphafe[slitindex],bafe,
                                            dlam[slitindex][0],debug=False,
                                            grid=grid,test=test)
    #print teff[slitindex], logg[slitindex], feh[slitindex], alphafe[slitindex], 'good=',good[slitindex]
    title1 = r"T$_{eff}$=%i, log(g)=%.2f, [Fe/H]=%.2f, [$\alpha$/Fe]=%.2f, [Ba/Fe]=%.2f"%(teff[slitindex], logg[slitindex], feh[slitindex], alphafe[slitindex],0.0)
    title = name+'_%i %s '%(slitindex,objnames[slitindex])
    print title
        
    if debug == True and plot == True:
        plt.errorbar(wvls[slitindex],contdiv[slitindex],contdivstd[slitindex])
        plt.plot(wvl0,relflux0)
        plt.plot(wvls[slitindex][4089],[1],'ro')
        plt.plot(wvls[slitindex][4101],[1],'ro')
        plt.plot(h_lines,[1,1,1],'go')
        plt.ylim([0,1.5])
        plt.show()
        plt.close()
    
    if wvl_max_shift == 0:
        wvls_adj = wvls[slitindex]
        if slitindex==0:
            print "Wavelength is **NOT** being finetuned by balmer lines."
    else:
        # correct wavelength using three h lines (add 5170 line to improve fit - need to synthesize new spec?)
        wvls_adj = fit_wvl(filename, wvls[slitindex], contdiv[slitindex], contdivstd[slitindex], 
                           dlam[slitindex][0], teff[slitindex], logg[slitindex], feh[slitindex], 
                           alphafe[slitindex],f_data,name,name_fragment, slitindex, 
                           wvl_max_shift=wvl_max_shift, 
                           simple_line_correction=simple_line_correction, debug=debug,plot=plot,
                           carbon_zrest=carbon_zrest)
        if len(wvls_adj) == 0:
            print "Skipping %s slitindex %d - wvl correction failed"%(name,slitindex)
            return 0.0, -99, []
        
    # adjust synthetic spectra flux to match the continuum of the data
    f_synth_adj,ba_continuum, covered_lines,data_mask,spec_mask = data_continuum(wvls_adj,
                                    contdiv[slitindex],contdivstd[slitindex],
                                    wvl0,relflux0,title, debug, plot)
    if len(f_synth_adj)==0:
        print "Skipping %s slitindex %d - less than 4 out of 6 barium lines are valid"%(name,slitindex)
        return 0.0, -99, []
    wvl0_adapted = wvl0[spec_mask]

    data_wvl_adapted = wvls_adj[data_mask]
    data_flux_adapted = np.copy(contdiv)[slitindex][data_mask]
    # fit barium abundance
    def find_ba(wvl,ba_fit):
        wvl0, relflux0, title0, outfilename0=moog_functions.interp_moog_ba(teff[slitindex], 
                                                logg[slitindex], feh[slitindex], 
                                                alphafe[slitindex],ba_fit,
                                                dlam[slitindex][0],debug=False,
                                                grid=grid,test=test)
        #num = len(wvl0)/len(xlines) 
        #relflux_array = np.array([relflux0[i*num:(i+1)*num] for i in np.arange(len(xlines))[covered_lines]])
        #relflux0 = relflux_array.reshape(-1)
        relflux0 = relflux0[spec_mask]
        f_synth = scipy.interpolate.interp1d(wvl, relflux0) 
        return f_synth(data_wvl_adapted)*ba_continuum
    popt,pcov = scipy.optimize.curve_fit(find_ba, wvl0_adapted, data_flux_adapted, 
                                         sigma = contdivstd[slitindex][data_mask],p0=[0.0])
    ba_result = popt
    ba_error = np.sqrt(np.diag(pcov))
    #if np.around(ba_error,3)==0.0:
    #    return 0.0, -99

    wvl0, relflux0, title0, outfilename0=moog_functions.interp_moog_ba(teff[slitindex], 
                                            logg[slitindex], feh[slitindex], 
                                            alphafe[slitindex],ba_result,
                                            dlam[slitindex][0],debug=False,
                                            grid=grid,test=test)
    relflux0 = relflux0[spec_mask]
    f_synth = scipy.interpolate.interp1d(wvl0_adapted, relflux0) 
    f_synth_final= f_synth(data_wvl_adapted)*ba_continuum
    #num = len(wvl0)/len(xlines) 
    #relflux_array = np.array([relflux0[i*num:(i+1)*num] for i in np.arange(len(xlines))[covered_lines]])
    #relflux0 = relflux_array.reshape(-1)
    #f_synth_final = relflux0*ba_continuum 
    outfilename = name+'_%i_'%slitindex+outfilename0
    print outfilename
        
    def chi_square_plot(ba_result,ba_error,name,slitindex,name_fragment,wvl,f_obs,f_err,debug=False,plot=False):
            
        ba_list = np.array([-3,-2,-1.5,-1,-0.5,-0.1,0,0.1,0.5,1,1.5,2,3])*ba_error + ba_result
        #ba_mask = (-2<ba_list) & (ba_list<1) #upper and lower limits
        #ba_list = ba_list[ba_mask]
        chi_sqr_list = []
        for k in range(len(ba_list)):
            chi_sqr, reduced_chi_sqr = reduced_chi_square(f_obs, f_exp=find_ba(wvl,ba_list[k]), f_err=f_err, dof=1)
            chi_sqr_list.append(chi_sqr)
        chi_sqr_list = np.array(chi_sqr_list)
        
        if debug == True and plot == True:
            fig, axs = plt.subplots(6,1, figsize=(8.5,11))
            fig.subplots_adjust(bottom=0.07,top=0.93,hspace=.35, right=0.95)
            plt.title(title, y=7.96)
            plt.xlabel(r'Wavelength ($\AA$)',labelpad=10)#,fontsize=14, labelpad=10)
            xminorLocator = plt. MultipleLocator (1)
            ymajorLocator = plt. MultipleLocator (0.25)
            axs = axs.ravel()
            for k in range(len(ba_list)):
                for i in range(len(xstart)):
                    if i == 3:
                        axs[i].set_ylabel('Normalized Flux',labelpad=10)#,fontsize=14, labelpad=10)
                    axs[i].plot(data_wvl_adapted,find_ba(wvl,ba_list[k]))
                    #axs[i].set_xlim([xlines[i]-0.3,xlines[i]+0.3])
                    axs[i].set_xlim([xstart[i],xstop[i]])
                    axs[i].set_ylim([0,1.1])
                    axs[i].xaxis.set_minor_locator(xminorLocator)
            plt.show()

        min_chi_sqr_arg = np.argmin(chi_sqr_list)
        min_chi_sqr = chi_sqr_list[min_chi_sqr_arg]
        print min_chi_sqr+1, chi_sqr_list[0],chi_sqr_list[min_chi_sqr_arg+1]
        min_ba = ba_list[min_chi_sqr_arg]
        #plt.plot(function(chi_sqr_list[0:min_chi_sqr_arg+1]),chi_sqr_list[0:min_chi_sqr_arg+1])    
        if min_chi_sqr+1 > chi_sqr_list[0]:
            lower_ba = ba_list[0]
            print "Setting lower_ba to smallest [Ba/Fe] plotted"
        else:
            function_low = scipy.interpolate.interp1d(chi_sqr_list[0:min_chi_sqr_arg+1],ba_list[0:min_chi_sqr_arg+1])
            lower_ba = function_low(min_chi_sqr+1)
        if min_chi_sqr+1 > chi_sqr_list[-1]:
            print "Setting higher_ba to highest [Ba/Fe] plotted"
            higher_ba = ba_list[-1]
        else:
            function_high = scipy.interpolate.interp1d(chi_sqr_list[min_chi_sqr_arg:],ba_list[min_chi_sqr_arg:])
            higher_ba = function_high(min_chi_sqr+1) 
        #plt.plot(function(chi_sqr_list[min_chi_sqr_arg:]),chi_sqr_list[min_chi_sqr_arg:])    
        string = " [Ba/Fe] = $%.2g^{+%.2g}_{-%.2g}$"%(min_ba, higher_ba-min_ba,min_ba-lower_ba)   

        print "Chi-square results:",string
        red_chi_sqr = chi_sqr_list/(len(f_obs)-1)
        chi_sqr_eq_1 = (min(chi_sqr_list)+1)/(len(f_obs)-1)
    
        return ba_list, red_chi_sqr, chi_sqr_eq_1,string
    
    plot_chi_sq = False
    if ba_error<0.3 and np.around(ba_error,3) != 0.0:    
        # plot chi-square of barium fit to evaluate fitting more carefully
        try:
            plot_chi_sq = True
            ba_list, red_chi_sqr, chi_sqr_eq_1,chi_string = chi_square_plot(ba_result,ba_error,
                                                                name,slitindex,name_fragment,
                                                                wvl=wvl0_adapted,f_obs=data_flux_adapted,
                                                                f_err=contdivstd[slitindex][data_mask],
                                                                debug=debug,plot=plot)
        except():
            print "Chi square plot failed: ",name, slitindex, ValueError

    # plot adjusted synthetic spectra with the wvl corrected data
    fig, axs = plt.subplots(2,3, figsize=(11,8.5))
    fig.subplots_adjust(bottom=0.10,top=0.91,hspace=.21, right=0.97, left=0.1,wspace=0.4)
    #plt.suptitle(title0+'$\pm$ %.2f'%ba_error, y = 0.95)
    xminorLocator = plt. MultipleLocator (1)
    ymajorLocator = plt. MultipleLocator (0.25)
    

    axs = axs.ravel()
    plot_index = np.copy(covered_lines)
    plot_index[0] = True
    for i in ([0]+np.arange(len(xstart))[plot_index]):
        if (plot_chi_sq == True) and (i==0):
            #ba_mask = (-2<ba_list) & (ba_list<1) #upper and lower limits
            axs[i].plot(ba_list, red_chi_sqr,'bo-', linewidth=2)
            axs[i].set_title(chi_string,y=1)
            axs[i].plot(ba_list,[chi_sqr_eq_1]*len(ba_list),'g--',linewidth=2)
            axs[i].set_xlabel('[Ba/Fe]')
            axs[i].set_ylabel('Reduced $\chi^2$')
        if (i==1) or (i==3):
            axs[i].set_ylabel('Normalized Flux')#,fontsize=14, labelpad=10)
        if i==1:
            axs[i].set_title(title+' '+title0+'$\pm$ %.2f'%ba_error, y=1.11) 
        if i == 4:
            axs[i].set_xlabel('Wavelength ($\AA$)')#,fontsize=14, labelpad=10)
        if i > 0:
            single_line_mask = (data_wvl_adapted>=xstart[i]) & (data_wvl_adapted<=xstop[i])
            axs[i].plot(data_wvl_adapted[single_line_mask],f_synth_adj[single_line_mask],'-',color="darkgoldenrod",linewidth=2)#relflux0/f_moog_cont(wvl0))
            axs[i].errorbar(data_wvl_adapted[single_line_mask],data_flux_adapted[single_line_mask],yerr=contdivstd[slitindex][data_mask][single_line_mask],fmt='.b') #add errorbar 
            axs[i].plot(data_wvl_adapted[single_line_mask],f_synth_final[single_line_mask],'-g',linewidth=2)#relflux0/f_moog_cont(wvl0))
            axs[i].set_xlim([xstart[i],xstop[i]])
            axs[i].set_ylim([0.8,1.05])
            axs[i].xaxis.set_minor_locator(xminorLocator)
    if debug == True or ba_error<0.3:
        plt.savefig(plot_path+outfilename+name_fragment+'.png')
        if plot == True:
            plt.show()
    plt.close()
    return ba_result, ba_error, wvls_adj
