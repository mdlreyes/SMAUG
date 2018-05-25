import ba_measurement as ba_meas
import numpy as np

############# For masks that have carbon measurements, adopt fitted ZREST parameter ###############
carbon_mask_list = ['7078d_blue','7078e_blue','bdra1','bdra2','bdra3','bfor6',
                    'bpal13','bscl1','bscl2','bscl6','bumi1','bumi2','bumi3',
                    'n2419b_blue','n4590a_blue'] #masks that have had carbon measured.
carbon_filenames = np.array(['/raid/caltech/moogify/%s/moogifych.fits.gz'%
                             name for name in carbon_mask_list])

############# Measure Halo Stars #############
#halo_moogify_filename = ba_meas.write_lit_paramaters_halo_stars(overwrite=True)
##print halo_moogify_filename
#halo_moogify_filename = '/raid/gduggan/moogify/lit_halo_900ZD_moogify.fits.gz'
#ba_meas.measure_ba_duggan_moogify([halo_moogify_filename],carbon_filenames,overwrite=True,
#                                  verify_member=False,fine_tune_wvl=True,plot=False,
#                                  debug=False, wvl_max_shift=20)
############ Measure Cluster Stars ############
#cluster_names = ['n2419','n7078','n7078']#,'n4590']
name_list=['n2419b_blue','7078d_blue','7078e_blue','n4590a_blue']#,'6341l1_blue_enk']
####filename = ['/raid/caltech/moogify/%s_blue/moogify.fits.gz'%name for name in name_list]
#filename = ['/raid/gduggan/moogify/%s_moogify.fits.gz'%name for name in name_list]

##ba_meas.cluster_literature() #already called in ba_meas.read_cluster_observation
#filename_ba = ba_meas.read_cluster_observation(cluster_names, filename, carbon_filenames,
#                                 name_fragment="", overwrite=False, simple_line_correction=True, 
#                                 debug = False, plot = False,wvl_max_shift =20)
#print filename_ba

#filename_ba = ['/raid/gduggan/moogify/%s_moogify_ba.fits.gz'%name for name in name_list]
#filename_ba.append('/raid/gduggan/moogify/lit_halo_900ZD_moogify_ba.fits.gz')
########### Make HRS Comparison Table ###########
#ba_meas.make_HRS_comp_table(filename_ba)
########### Compare with HRS measurements ###########
#ba_meas.plot_HRS_comparison(filename_ba)

########### Compare with non-LTE measurements ###########
#ba_meas.plot_nlte_comparison(show_plot=False)

########### Measure dSph Barium Abundances ###########
#ba_meas.measure_all()
########### Make Barium Results Table ###########
#ba_meas.make_barium_results_table()
########### Plot dSph Barium Abundances ###########
#ba_meas.plot_all(show_plot=False)
#ba_meas.plot_all_bafe(show_plot=True,use_catalog=True)

########### Plot BaFe and MgFe with moving average ###############
#ba_meas.plot_moving_av(element_Z=56,name_fragment='04err_06step_')
#ba_meas.plot_moving_av(element_Z=12,name_fragment='04err_06step_')

########### Plot BaFe and MgFe with lit values ###############
ba_meas.plot_ba_results_w_lit(element_Z=56,name_fragment='',ylim=True,errorbars=True)
ba_meas.plot_ba_results_w_lit(element_Z=56,name_fragment='',data=False,ylim=True,errorbars=True)
ba_meas.plot_ba_results_w_lit(element_Z=56,name_fragment='',ylim=True)
ba_meas.plot_ba_results_w_lit(element_Z=56,name_fragment='',data=False,ylim=True)
ba_meas.plot_ba_results_w_lit(element_Z=12,name_fragment='')
ba_meas.plot_ba_results_w_lit(element_Z=63,name_fragment='')

########### Other #############

#ba_meas.sn_hist_all(plot=False)
#ba_meas.simulate_data()



#name_list = ['umi2']
#find_star(dsph_name, name_list, 'Bel80022')
#find_star(dsph_name, name_list, 'Bel10044')


#find_star(dsph_name, name_list, '37716')

#find_star(dsph_name, name_list, '1013644')

dsph_name = 'Sex'
name_list = ['bsex2','bsex3']

dsph_name = 'Scl'
name_list = ['bscl1','bscl2','bscl6']

dsph_name = 'Dra'
name_list = ['bdra1','bdra2','bdra3']

dsph_name = 'UMi'
name_list = ['bumi1','bumi2','bumi3']

dsph_name = 'Pal13'
name_list = ['bpal13']

dsph_name = '6341l1'
name_list = ['6341l1_blue_enk']


dsph_name = 'CVnII'
name_list = ['CVnII6']

dsph_name = 'LeoT'
#name_list = ['leota_2017mar','leota_2016dec']
#ba_meas.average_two_slitmasks('leota_2017mar','leota_2016dec') #average spectra and save to leota_moogify.fits.gz
name_list = ['leota']

dsph_name = 'For'
name_list = ['bfor6']

#filename_red = ['/raid/m31/dsph/%s/%s/moogify7_flexteff.fits.gz'%(name_list[0][:-1],name) for name in name_list]
filename = ['/raid/gduggan/moogify/%s_moogify.fits.gz'%name for name in name_list]
filename_ba = ['/raid/gduggan/moogify/%s_moogify_ba.fits.gz'%name for name in name_list]

#ba_meas.measure_ba_duggan_moogify(filename,carbon_filenames,overwrite=True,
#                                  verify_member=True,fine_tune_wvl=True,
#                                  simple_line_correction=True,plot=True,
#                                  debug=False, wvl_max_shift=20)

#ba_meas.plot_bafe_feh(dsph_name,filename_ba,use_catalog=True,outliers=False,show_plot=True,plot_full=True)
#ba_meas.plot_bafe_feh(dsph_name,filename_ba,outliers=True)
#ba_meas.plot_scatter(dsph_name,filename_ba,x_keyword='FEH',y_keyword='ALPHAFE')
#ba_meas.plot_scatter(dsph_name,filename_ba,x_keyword='TEFF',y_keyword='LOGG')

#filename = ['/raid/caltech/moogify/b%s/moogify.fits.gz'%name for name in name_list]
#ba_meas.read_observation(dsph_name, name_list, filename, filename_red,
#                         name_fragment='',debug = False, plot = False)   
#ba_meas.plot_ba_results('Sex', plot_full=False) 
#ba_meas.plot_ba_results('Scl', plot_full=False) 


#### ascii files for Andrivesky to evaluate non-LTE effects ####
#ba_meas.plot_single_spectrum_ba('bdra1',6,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bdra1',12,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bscl1',63,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bscl1',7,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bscl1',53,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bscl6',69,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bscl6',52,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bumi1',23,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bfor6',9,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bfor6',88,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bscl2',72,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bfor6',84,save_ascii=True)
#ba_meas.plot_single_spectrum_ba('bfor6',89,save_ascii=True)


#ba_meas.plot_all(plot_full=True, copy_files=False,name_fragment='')

#LeoT
ufd_name = 'LeoT'
name_list = ['LeoTa']
filename_red = ['/raid/m31/udwarf/%s/moogify7_flexteff.fits.gz'%(ufd_name)]
filename = ['/raid/caltech/moogify/%s/moogify.fits.gz'%name for name in name_list]
#ba_meas.read_observation(ufd_name, name_list, filename, filename_red,name_fragment='', wvl_max_shift = 10, debug = False, plot = False, verify_member=False)   
#ba_meas.plot_single_spectrum(filename[0],18)
#ba_meas.sn_hist_slitmask(filename[0],filename_red[0],ufd_name,plot=True)
#ba_meas.plot_ba_results('LeoT',name_fragment='',plot_full = False, copy_files = False)


#ba_meas.test_impact_on_rep_stars(test_keyword='scat',name_fragment="_compare", plot=False, compare='grid')
#ba_meas.test_impact_on_rep_stars(test_keyword='scat',name_fragment="_compare_moog14", plot=False, compare='2014')
#ba_meas.test_impact_on_rep_stars(test_keyword='iso_r',name_fragment="_compare", plot=False, compare='2014')
#ba_meas.test_impact_on_rep_stars(test_keyword='iso_s',name_fragment="_compare", plot=False, compare='2014')
#ba_meas.test_impact_on_rep_stars(test_keyword='outliers',name_fragment="_compare", plot=False, compare='grid')

#[['for6', 23], ['for6', 27], ['for6', 31], ['for6', 32], ['for6', 34], ['for6', 35], ['for6', 38], ['for6', 69], ['for6', 70], ['for6', 71], 
#['for6', 72], ['for6', 74], ['for6', 76], ['for6', 78], ['for6', 79]]
 
