# plot_mn.py
# Make plots.
# 
# Created 22 June 18
# Updated 22 June 18
###################################################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import math
from astropy.io import fits
import pandas

def plot_mn_fe(filename):
	"""Plot [Mn/Fe] vs [Fe/H] for all the stars in a file.

	Inputs:
	filename -- input filename
	"""

	# Get data from file
	file = np.genfromtxt(filename, delimiter='\t', skip_header=1)

	feh 	= file[:,3]
	mnh 	= file[:,5]
	mnherr 	= file[:,6]

	# Compute [Mn/Fe]
	mnfe = mnh - feh

	# Make plots
	plt.figure()
	plt.plot(feh,mnfe,'ko')
	plt.title('Sculptor dSph')
	plt.xlabel('[Fe/H]')
	plt.ylabel('[Mn/Fe]')
	plt.savefig('figures/mnfe_scl1.png')
	plt.show()

def main():
	plot_mn_fe('data/scl1.csv')

if __name__ == "__main__":
	main()