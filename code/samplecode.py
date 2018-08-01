import numpy as np
import matplotlib.pyplot as plt

def weighted_linear_interpolation(lambda_in, array_in, ivar_in, lambda_out):
	"""
	Inputs:
	lambda_in 	-- input wavelength array
	array_in 	-- input array to be interpolated
	ivar_in 	-- input ivar array
	lambda_out 	-- common wavelength array to interpolate to

	Outputs:
	array_out	-- output array
	"""

	lenxin = len(lambda_in)

	i1 = np.searchsorted(lambda_in, lambda_out)
	i1[ i1==0 ] = 1
	i1[ i1==lenxin ] = lenxin-1

	x0 = lambda_in[i1-1]
	x1 = lambda_in[i1]
	y0 = array_in[i1-1]
	y1 = array_in[i1]
	ivar0 = ivar_in[i1-1]
	ivar1 = ivar_in[i1]

	leftweight = 1. - ((lambda_out - x0)/ivar0) / ((lambda_out - x0)/ivar0 + (x1 - lambda_out)/ivar1)
	rightweight = 1. - ((x1 - lambda_out)/ivar1) / ((lambda_out - x0)/ivar0 + (x1 - lambda_out)/ivar1)

	array_out = y0 * leftweight + y1 * rightweight

	return array_out

def testcode():
	"""This is just to show that it works."""

	lambda_in = np.linspace(0,10,10)
	flux_in = np.sin(lambda_in)

	# Evenly weighted ivar
	ivar_in0 = np.ones(len(lambda_in))

	# One of the points is strongly weighted, and one isn't
	ivar_in1 = np.ones(len(lambda_in))
	ivar_in1[2] = 0.0000001 # Less certain
	ivar_in1[3] = 100. # More certain

	lambda_out = np.linspace(0,9,9) + 0.1

	array_out0 = weighted_linear_interpolation(lambda_in, flux_in, ivar_in0, lambda_out)
	array_out1 = weighted_linear_interpolation(lambda_in, flux_in, ivar_in1, lambda_out)

	plt.plot(lambda_in, flux_in, 'o-', label='Original function')
	plt.plot(lambda_out, array_out0, 'o', label='Evenly weighted')
	plt.plot(lambda_out, array_out1, 'o', label='Uneven weights')
	plt.legend()
	plt.show()

	return

def main():
	testcode()

if __name__ == "__main__":
	main()