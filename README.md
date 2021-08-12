# SMAUG

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About](#about)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About

This is a pipeline for measuring Mn stellar abundances in dwarf galaxies from medium-resolution DEIMOS spectra.


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Here are the main Python packages required and the versions used:
* astropy (4.0)
* matplotlib (3.2.1)
* numpy (1.18.2)
* pandas (1.0.3)

You will also need to install MOOG (the [moog17scat version from Alex Ji](https://github.com/alexji/moog17scat)).

### Installation

Follow the usual steps to clone the repo:
```sh
git clone https://github.com/mdlreyes/SMAUG.git
```

Wrap the FORTRAN code smooth_gauss to Python using the following:
```sh
f2py -c -m smooth_gauss smooth_gauss.f
```

### How the code works

Here are the main parts of the Mn measurement pipeline:

* Code to correct the observed spectra 
  * continuum_division.py: gets rid of slowly-varying continuum in observed spectrum
    * get synthetic spectrum (no [Mn/H] specified) from Ivanna’s grid
    * divide obs/synth
    * fit spline to quotient, mask out Mn regions (and other "bad" regions)
    * divide obs spectrum by spline
  * wvl_corr.py: does empirical wavelength correction using Balmer lines

* Code to make synthetic spectra
  * interp_atmosphere.py: given atmospheric params and [Mn/H], create *.atm file for MOOG
  * run_moog.py: use MOOG to make synthetic spectrum - run 8 times, one for each line list!
    * make *.par file
    * specify linelist
    * specify *.atm file
    * splice all pieces of spectrum together into array

* Code to do the fitting
  * match_spectrum.py: prep the observed and synthetic spectra for fitting
    * open observed spectrum
    * interpolate and smooth synth spectrum to match wavelength array & resolution of observed spectrum
  * chi_squared.py: actually fit the observed spectrum with synthetic spectra
    * calculate chi-square value
    * minimize chi-square ([Mn/H] is free parameter)
  * output_mn.py: run chi_squared.py on all stars in a folder
  * output_mn_mpi.py: same as output_mn.py, but using multiprocessing to make it faster


<!-- USAGE EXAMPLES 
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_
-->

<!-- CONTRIBUTING 
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
-->


<!-- CONTACT -->
## Contact

Mia de los Reyes - [@MiaDoesAstro](https://twitter.com/MiaDoesAstro) - mdelosre@caltech.edu

Project Link: [https://github.com/mdlreyes/SMAUG](https://github.com/mdlreyes/SMAUG)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

[README.md template](https://github.com/othneildrew/Best-README-Template)
