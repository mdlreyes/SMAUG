# test_interp_atm.py
# Code for testing interp_atmosphere.py
# 
# Created 22 Jan 18
# Updated 22 Jan 18
###################################################################

import os
import numpy as np
import math
import interp_atmosphere as interp

myanswer = interp.interpolateAtm(temp=7000, logg=2.0, fe=-0.0, alpha=-0.32)

testUp = interp.readAtm(temp=7000, logg=2.0, fe=0.0, alpha=-0.3)
testDown = interp.readAtm(temp=7000, logg=2.0, fe=0.0, alpha=-0.4)

testfinal = (testDown*(-0.3+0.32) + testUp*(-0.32+0.4))/(-0.3+0.4)

print(myanswer - testfinal)