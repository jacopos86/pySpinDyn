#
#   set of spin matrix elements
#   functions:
#   1)  <s1, ms1|S+|s2, ms2>
#   2)  <s1, ms1|S-|s2, ms2>
#   3)  <s1, ms1|Sx|s2, ms2>
#   4)  <s1, ms1|Sy|s2, ms2>
#   5)  <s1, ms1|Sz|s2, ms2>
#
import numpy as np
from pyspinorbitevol.utility_functions import delta
#
#   function (1)
#
def Splus_mtxel(s1, ms1, s2, ms2):
	r = np.sqrt((s2 - ms2) * (s2 + ms2 + 1)) * delta(s1, s2) * delta(ms1, ms2+1)
	return r
#
#   function (2)
#
def Sminus_mtxel(s1, ms1, s2, ms2):
	r = np.sqrt((s2 + ms2) * (s2 - ms2 + 1)) * delta(s1, s2) * delta(ms1, ms2-1)
	return r
#
#   function (3)
#
def Sx_mtxel(s1, ms1, s2, ms2):
	rp = Splus_mtxel(s1, ms1, s2, ms2)
	rm = Sminus_mtxel(s1, ms1, s2, ms2)
	r = (rp + rm) / 2.
	return r
#
#   function (4)
#
def Sy_mtxel(s1, ms1, s2, ms2):
	rp = Splus_mtxel(s1, ms1, s2, ms2)
	rm = Sminus_mtxel(s1, ms1, s2, ms2)
	r = (rp - rm) / (2.*1j)
	return r
#
#   function (5)
#
def Sz_mtxel(s1, ms1, s2, ms2):
	r = ms1 * delta(s1, s2) * delta(ms1, ms2)
	return r
