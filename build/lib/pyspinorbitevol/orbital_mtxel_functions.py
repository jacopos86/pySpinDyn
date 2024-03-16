#
#   set of spin matrix elements
#   functions:
#   3)  <l1, ml1|L0x|l2, ml2>
#   4)  <l1, ml1|L0y|l2, ml2>
#   5)  <l1, ml1|L0z|l2, ml2>
#
import numpy as np
from pyspinorbitevol.utility_functions import delta
#
#   function 1)
#
def L0x_mtxel(l1, ml1, l2, ml2):
	c1 = 0.5*np.sqrt((l2-ml2)*(l2+ml2+1))*delta(ml1, ml2+1)*delta(l1, l2)
	c2 = 0.5*np.sqrt((l2+ml2)*(l2-ml2+1))*delta(ml1, ml2-1)*delta(l1, l2)
	r = c1 + c2
	return r
#
#   function 2)
#
def L0y_mtxel(l1, ml1, l2, ml2):
	c1 = -0.5*1j*np.sqrt((l2-ml2)*(l2+ml2+1))*delta(ml1, ml2+1)*delta(l1, l2)
	c2 = 0.5*1j*np.sqrt((l2+ml2)*(l2-ml2+1))*delta(ml1, ml2-1)*delta(l1, l2)
	r = c1 + c2
	return r
#
#   function 3)
#
def L0z_mtxel(l1, ml1, l2, ml2):
	r = ml2*delta(ml1, ml2)*delta(l1, l2)
	return r
