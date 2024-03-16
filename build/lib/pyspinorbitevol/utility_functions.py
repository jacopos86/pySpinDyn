import numpy as np
from pyspinorbitevol.logging_module import log
#
# utility functions module
#
# 1) function computing the norm of a vector
#    input : n-dimensional vector
# 2) function computing the norm of a c-vector
#    input : n-dimensional vector
# 3) delta function
#    input : x, y
# 4) extract row entry function
# 5) lambda coef
# 6) gradient lambda coef
# 7) compute total number of atoms
#    input : siteslist, kg
# 8) logistic function
# 9) occup. function F
#
#    function (1)
#
def norm_realv(v):
	nrm = np.sqrt(sum(v[:]*v[:]))
	return nrm
#
#    function (2)
#
def norm_cmplv(v):
	nrm = np.sqrt(sum(v[:]*v[:].conjugate()).real)
	return nrm
#
#    function (3)
#
def delta(x, y):
	if x == y:
		return 1.
	else:
		return 0.
#
#    function (4)
#
'''
def MatrixEntry(Site, l, ml, ms):
	k = 0
	for i in range(Site-1):
		llist = sites_list.Atomslist[i].basisset
		for j in range(len(llist)):
			ll = llist[j]
			k = k + 2*(2*ll+1)
	for i in range(l):
		k = k + 2*(2*i+1)
	mllist = np.arange(-l, ml, 1)
	k = k + 2*len(mllist)
	if ms == 0.5:
		k = k
	elif ms == -0.5:
		k = k + 1
	else:
		raise Exception("Error in matrix entry -> ms != +/-0.5")
		log.error("Error in matrix entry -> ms != +/-0.5")
	return k
'''
#
#    function (5)
#
def set_lambda_coef(R0, R1):
	R = R0 - R1
	lx = R[0]/norm_realv(R)
	ly = R[1]/norm_realv(R)
	lz = R[2]/norm_realv(R)
	return [lx, ly, lz]
#
#    function (6)
#
def set_grad_lambda_coef(lc, d):
	grad_lc = np.zeros((3,3))
	for i in range(3):
		for j in range(3):
			r = -lc[i] * lc[j] / d + delta(i, j) / d
			grad_lc[i,j] = r
	return grad_lc
#
#    function (7)
#
def compute_nelec(siteslist, kg):
	nel = 0
	# run over atomic sites
	for i1 in range(siteslist.Nsites):
		nel = nel + siteslist.Atomslist[i1].nel
	return nel
#
#    function 8)
#
def logistic_function(x):
	r = 0.5 * (1. + np.tanh(0.5*x))
	return r
#
#    function 9)
#
def F(z, nel, eigs, kT, kg):
	# F(z) = int dE D(E) f(E, z) - nel
	# F(z) = sum_n sum_k f(e_n(k), z) - nel
	# z = mu
	n = eigs.shape[0]
	r = 0.
	for i in range(n):
		# check dimension
		if kg.D == 0:
			e = eigs[i]
			x = (z - e) / kT
			f = logistic_function(x)
			r = r + f
		elif kg.D == 1:
			nk = eigs.shape[1]
			for ik in range(nk):
				e = eigs[i,ik]
				x = (z - e) / kT
				f = logistic_function(x)
				r = r + kg.wk[ik] * f
		elif kg.D == 2:
			nk1 = eigs.shape[1]
			nk2 = eigs.shape[2]
			for ik in range(nk1):
				for jk in range(nk2):
					e = eigs[i,ik,jk]
					x = (z - e) / kT
					f = logistic_function(x)
					r = r + kg.wk[ik,jk] * f
		elif kg.D == 3:
			[nk1,nk2,nk3] = kg.nkpts
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						e = eigs[i,ik,jk,kk]
						x = (z - e) / kT
						f = logistic_function(x)
						r = r + kg.wk[ik,jk,kk] * f
	r = r - nel
	return r
#
#    function 10)
#
def lorentzian(x, gamma):
	r = 0.
	r = 1./np.pi * gamma / (gamma ** 2 + x ** 2)
	return r
