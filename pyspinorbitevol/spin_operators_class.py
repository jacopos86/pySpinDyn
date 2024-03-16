#
#   This module defines the spin orbital momentum
#   class operators -
#   it implements
#   1) the spin operators (S)
#
import numpy as np
import sys
from pyspinorbitevol.spin_mtxel_functions import Sx_mtxel, Sy_mtxel, Sz_mtxel
#
class SpinMomentumOperators:
	def __init__(self, siteslist, kg, MatrixEntry):
		# init spin operators
		self.init_spin_operators(siteslist, kg)
		# set spin operators
		self.set_spin_operators(siteslist, kg, MatrixEntry)
	def init_spin_operators(self, siteslist, kg):
		# S operators
		Nst = siteslist.Nst
		# check dimension
		if kg.D == 0:
			self.S = np.zeros((Nst, Nst, 3), dtype=np.complex128)
		elif kg.D == 1:
			[nk1, nk2, nk3] = kg.nkpts
			if nk1 != 0:
				self.S = np.zeros((Nst, Nst, nk1, 3), dtype=np.complex128)
			elif nk2 != 0:
				self.S = np.zeros((Nst, Nst, nk2, 3), dtype=np.complex128)
			elif nk3 != 0:
				self.S = np.zeros((Nst, Nst, nk3, 3), dtype=np.complex128)
			else:
				print("wrong n. k-pts for D=1")
				sys.exit(1)
		elif kg.D == 2:
			[nk1, nk2, nk3] = kg.nkpts
			if nk1 == 0:
				self.S = np.zeros((Nst, Nst, nk2, nk3, 3), dtype=np.complex128)
			elif nk2 == 0:
				self.S = np.zeros((Nst, Nst, nk1, nk3, 3), dtype=np.complex128)
			elif nk3 == 0:
				self.S = np.zeros((Nst, Nst, nk1, nk2, 3), dtype=np.complex128)
			else:
				print("wrong n. k-pts for D=2")
				sys.exit(1)
		elif kg.D == 3:
			[nk1, nk2, nk3] = kg.nkpts
			self.S = np.zeros((Nst, Nst, nk1, nk2, nk3, 3), dtype=np.complex128)
	# compute spin operators
	def set_spin_operators(self, siteslist, kg, MatrixEntry):
		s1 = 0.5
		s2 = 0.5
		# run over atoms list
		for i in range(siteslist.Nsites):
			site = i+1
			for l1 in siteslist.Atomslist[i].OrbitalList:
				for ml1 in range(-l1, l1+1):
					for ms1 in [-0.5, 0.5]:
						row = MatrixEntry(siteslist.Atomslist, site, l1, ml1, ms1)
						for l2 in siteslist.Atomslist[i].OrbitalList:
							for ml2 in range(-l2, l2+1):
								for ms2 in [-0.5, 0.5]:
									col = MatrixEntry(siteslist.Atomslist, site, l2, ml2, ms2)
									if kg.D == 0:
										self.S[row,col,0] = Sx_mtxel(s1, ms1, s2, ms2)
										self.S[row,col,1] = Sy_mtxel(s1, ms1, s2, ms2)
										self.S[row,col,2] = Sz_mtxel(s1, ms1, s2, ms2)
									elif kg.D == 1:
										self.S[row,col,:,0] = Sx_mtxel(s1, ms1, s2, ms2)
										self.S[row,col,:,1] = Sy_mtxel(s1, ms1, s2, ms2)
										self.S[row,col,:,2] = Sz_mtxel(s1, ms1, s2, ms2)
									elif kg.D == 2:
										self.S[row,col,:,:,0] = Sx_mtxel(s1, ms1, s2, ms2)
										self.S[row,col,:,:,1] = Sy_mtxel(s1, ms1, s2, ms2)
										self.S[row,col,:,:,2] = Sz_mtxel(s1, ms1, s2, ms2)
									elif kg.D == 3:
										self.S[row,col,:,:,:,0] = Sx_mtxel(s1, ms1, s2, ms2)
										self.S[row,col,:,:,:,1] = Sy_mtxel(s1, ms1, s2, ms2)
										self.S[row,col,:,:,:,2] = Sz_mtxel(s1, ms1, s2, ms2)
									else:
										print("Error: wrong k grid size")
										sys.exit(1)
