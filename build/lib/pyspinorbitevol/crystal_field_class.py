#
#   This module defines the crystal field
#   Hamiltonian of the system
#   H0(t_ij^l) = -\sum_ij,l1l2 t_ij^l1l2 c^+_ij^l1 c_ij^l2
#
#   the wave function evolving in time has both electrons+phonons dof
#   Schrodinger formalism
#
import numpy as np
import cmath
from pyspinorbitevol.utility_functions import set_lambda_coef
#
class CrystalFieldHamilt:
	def __init__(self, hopping_params, siteslist, kg, Unitcell, MatrixEntry):
		# set hopping terms
		self.hopping_params = hopping_params
		# set H0
		self.set_H0(siteslist, kg)
		# compute matrix elements
		self.set_tij(siteslist, kg, Unitcell, MatrixEntry)
	# set crystal field Hamiltonian H0
	def set_H0(self, siteslist, kg):
		# set up the Hamiltonian matrix
		Nst = siteslist.Nst
		if kg.D == 0:
			self.H0 = np.zeros((Nst, Nst), dtype=np.complex128)
		elif kg.D == 1:
			[nk1, nk2, nk3] = kg.nkpts
			if nk1 != 0:
				self.H0 = np.zeros((Nst, Nst, nk1), dtype=np.complex128)
			elif nk2 != 0:
				self.H0 = np.zeros((Nst, Nst, nk2), dtype=np.complex128)
			elif nk3 != 0:
				self.H0 = np.zeros((Nst, Nst, nk3), dtype=np.complex128)
			else:
				print("wrong n. k-pts for D=1")
				sys.exit(1)
		elif kg.D == 2:
			[nk1, nk2, nk3] = kg.nkpts
			if nk1 == 0:
				self.H0 = np.zeros((Nst, Nst, nk2, nk3), dtype=np.complex128)
			elif nk2 == 0:
				self.H0 = np.zeros((Nst, Nst, nk1, nk3), dtype=np.complex128)
			elif nk3 == 0:
				self.H0 = np.zeros((Nst, Nst, nk1, nk2), dtype=np.complex128)
			else:
				print("wrong n. k-pts for D=2")
				sys.exit(1)
		elif kg.D == 3:
			[nk1, nk2, nk3] = kg.nkpts
			self.H0 = np.zeros((Nst, Nst, nk1, nk2, nk3), dtype=np.complex128)
	# set s-s hopping matrix elements
	def set_ss_hopping_mtxel(self, Site1, Site2, tss, siteslist, kg, Unitcell, MatrixEntry):
		# (s,s) orbitals pair
		l1 = 0
		l2 = 0
		m  = 0
		if kg.D == 0:
			## nn data site1
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					r = tss
			for ms in [-0.5, 0.5]:
				row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
				col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
				self.H0[row,col] = r
				if row != col:
					self.H0[col,row] = np.conjugate(r)
		elif kg.D == 1:
			# run over k-pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						r = tss * cmath.exp(1j*kR) + r
				for ms in [-0.5, 0.5]:
					row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
					col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
					self.H0[row,col,ik] = r
					if row != col:
						self.H0[col,row,ik] = np.conjugate(r)
					if row == col:
						self.H0[col,row,ik] = self.H0[col,row,ik].real
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							r = tss * cmath.exp(1j*kR) + r
					for ms in [-0.5, 0.5]:
						row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
						col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
						self.H0[row,col,ik,jk] = r
						if row != col:
							self.H0[col,row,ik,jk] = np.conjugate(r)
						if row == col:
							self.H0[row,col,ik,jk] = self.H0[row,col,ik,jk].real
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								r = tss * cmath.exp(1j*kR) + r
						for ms in [-0.5, 0.5]:
							row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
							col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
							self.H0[row,col,ik,jk,kk] = r
							if row != col:
								self.H0[col,row,ik,jk,kk] = np.conjugate(r)
							if row == col:
								self.H0[col,row,ik,jk,kk] = self.H0[col,row,ik,jk,kk].real
						# iterate iik
						iik = iik + 1
	# set s-p hopping matrix elements
	def set_sp_hopping_mtxel(self, Site1, Site2, tsp, siteslist, kg, Unitcell, MatrixEntry):
		# (s,p) orbitals pair
		l1 = 0
		l2 = 1
		m  = 0
		# check dimensionality
		if kg.D == 0:
			# nn data site1
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r1 = 0.
			r2 = 0.
			r3 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r1 = lcoef[2]*tsp
					r2 = -1./np.sqrt(2.)*(lcoef[0]+1j*lcoef[1])*tsp
					r3 = 1./np.sqrt(2.)*(lcoef[0]-1j*lcoef[1])*tsp
			for ms in [-0.5, 0.5]:
				row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
				col1= MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
				col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				self.H0[row,col1] = r1
				self.H0[col1,row] = np.conjugate(r1)
				self.H0[row,col2] = r2
				self.H0[col2,row] = np.conjugate(r2)
				self.H0[row,col3] = r3
				self.H0[col3,row] = np.conjugate(r3)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r1 = 0.
				r2 = 0.
				r3 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r1 = lcoef[2]*tsp*cmath.exp(1j*kR) + r1
						r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r2
						r3 = 1./np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r3
				for ms in [-0.5, 0.5]:
					row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
					col1= MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
					col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					self.H0[row,col1,ik] = r1
					self.H0[col1,row,ik] = np.conjugate(r1)
					self.H0[row,col2,ik] = r2
					self.H0[col2,row,ik] = np.conjugate(r2)
					self.H0[row,col3,ik] = r3
					self.H0[col3,row,ik] = np.conjugate(r3)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r1 = 0.
					r2 = 0.
					r3 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r1 = lcoef[2]*tsp*cmath.exp(1j*kR) + r1
							r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r2
							r3 = 1./np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r3
					for ms in [-0.5, 0.5]:
						row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
						col1= MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
						col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						self.H0[row,col1,ik,jk] = r1
						self.H0[col1,row,ik,jk] = np.conjugate(r1)
						self.H0[row,col2,ik,jk] = r2
						self.H0[col2,row,ik,jk] = np.conjugate(r2)
						self.H0[row,col3,ik,jk] = r3
						self.H0[col3,row,ik,jk] = np.conjugate(r3)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r1 = 0.
						r2 = 0.
						r3 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r1 = lcoef[2]*tsp*cmath.exp(1j*kR) + r1
								r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r2
								r3 = 1./np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r3
						for ms in [-0.5, 0.5]:
							row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
							col1= MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
							col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							self.H0[row,col1,ik,jk,kk] = r1
							self.H0[col1,row,ik,jk,kk] = np.conjugate(r1)
							self.H0[row,col2,ik,jk,kk] = r2
							self.H0[col2,row,ik,jk,kk] = np.conjugate(r2)
							self.H0[row,col3,ik,jk,kk] = r3
							self.H0[col3,row,ik,jk,kk] = np.conjugate(r3)
						# iterate iik
						iik = iik + 1
	# set p-s hopping matrix elements
	def set_ps_hopping_mtxel(self, Site1, Site2, tsp, siteslist, kg, Unitcell, MatrixEntry):
		# (p,s) orbitals pair
		l1 = 1
		l2 = 0
		m  = 0
		# check dimensionality
		if kg.D == 0:
			# nn data site1
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r1 = 0.
			r2 = 0.
			r3 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r1 = lcoef[2]*tsp
					r2 = -1./np.sqrt(2.)*(lcoef[0]+1j*lcoef[1])*tsp
					r3 = 1./np.sqrt(2.)*(lcoef[0]-1j*lcoef[1])*tsp
			for ms in [-0.5, 0.5]:
				row1= MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
				row2= MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row3= MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
				self.H0[row1,col] = (-1)**(l1+l2) * r1
				self.H0[col,row1] = (-1)**(l1+l2) * np.conjugate(r1)
				self.H0[row2,col] = (-1)**(l1+l2) * r2
				self.H0[col,row2] = (-1)**(l1+l2) * np.conjugate(r2)
				self.H0[row3,col] = (-1)**(l1+l2) * r3
				self.H0[col,row3] = (-1)**(l1+l2) * np.conjugate(r3)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r1 = 0.
				r2 = 0.
				r3 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r1 = lcoef[2]*tsp*cmath.exp(1j*kR) + r1
						r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r2
						r3 = 1./np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r3
				for ms in [-0.5, 0.5]:
					row1= MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
					row2= MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row3= MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
					self.H0[row1,col,ik] = (-1)**(l1+l2) * r1
					self.H0[col,row1,ik] = (-1)**(l1+l2) * np.conjugate(r1)
					self.H0[row2,col,ik] = (-1)**(l1+l2) * r2
					self.H0[col,row2,ik] = (-1)**(l1+l2) * np.conjugate(r2)
					self.H0[row3,col,ik] = (-1)**(l1+l2) * r3
					self.H0[col,row3,ik] = (-1)**(l1+l2) * np.conjugate(r3)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r1 = 0.
					r2 = 0.
					r3 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r1 = lcoef[2]*tsp*cmath.exp(1j*kR) + r1
							r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r2
							r3 = 1./np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r3
					for ms in [-0.5, 0.5]:
						row1= MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
						row2= MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row3= MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
						self.H0[row1,col,ik,jk] = (-1)**(l1+l2) * r1
						self.H0[col,row1,ik,jk] = (-1)**(l1+l2) * np.conjugate(r1)
						self.H0[row2,col,ik,jk] = (-1)**(l1+l2) * r2
						self.H0[col,row2,ik,jk] = (-1)**(l1+l2) * np.conjugate(r2)
						self.H0[row3,col,ik,jk] = (-1)**(l1+l2) * r3
						self.H0[col,row3,ik,jk] = (-1)**(l1+l2) * np.conjugate(r3)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r1 = 0.
						r2 = 0.
						r3 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r1 = lcoef[2]*tsp*cmath.exp(1j*kR) + r1
								r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r2
								r3 = 1./np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*tsp*cmath.exp(1j*kR) + r3
						for ms in [-0.5, 0.5]:
							row1= MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
							row2= MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row3= MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
							self.H0[row1,col,ik,jk,kk] = (-1)**(l1+l2) * r1
							self.H0[col,row1,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r1)
							self.H0[row2,col,ik,jk,kk] = (-1)**(l1+l2) * r2
							self.H0[col,row2,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r2)
							self.H0[row3,col,ik,jk,kk] = (-1)**(l1+l2) * r3
							self.H0[col,row3,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r3)
						# iterate iik
						iik = iik + 1
	# set pp SK integrals
	def set_pp_hopping_mtxel(self, Site1, Site2, tpp, siteslist, kg, Unitcell, MatrixEntry):
		# (p,p) orbitals pair
		[tpp0, tpp1] = tpp
		# tpp0 -> tpp,sigma
		# tpp1 -> tpp.pi
		l1 = 1
		l2 = 1
		# check dimensionality
		if kg.D == 0:
			# nn data site1
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r1 = 0.
			r2 = 0.
			r3 = 0.
			r4 = 0.
			r5 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r1 = lcoef[2]**2 * tpp0 + (1 - lcoef[2]**2) * tpp1
					r2 = -lcoef[2]/np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(tpp0-tpp1)
					r3 = lcoef[2]/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(tpp0-tpp1)
					r4 = 0.5*((1-lcoef[2]**2)*tpp0 + (1+lcoef[2]**2)*tpp1)
					r5 = -0.5*(lcoef[0]-1j*lcoef[1])**2 *(tpp0-tpp1)
			for ms in [-0.5, 0.5]:
				row1= MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
				col1= MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
				row2= MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				row3= MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				self.H0[row1,col1] = r1
				self.H0[col1,row1] = np.conjugate(r1)
				self.H0[row1,col2] = r2
				self.H0[col2,row1] = np.conjugate(r2)
				self.H0[row2,col1] = r2
				self.H0[col1,row2] = np.conjugate(r2)
				self.H0[row1,col3] = r3
				self.H0[col3,row1] = np.conjugate(r3)
				self.H0[row3,col1] = r3
				self.H0[col1,row3] = np.conjugate(r3)
				self.H0[row2,col2] = r4
				self.H0[col2,row2] = np.conjugate(r4)
				self.H0[row3,col3] = r4
				self.H0[col3,row3] = np.conjugate(r4)
				self.H0[row2,col3] = r5
				self.H0[col3,row2] = np.conjugate(r5)
				self.H0[row3,col2] = r5
				self.H0[col2,row3] = np.conjugate(r5)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r1 = 0.
				r2 = 0.
				r3 = 0.
				r4 = 0.
				r5 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r1 = (lcoef[2]**2 * tpp0 + (1 - lcoef[2]**2) * tpp1)*cmath.exp(1j*kR) + r1
						r2 = -lcoef[2]/np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(tpp0-tpp1)*cmath.exp(1j*kR) + r2
						r3 = lcoef[2]/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(tpp0-tpp1)*cmath.exp(1j*kR) + r3
						r4 = 0.5*((1-lcoef[2]**2)*tpp0 + (1+lcoef[2]**2)*tpp1)*cmath.exp(1j*kR) + r4
						r5 = -0.5*(lcoef[0]-1j*lcoef[1])**2 *(tpp0-tpp1)*cmath.exp(1j*kR) + r5
				for ms in [-0.5, 0.5]:
					row1= MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
					col1= MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
					row2= MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					row3= MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					self.H0[row1,col1,ik] = r1
					self.H0[col1,row1,ik] = np.conjugate(r1)
					self.H0[row1,col2,ik] = r2
					self.H0[col2,row1,ik] = np.conjugate(r2)
					self.H0[row2,col1,ik] = r2
					self.H0[col1,row2,ik] = np.conjugate(r2)
					self.H0[row1,col3,ik] = r3
					self.H0[col3,row1,ik] = np.conjugate(r3)
					self.H0[row3,col1,ik] = r3
					self.H0[col1,row3,ik] = np.conjugate(r3)
					self.H0[row2,col2,ik] = r4
					self.H0[col2,row2,ik] = np.conjugate(r4)
					self.H0[row3,col3,ik] = r4
					self.H0[col3,row3,ik] = np.conjugate(r4)
					self.H0[row2,col3,ik] = r5
					self.H0[col3,row2,ik] = np.conjugate(r5)
					self.H0[row3,col2,ik] = r5
					self.H0[col2,row3,ik] = np.conjugate(r5)
					if row1 == col1:
						self.H0[row1,col1,ik] = self.H0[row1,col1,ik].real
					if row2 == col2:
						self.H0[row2,col2,ik] = self.H0[row2,col2,ik].real
					if row3 == col3:
						self.H0[row3,col3,ik] = self.H0[row3,col3,ik].real
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r1 = 0.
					r2 = 0.
					r3 = 0.
					r4 = 0.
					r5 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r1 = (lcoef[2]**2 * tpp0 + (1 - lcoef[2]**2) * tpp1)*cmath.exp(1j*kR) + r1
							r2 = -lcoef[2]/np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(tpp0-tpp1)*cmath.exp(1j*kR) + r2
							r3 = lcoef[2]/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(tpp0-tpp1)*cmath.exp(1j*kR) + r3
							r4 = 0.5*((1-lcoef[2]**2)*tpp0 + (1+lcoef[2]**2)*tpp1)*cmath.exp(1j*kR) + r4
							r5 = -0.5*(lcoef[0]-1j*lcoef[1])**2 *(tpp0-tpp1)*cmath.exp(1j*kR) + r5
					for ms in [-0.5, 0.5]:
						row1= MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
						col1= MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
						row2= MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						row3= MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						self.H0[row1,col1,ik,jk] = r1
						self.H0[col1,row1,ik,jk] = np.conjugate(r1)
						self.H0[row1,col2,ik,jk] = r2
						self.H0[col2,row1,ik,jk] = np.conjugate(r2)
						self.H0[row2,col1,ik,jk] = r2
						self.H0[col1,row2,ik,jk] = np.conjugate(r2)
						self.H0[row1,col3,ik,jk] = r3
						self.H0[col3,row1,ik,jk] = np.conjugate(r3)
						self.H0[row3,col1,ik,jk] = r3
						self.H0[col1,row3,ik,jk] = np.conjugate(r3)
						self.H0[row2,col2,ik,jk] = r4
						self.H0[col2,row2,ik,jk] = np.conjugate(r4)
						self.H0[row3,col3,ik,jk] = r4
						self.H0[col3,row3,ik,jk] = np.conjugate(r4)
						self.H0[row2,col3,ik,jk] = r5
						self.H0[col3,row2,ik,jk] = np.conjugate(r5)
						self.H0[row3,col2,ik,jk] = r5
						self.H0[col2,row3,ik,jk] = np.conjugate(r5)
						if row1 == col1:
							self.H0[row1,col1,ik,jk] = self.H0[row1,col1,ik,jk].real
						if row2 == col2:
							self.H0[row2,col2,ik,jk] = self.H0[row2,col2,ik,jk].real
						if row3 == col3:
							self.H0[row3,col3,ik,jk] = self.H0[row3,col3,ik,jk].real
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r1 = 0.
						r2 = 0.
						r3 = 0.
						r4 = 0.
						r5 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r1 = (lcoef[2]**2 * tpp0 + (1 - lcoef[2]**2) * tpp1)*cmath.exp(1j*kR) + r1
								r2 = -lcoef[2]/np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(tpp0-tpp1)*cmath.exp(1j*kR) + r2
								r3 = lcoef[2]/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(tpp0-tpp1)*cmath.exp(1j*kR) + r3
								r4 = 0.5*((1-lcoef[2]**2)*tpp0 + (1+lcoef[2]**2)*tpp1)*cmath.exp(1j*kR) + r4
								r5 = -0.5*(lcoef[0]-1j*lcoef[1])**2 *(tpp0-tpp1)*cmath.exp(1j*kR) + r5
						for ms in [-0.5, 0.5]:
							row1= MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
							col1= MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
							row2= MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							row3= MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							self.H0[row1,col1,ik,jk,kk] = r1
							self.H0[col1,row1,ik,jk,kk] = np.conjugate(r1)
							self.H0[row1,col2,ik,jk,kk] = r2
							self.H0[col2,row1,ik,jk,kk] = np.conjugate(r2)
							self.H0[row2,col1,ik,jk,kk] = r2
							self.H0[col1,row2,ik,jk,kk] = np.conjugate(r2)
							self.H0[row1,col3,ik,jk,kk] = r3
							self.H0[col3,row1,ik,jk,kk] = np.conjugate(r3)
							self.H0[row3,col1,ik,jk,kk] = r3
							self.H0[col1,row3,ik,jk,kk] = np.conjugate(r3)
							self.H0[row2,col2,ik,jk,kk] = r4
							self.H0[col2,row2,ik,jk,kk] = np.conjugate(r4)
							self.H0[row3,col3,ik,jk,kk] = r4
							self.H0[col3,row3,ik,jk,kk] = np.conjugate(r4)
							self.H0[row2,col3,ik,jk,kk] = r5
							self.H0[col3,row2,ik,jk,kk] = np.conjugate(r5)
							self.H0[row3,col2,ik,jk,kk] = r5
							self.H0[col2,row3,ik,jk,kk] = np.conjugate(r5)
							if row1 == col1:
								self.H0[row1,col1,ik,jk,kk] = self.H0[row1,col1,ik,jk,kk].real
							if row2 == col2:
								self.H0[row2,col2,ik,jk,kk] = self.H0[row2,col2,ik,jk,kk].real
							if row3 == col3:
								self.H0[row3,col3,ik,jk,kk] = self.H0[row3,col3,ik,jk,kk].real
						# iterate iik
						iik = iik + 1
	# set sd SK integrals
	def set_sd_hopping_mtxel(self, Site1, Site2, tsd, siteslist, kg, Unitcell, MatrixEntry):
		# (s,d) orbital
		l1 = 0
		l2 = 2
		m = 0
		# check dimension
		if kg.D == 0:
			# nn data site1
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r1 = 0.
			r2 = 0.
			r3 = 0.
			r4 = 0.
			r5 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r1 = 0.5*(3*lcoef[2]**2 - 1)*tsd
					r2 = -np.sqrt(3./2)*lcoef[2]*(lcoef[0]+1j*lcoef[1])*tsd
					r3 = np.sqrt(3./2)*lcoef[2]*(lcoef[0]-1j*lcoef[1])*tsd
					r4 = 0.5*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2 *tsd
					r5 = 0.5*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2 *tsd
			for ms in [-0.5, 0.5]:
				row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
				col1= MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
				col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				col4= MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
				col5= MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
				self.H0[row,col1] = r1
				self.H0[col1,row] = np.conjugate(r1)
				self.H0[row,col2] = r2
				self.H0[col2,row] = np.conjugate(r2)
				self.H0[row,col3] = r3
				self.H0[col3,row] = np.conjugate(r3)
				self.H0[row,col4] = r4
				self.H0[col4,row] = np.conjugate(r4)
				self.H0[row,col5] = r5
				self.H0[col5,row] = np.conjugate(r5)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r1 = 0.
				r2 = 0.
				r3 = 0.
				r4 = 0.
				r5 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r1 = 0.5*(3*lcoef[2]**2 - 1.)*tsd*cmath.exp(1j*kR) + r1
						r2 = -np.sqrt(3./2)*lcoef[2]*(lcoef[0]+1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r2
						r3 = np.sqrt(3./2)*lcoef[2]*(lcoef[0]-1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r3
						r4 = 0.5*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r4
						r5 = 0.5*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r5 
				for ms in [-0.5, 0.5]:
					row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
					col1= MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
					col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					col4= MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
					col5= MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
					self.H0[row,col1,ik] = r1
					self.H0[col1,row,ik] = np.conjugate(r1)
					self.H0[row,col2,ik] = r2
					self.H0[col2,row,ik] = np.conjugate(r2)
					self.H0[row,col3,ik] = r3
					self.H0[col3,row,ik] = np.conjugate(r3)
					self.H0[row,col4,ik] = r4
					self.H0[col4,row,ik] = np.conjugate(r4)
					self.H0[row,col5,ik] = r5
					self.H0[col5,row,ik] = np.conjugate(r5)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r1 = 0.
					r2 = 0.
					r3 = 0.
					r4 = 0.
					r5 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r1 = 0.5*(3*lcoef[2]**2 - 1.)*tsd*cmath.exp(1j*kR) + r1
							r2 = -np.sqrt(3./2)*lcoef[2]*(lcoef[0]+1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r2
							r3 = np.sqrt(3./2)*lcoef[2]*(lcoef[0]-1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r3
							r4 = 0.5*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r4
							r5 = 0.5*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r5 
					for ms in [-0.5, 0.5]:
						row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
						col1= MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
						col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						col4= MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
						col5= MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
						self.H0[row,col1,ik,jk] = r1
						self.H0[col1,row,ik,jk] = np.conjugate(r1)
						self.H0[row,col2,ik,jk] = r2
						self.H0[col2,row,ik,jk] = np.conjugate(r2)
						self.H0[row,col3,ik,jk] = r3
						self.H0[col3,row,ik,jk] = np.conjugate(r3)
						self.H0[row,col4,ik,jk] = r4
						self.H0[col4,row,ik,jk] = np.conjugate(r4)
						self.H0[row,col5,ik,jk] = r5
						self.H0[col5,row,ik,jk] = np.conjugate(r5)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r1 = 0.
						r2 = 0.
						r3 = 0.
						r4 = 0.
						r5 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r1 = 0.5*(3*lcoef[2]**2 - 1.)*tsd*cmath.exp(1j*kR) + r1
								r2 = -np.sqrt(3./2)*lcoef[2]*(lcoef[0]+1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r2
								r3 = np.sqrt(3./2)*lcoef[2]*(lcoef[0]-1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r3
								r4 = 0.5*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r4
								r5 = 0.5*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r5 
						for ms in [-0.5, 0.5]:
							row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
							col1= MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
							col2= MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col3= MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							col4= MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
							col5= MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
							self.H0[row,col1,ik,jk,kk] = r1
							self.H0[col1,row,ik,jk,kk] = np.conjugate(r1)
							self.H0[row,col2,ik,jk,kk] = r2
							self.H0[col2,row,ik,jk,kk] = np.conjugate(r2)
							self.H0[row,col3,ik,jk,kk] = r3
							self.H0[col3,row,ik,jk,kk] = np.conjugate(r3)
							self.H0[row,col4,ik,jk,kk] = r4
							self.H0[col4,row,ik,jk,kk] = np.conjugate(r4)
							self.H0[row,col5,ik,jk,kk] = r5
							self.H0[col5,row,ik,jk,kk] = np.conjugate(r5)
						# iterate iik
						iik = iik + 1
	# set ds SK integrals
	def set_ds_hopping_mtxel(self, Site1, Site2, tsd, siteslist, kg, Unitcell, MatrixEntry):
		# (d,s) orbitals
		l1 = 2
		l2 = 0
		m = 0
		# check dimension
		if kg.D == 0:
			# nn data site1
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r1 = 0.
			r2 = 0.
			r3 = 0.
			r4 = 0.
			r5 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r1 = 0.5*(3*lcoef[2]**2 - 1)*tsd
					r2 = -np.sqrt(3./2)*lcoef[2]*(lcoef[0]+1j*lcoef[1])*tsd
					r3 = np.sqrt(3./2)*lcoef[2]*(lcoef[0]-1j*lcoef[1])*tsd
					r4 = 0.5*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2 *tsd
					r5 = 0.5*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2 *tsd
			for ms in [-0.5, 0.5]:
				row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
				row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
				row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
				col  = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
				self.H0[row1,col] = (-1)**(l1+l2) * r1
				self.H0[col,row1] = (-1)**(l1+l2) * np.conjugate(r1)
				self.H0[row2,col] = (-1)**(l1+l2) * r2
				self.H0[col,row2] = (-1)**(l1+l2) * np.conjugate(r2)
				self.H0[row3,col] = (-1)**(l1+l2) * r3
				self.H0[col,row3] = (-1)**(l1+l2) * np.conjugate(r3)
				self.H0[row4,col] = (-1)**(l1+l2) * r4
				self.H0[col,row4] = (-1)**(l1+l2) * np.conjugate(r4)
				self.H0[row5,col] = (-1)**(l1+l2) * r5
				self.H0[col,row5] = (-1)**(l1+l2) * np.conjugate(r5)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r1 = 0.
				r2 = 0.
				r3 = 0.
				r4 = 0.
				r5 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r1 = 0.5*(3*lcoef[2]**2 - 1.)*tsd*cmath.exp(1j*kR) + r1
						r2 = -np.sqrt(3./2)*lcoef[2]*(lcoef[0]+1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r2
						r3 = np.sqrt(3./2)*lcoef[2]*(lcoef[0]-1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r3
						r4 = 0.5*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r4
						r5 = 0.5*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r5
				for ms in [-0.5, 0.5]:
					row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
					row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
					row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
					col  = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
					self.H0[row1,col,ik] = (-1)**(l1+l2) * r1
					self.H0[col,row1,ik] = (-1)**(l1+l2) * np.conjugate(r1)
					self.H0[row2,col,ik] = (-1)**(l1+l2) * r2
					self.H0[col,row2,ik] = (-1)**(l1+l2) * np.conjugate(r2)
					self.H0[row3,col,ik] = (-1)**(l1+l2) * r3
					self.H0[col,row3,ik] = (-1)**(l1+l2) * np.conjugate(r3)
					self.H0[row4,col,ik] = (-1)**(l1+l2) * r4
					self.H0[col,row4,ik] = (-1)**(l1+l2) * np.conjugate(r4)
					self.H0[row5,col,ik] = (-1)**(l1+l2) * r5
					self.H0[col,row5,ik] = (-1)**(l1+l2) * np.conjugate(r5)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r1 = 0.
					r2 = 0.
					r3 = 0.
					r4 = 0.
					r5 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r1 = 0.5*(3*lcoef[2]**2 - 1.)*tsd*cmath.exp(1j*kR) + r1
							r2 = -np.sqrt(3./2)*lcoef[2]*(lcoef[0]+1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r2
							r3 = np.sqrt(3./2)*lcoef[2]*(lcoef[0]-1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r3
							r4 = 0.5*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r4
							r5 = 0.5*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r5
					for ms in [-0.5, 0.5]:
						row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
						row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
						row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
						col  = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
						self.H0[row1,col,ik,jk] = (-1)**(l1+l2) * r1
						self.H0[col,row1,ik,jk] = (-1)**(l1+l2) * np.conjugate(r1)
						self.H0[row2,col,ik,jk] = (-1)**(l1+l2) * r2
						self.H0[col,row2,ik,jk] = (-1)**(l1+l2) * np.conjugate(r2)
						self.H0[row3,col,ik,jk] = (-1)**(l1+l2) * r3
						self.H0[col,row3,ik,jk] = (-1)**(l1+l2) * np.conjugate(r3)
						self.H0[row4,col,ik,jk] = (-1)**(l1+l2) * r4
						self.H0[col,row4,ik,jk] = (-1)**(l1+l2) * np.conjugate(r4)
						self.H0[row5,col,ik,jk] = (-1)**(l1+l2) * r5
						self.H0[col,row5,ik,jk] = (-1)**(l1+l2) * np.conjugate(r5)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r1 = 0.
						r2 = 0.
						r3 = 0.
						r4 = 0.
						r5 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r1 = 0.5*(3*lcoef[2]**2 - 1.)*tsd*cmath.exp(1j*kR) + r1
								r2 = -np.sqrt(3./2)*lcoef[2]*(lcoef[0]+1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r2
								r3 = np.sqrt(3./2)*lcoef[2]*(lcoef[0]-1j*lcoef[1])*tsd*cmath.exp(1j*kR) + r3
								r4 = 0.5*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r4
								r5 = 0.5*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2 *tsd*cmath.exp(1j*kR) + r5
						for ms in [-0.5, 0.5]:
							row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
							row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
							row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
							col  = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
							self.H0[row1,col,ik,jk,kk] = (-1)**(l1+l2) * r1
							self.H0[col,row1,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r1)
							self.H0[row2,col,ik,jk,kk] = (-1)**(l1+l2) * r2
							self.H0[col,row2,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r2)
							self.H0[row3,col,ik,jk,kk] = (-1)**(l1+l2) * r3
							self.H0[col,row3,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r3)
							self.H0[row4,col,ik,jk,kk] = (-1)**(l1+l2) * r4
							self.H0[col,row4,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r4)
							self.H0[row5,col,ik,jk,kk] = (-1)**(l1+l2) * r5
							self.H0[col,row5,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r5)
						# iterate iik
						iik = iik + 1
	# set pd SK integrals
	def set_pd_hopping_mtxel(self, Site1, Site2, tpd, siteslist, kg, Unitcell, MatrixEntry):
		# (p,d) orbitals
		l1 = 1
		l2 = 2
		# hopping
		[tpd0, tpd1] = tpd
		# dimensions
		if kg.D == 0:
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r1 = 0.
			r2 = 0.
			r3 = 0.
			r4 = 0.
			r5 = 0.
			r6 = 0.
			r7 = 0.
			r8 = 0.
			r9 = 0.
			r10= 0.
			r11= 0.
			r12= 0.
			r13= 0.
			r14= 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r1 = 0.5*lcoef[2]*((3.*lcoef[2]**2-1.)*tpd0 + 2*np.sqrt(3)*(1.-lcoef[2]**2)*tpd1)
					r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)
					r3 = 1/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)
					r4 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)
					r5 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)
					r6 = -1./(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)
					r7 = 1./(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)
					r8 = lcoef[2]/2*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*lcoef[2]**2 *tpd1)
					r9 = -1./4*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)
					r10= 1./4*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)
					r11= -0.5*lcoef[2]*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)
					r12= -0.5*lcoef[2]*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)
					r13= -1./4*(lcoef[0]-1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)
					r14= 1./4*(lcoef[0]+1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)
			for ms in [-0.5, 0.5]:
				row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
				row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
				col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
				col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
				self.H0[row1,col1] = r1
				self.H0[col1,row1] = np.conjugate(r1)
				self.H0[row1,col2] = r2
				self.H0[col2,row1] = np.conjugate(r2)
				self.H0[row1,col3] = r3
				self.H0[col3,row1] = np.conjugate(r3)
				self.H0[row1,col4] = r4
				self.H0[col4,row1] = np.conjugate(r4)
				self.H0[row1,col5] = r5
				self.H0[col5,row1] = np.conjugate(r5)
				self.H0[row2,col1] = r6
				self.H0[col1,row2] = np.conjugate(r6)
				self.H0[row3,col1] = r7
				self.H0[col1,row3] = np.conjugate(r7)
				self.H0[row2,col2] = r8
				self.H0[col2,row2] = np.conjugate(r8)
				self.H0[row3,col3] = r8
				self.H0[col3,row3] = np.conjugate(r8)
				self.H0[row2,col4] = r9
				self.H0[col4,row2] = np.conjugate(r9)
				self.H0[row3,col5] = r10
				self.H0[col5,row3] = np.conjugate(r10)
				self.H0[row2,col3] = r11
				self.H0[col3,row2] = np.conjugate(r11)
				self.H0[row3,col2] = r12
				self.H0[col2,row3] = np.conjugate(r12)
				self.H0[row2,col5] = r13
				self.H0[col5,row2] = np.conjugate(r13)
				self.H0[row3,col4] = r14
				self.H0[col4,row3] = np.conjugate(r14)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r1 = 0.
				r2 = 0.
				r3 = 0.
				r4 = 0.
				r5 = 0.
				r6 = 0.
				r7 = 0.
				r8 = 0.
				r9 = 0.
				r10= 0.
				r11= 0.
				r12= 0.
				r13= 0.
				r14= 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r1 = 0.5*lcoef[2]*((3.*lcoef[2]**2-1.)*tpd0 + 2*np.sqrt(3)*(1.-lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r1
						r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r2
						r3 = 1/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r3
						r4 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r4
						r5 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r5
						r6 = -1./(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r6
						r7 = 1./(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r7
						r8 = lcoef[2]/2*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r8
						r9 = -1./4*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r9
						r10= 1./4*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r10
						r11= -0.5*lcoef[2]*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r11
						r12= -0.5*lcoef[2]*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r12
						r13= -1./4*(lcoef[0]-1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r13
						r14= 1./4*(lcoef[0]+1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r14
				for ms in [-0.5, 0.5]:
					row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
					row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
					col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
					col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
					self.H0[row1,col1,ik] = r1
					self.H0[col1,row1,ik] = np.conjugate(r1)
					self.H0[row1,col2,ik] = r2
					self.H0[col2,row1,ik] = np.conjugate(r2)
					self.H0[row1,col3,ik] = r3
					self.H0[col3,row1,ik] = np.conjugate(r3)
					self.H0[row1,col4,ik] = r4
					self.H0[col4,row1,ik] = np.conjugate(r4)
					self.H0[row1,col5,ik] = r5
					self.H0[col5,row1,ik] = np.conjugate(r5)
					self.H0[row2,col1,ik] = r6
					self.H0[col1,row2,ik] = np.conjugate(r6)
					self.H0[row3,col1,ik] = r7
					self.H0[col1,row3,ik] = np.conjugate(r7)
					self.H0[row2,col2,ik] = r8
					self.H0[col2,row2,ik] = np.conjugate(r8)
					self.H0[row3,col3,ik] = r8
					self.H0[col3,row3,ik] = np.conjugate(r8)
					self.H0[row2,col4,ik] = r9
					self.H0[col4,row2,ik] = np.conjugate(r9)
					self.H0[row3,col5,ik] = r10
					self.H0[col5,row3,ik] = np.conjugate(r10)
					self.H0[row2,col3,ik] = r11
					self.H0[col3,row2,ik] = np.conjugate(r11)
					self.H0[row3,col2,ik] = r12
					self.H0[col2,row3,ik] = np.conjugate(r12)
					self.H0[row2,col5,ik] = r13
					self.H0[col5,row2,ik] = np.conjugate(r13)
					self.H0[row3,col4,ik] = r14
					self.H0[col4,row3,ik] = np.conjugate(r14)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r1 = 0.
					r2 = 0.
					r3 = 0.
					r4 = 0.
					r5 = 0.
					r6 = 0.
					r7 = 0.
					r8 = 0.
					r9 = 0.
					r10= 0.
					r11= 0.
					r12= 0.
					r13= 0.
					r14= 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r1 = 0.5*lcoef[2]*((3.*lcoef[2]**2-1.)*tpd0 + 2*np.sqrt(3)*(1.-lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r1
							r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r2
							r3 = 1/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r3
							r4 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r4
							r5 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r5
							r6 = -1./(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r6
							r7 = 1./(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r7
							r8 = lcoef[2]/2*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r8
							r9 = -1./4*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r9
							r10= 1./4*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r10
							r11= -0.5*lcoef[2]*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r11
							r12= -0.5*lcoef[2]*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r12
							r13= -1./4*(lcoef[0]-1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r13
							r14= 1./4*(lcoef[0]+1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r14
					for ms in [-0.5, 0.5]:
						row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
						row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
						col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
						col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
						self.H0[row1,col1,ik,jk] = r1
						self.H0[col1,row1,ik,jk] = np.conjugate(r1)
						self.H0[row1,col2,ik,jk] = r2
						self.H0[col2,row1,ik,jk] = np.conjugate(r2)
						self.H0[row1,col3,ik,jk] = r3
						self.H0[col3,row1,ik,jk] = np.conjugate(r3)
						self.H0[row1,col4,ik,jk] = r4
						self.H0[col4,row1,ik,jk] = np.conjugate(r4)
						self.H0[row1,col5,ik,jk] = r5
						self.H0[col5,row1,ik,jk] = np.conjugate(r5)
						self.H0[row2,col1,ik,jk] = r6
						self.H0[col1,row2,ik,jk] = np.conjugate(r6)
						self.H0[row3,col1,ik,jk] = r7
						self.H0[col1,row3,ik,jk] = np.conjugate(r7)
						self.H0[row2,col2,ik,jk] = r8
						self.H0[col2,row2,ik,jk] = np.conjugate(r8)
						self.H0[row3,col3,ik,jk] = r8
						self.H0[col3,row3,ik,jk] = np.conjugate(r8)
						self.H0[row2,col4,ik,jk] = r9
						self.H0[col4,row2,ik,jk] = np.conjugate(r9)
						self.H0[row3,col5,ik,jk] = r10
						self.H0[col5,row3,ik,jk] = np.conjugate(r10)
						self.H0[row2,col3,ik,jk] = r11
						self.H0[col3,row2,ik,jk] = np.conjugate(r11)
						self.H0[row3,col2,ik,jk] = r12
						self.H0[col2,row3,ik,jk] = np.conjugate(r12)
						self.H0[row2,col5,ik,jk] = r13
						self.H0[col5,row2,ik,jk] = np.conjugate(r13)
						self.H0[row3,col4,ik,jk] = r14
						self.H0[col4,row3,ik,jk] = np.conjugate(r14)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r1 = 0.
						r2 = 0.
						r3 = 0.
						r4 = 0.
						r5 = 0.
						r6 = 0.
						r7 = 0.
						r8 = 0.
						r9 = 0.
						r10= 0.
						r11= 0.
						r12= 0.
						r13= 0.
						r14= 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r1 = 0.5*lcoef[2]*((3.*lcoef[2]**2-1.)*tpd0 + 2*np.sqrt(3)*(1.-lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r1
								r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r2
								r3 = 1/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r3
								r4 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r4
								r5 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r5
								r6 = -1./(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r6
								r7 = 1./(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r7
								r8 = lcoef[2]/2*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r8
								r9 = -1./4*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r9
								r10= 1./4*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r10
								r11= -0.5*lcoef[2]*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r11
								r12= -0.5*lcoef[2]*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r12
								r13= -1./4*(lcoef[0]-1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r13
								r14= 1./4*(lcoef[0]+1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r14
						for ms in [-0.5, 0.5]:
							row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
							row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
							col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
							col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
							self.H0[row1,col1,ik,jk,kk] = r1
							self.H0[col1,row1,ik,jk,kk] = np.conjugate(r1)
							self.H0[row1,col2,ik,jk,kk] = r2
							self.H0[col2,row1,ik,jk,kk] = np.conjugate(r2)
							self.H0[row1,col3,ik,jk,kk] = r3
							self.H0[col3,row1,ik,jk,kk] = np.conjugate(r3)
							self.H0[row1,col4,ik,jk,kk] = r4
							self.H0[col4,row1,ik,jk,kk] = np.conjugate(r4)
							self.H0[row1,col5,ik,jk,kk] = r5
							self.H0[col5,row1,ik,jk,kk] = np.conjugate(r5)
							self.H0[row2,col1,ik,jk,kk] = r6
							self.H0[col1,row2,ik,jk,kk] = np.conjugate(r6)
							self.H0[row3,col1,ik,jk,kk] = r7
							self.H0[col1,row3,ik,jk,kk] = np.conjugate(r7)
							self.H0[row2,col2,ik,jk,kk] = r8
							self.H0[col2,row2,ik,jk,kk] = np.conjugate(r8)
							self.H0[row3,col3,ik,jk,kk] = r8
							self.H0[col3,row3,ik,jk,kk] = np.conjugate(r8)
							self.H0[row2,col4,ik,jk,kk] = r9
							self.H0[col4,row2,ik,jk,kk] = np.conjugate(r9)
							self.H0[row3,col5,ik,jk,kk] = r10
							self.H0[col5,row3,ik,jk,kk] = np.conjugate(r10)
							self.H0[row2,col3,ik,jk,kk] = r11
							self.H0[col3,row2,ik,jk,kk] = np.conjugate(r11)
							self.H0[row3,col2,ik,jk,kk] = r12
							self.H0[col2,row3,ik,jk,kk] = np.conjugate(r12)
							self.H0[row2,col5,ik,jk,kk] = r13
							self.H0[col5,row2,ik,jk,kk] = np.conjugate(r13)
							self.H0[row3,col4,ik,jk,kk] = r14
							self.H0[col4,row3,ik,jk,kk] = np.conjugate(r14)
						# iterate iik
						iik = iik + 1
	# set dp SK integrals
	def set_dp_hopping_mtxel(self, Site1, Site2, tpd, siteslist, kg, Unitcell, MatrixEntry):
		# (p,d) orbitals
		l1 = 2
		l2 = 1
		# hopping
		[tpd0, tpd1] = tpd
		# dimensions
		if kg.D == 0:
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r1 = 0.
			r2 = 0.
			r3 = 0.
			r4 = 0.
			r5 = 0.
			r6 = 0.
			r7 = 0.
			r8 = 0.
			r9 = 0.
			r10= 0.
			r11= 0.
			r12= 0.
			r13= 0.
			r14= 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r1 = 0.5*lcoef[2]*((3.*lcoef[2]**2-1.)*tpd0 + 2*np.sqrt(3)*(1.-lcoef[2]**2)*tpd1)
					r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)
					r3 = 1/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)
					r4 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)
					r5 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)
					r6 = -1./(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)
					r7 = 1./(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)
					r8 = lcoef[2]/2*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*lcoef[2]**2 *tpd1)
					r9 = -1./4*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)
					r10= 1./4*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)
					r11= -0.5*lcoef[2]*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)
					r12= -0.5*lcoef[2]*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)
					r13= -1./4*(lcoef[0]-1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)
					r14= 1./4*(lcoef[0]+1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)
			for ms in [-0.5, 0.5]:
				row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
				row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
				col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
				col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
				self.H0[row1,col1] = (-1)**(l1+l2) * r1
				self.H0[col1,row1] = (-1)**(l1+l2) * np.conjugate(r1)
				self.H0[row1,col2] = (-1)**(l1+l2) * r2
				self.H0[col2,row1] = (-1)**(l1+l2) * np.conjugate(r2)
				self.H0[row1,col3] = (-1)**(l1+l2) * r3
				self.H0[col3,row1] = (-1)**(l1+l2) * np.conjugate(r3)
				self.H0[row1,col4] = (-1)**(l1+l2) * r4
				self.H0[col4,row1] = (-1)**(l1+l2) * np.conjugate(r4)
				self.H0[row1,col5] = (-1)**(l1+l2) * r5
				self.H0[col5,row1] = (-1)**(l1+l2) * np.conjugate(r5)
				self.H0[row2,col1] = (-1)**(l1+l2) * r6
				self.H0[col1,row2] = (-1)**(l1+l2) * np.conjugate(r6)
				self.H0[row3,col1] = (-1)**(l1+l2) * r7
				self.H0[col1,row3] = (-1)**(l1+l2) * np.conjugate(r7)
				self.H0[row2,col2] = (-1)**(l1+l2) * r8
				self.H0[col2,row2] = (-1)**(l1+l2) * np.conjugate(r8)
				self.H0[row3,col3] = (-1)**(l1+l2) * r8
				self.H0[col3,row3] = (-1)**(l1+l2) * np.conjugate(r8)
				self.H0[row2,col4] = (-1)**(l1+l2) * r9
				self.H0[col4,row2] = (-1)**(l1+l2) * np.conjugate(r9)
				self.H0[row3,col5] = (-1)**(l1+l2) * r10
				self.H0[col5,row3] = (-1)**(l1+l2) * np.conjugate(r10)
				self.H0[row2,col3] = (-1)**(l1+l2) * r11
				self.H0[col3,row2] = (-1)**(l1+l2) * np.conjugate(r11)
				self.H0[row3,col2] = (-1)**(l1+l2) * r12
				self.H0[col2,row3] = (-1)**(l1+l2) * np.conjugate(r12)
				self.H0[row2,col5] = (-1)**(l1+l2) * r13
				self.H0[col5,row2] = (-1)**(l1+l2) * np.conjugate(r13)
				self.H0[row3,col4] = (-1)**(l1+l2) * r14
				self.H0[col4,row3] = (-1)**(l1+l2) * np.conjugate(r14)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r1 = 0.
				r2 = 0.
				r3 = 0.
				r4 = 0.
				r5 = 0.
				r6 = 0.
				r7 = 0.
				r8 = 0.
				r9 = 0.
				r10= 0.
				r11= 0.
				r12= 0.
				r13= 0.
				r14= 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r1 = 0.5*lcoef[2]*((3.*lcoef[2]**2-1.)*tpd0 + 2*np.sqrt(3)*(1.-lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r1
						r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r2
						r3 = 1/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r3
						r4 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r4
						r5 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r5
						r6 = -1./(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r6
						r7 = 1./(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r7
						r8 = lcoef[2]/2*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r8
						r9 = -1./4*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r9
						r10= 1./4*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r10
						r11= -0.5*lcoef[2]*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r11
						r12= -0.5*lcoef[2]*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r12
						r13= -1./4*(lcoef[0]-1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r13
						r14= 1./4*(lcoef[0]+1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r14
				for ms in [-0.5, 0.5]:
					row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
					row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
					col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
					col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
					self.H0[row1,col1,ik] = (-1)**(l1+l2) * r1
					self.H0[col1,row1,ik] = (-1)**(l1+l2) * np.conjugate(r1)
					self.H0[row1,col2,ik] = (-1)**(l1+l2) * r2
					self.H0[col2,row1,ik] = (-1)**(l1+l2) * np.conjugate(r2)
					self.H0[row1,col3,ik] = (-1)**(l1+l2) * r3
					self.H0[col3,row1,ik] = (-1)**(l1+l2) * np.conjugate(r3)
					self.H0[row1,col4,ik] = (-1)**(l1+l2) * r4
					self.H0[col4,row1,ik] = (-1)**(l1+l2) * np.conjugate(r4)
					self.H0[row1,col5,ik] = (-1)**(l1+l2) * r5
					self.H0[col5,row1,ik] = (-1)**(l1+l2) * np.conjugate(r5)
					self.H0[row2,col1,ik] = (-1)**(l1+l2) * r6
					self.H0[col1,row2,ik] = (-1)**(l1+l2) * np.conjugate(r6)
					self.H0[row3,col1,ik] = (-1)**(l1+l2) * r7
					self.H0[col1,row3,ik] = (-1)**(l1+l2) * np.conjugate(r7)
					self.H0[row2,col2,ik] = (-1)**(l1+l2) * r8
					self.H0[col2,row2,ik] = (-1)**(l1+l2) * np.conjugate(r8)
					self.H0[row3,col3,ik] = (-1)**(l1+l2) * r8
					self.H0[col3,row3,ik] = (-1)**(l1+l2) * np.conjugate(r8)
					self.H0[row2,col4,ik] = (-1)**(l1+l2) * r9
					self.H0[col4,row2,ik] = (-1)**(l1+l2) * np.conjugate(r9)
					self.H0[row3,col5,ik] = (-1)**(l1+l2) * r10
					self.H0[col5,row3,ik] = (-1)**(l1+l2) * np.conjugate(r10)
					self.H0[row2,col3,ik] = (-1)**(l1+l2) * r11
					self.H0[col3,row2,ik] = (-1)**(l1+l2) * np.conjugate(r11)
					self.H0[row3,col2,ik] = (-1)**(l1+l2) * r12
					self.H0[col2,row3,ik] = (-1)**(l1+l2) * np.conjugate(r12)
					self.H0[row2,col5,ik] = (-1)**(l1+l2) * r13
					self.H0[col5,row2,ik] = (-1)**(l1+l2) * np.conjugate(r13)
					self.H0[row3,col4,ik] = (-1)**(l1+l2) * r14
					self.H0[col4,row3,ik] = (-1)**(l1+l2) * np.conjugate(r14)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r1 = 0.
					r2 = 0.
					r3 = 0.
					r4 = 0.
					r5 = 0.
					r6 = 0.
					r7 = 0.
					r8 = 0.
					r9 = 0.
					r10= 0.
					r11= 0.
					r12= 0.
					r13= 0.
					r14= 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r1 = 0.5*lcoef[2]*((3.*lcoef[2]**2-1.)*tpd0 + 2*np.sqrt(3)*(1.-lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r1
							r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r2
							r3 = 1/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r3
							r4 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r4
							r5 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r5
							r6 = -1./(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r6
							r7 = 1./(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r7
							r8 = lcoef[2]/2*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r8
							r9 = -1./4*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r9
							r10= 1./4*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r10
							r11= -0.5*lcoef[2]*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r11
							r12= -0.5*lcoef[2]*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r12
							r13= -1./4*(lcoef[0]-1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r13
							r14= 1./4*(lcoef[0]+1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r14
					for ms in [-0.5, 0.5]:
						row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
						row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
						col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
						col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
						self.H0[row1,col1,ik,jk] = (-1)**(l1+l2) * r1
						self.H0[col1,row1,ik,jk] = (-1)**(l1+l2) * np.conjugate(r1)
						self.H0[row1,col2,ik,jk] = (-1)**(l1+l2) * r2
						self.H0[col2,row1,ik,jk] = (-1)**(l1+l2) * np.conjugate(r2)
						self.H0[row1,col3,ik,jk] = (-1)**(l1+l2) * r3
						self.H0[col3,row1,ik,jk] = (-1)**(l1+l2) * np.conjugate(r3)
						self.H0[row1,col4,ik,jk] = (-1)**(l1+l2) * r4
						self.H0[col4,row1,ik,jk] = (-1)**(l1+l2) * np.conjugate(r4)
						self.H0[row1,col5,ik,jk] = (-1)**(l1+l2) * r5
						self.H0[col5,row1,ik,jk] = (-1)**(l1+l2) * np.conjugate(r5)
						self.H0[row2,col1,ik,jk] = (-1)**(l1+l2) * r6
						self.H0[col1,row2,ik,jk] = (-1)**(l1+l2) * np.conjugate(r6)
						self.H0[row3,col1,ik,jk] = (-1)**(l1+l2) * r7
						self.H0[col1,row3,ik,jk] = (-1)**(l1+l2) * np.conjugate(r7)
						self.H0[row2,col2,ik,jk] = (-1)**(l1+l2) * r8
						self.H0[col2,row2,ik,jk] = (-1)**(l1+l2) * np.conjugate(r8)
						self.H0[row3,col3,ik,jk] = (-1)**(l1+l2) * r8
						self.H0[col3,row3,ik,jk] = (-1)**(l1+l2) * np.conjugate(r8)
						self.H0[row2,col4,ik,jk] = (-1)**(l1+l2) * r9
						self.H0[col4,row2,ik,jk] = (-1)**(l1+l2) * np.conjugate(r9)
						self.H0[row3,col5,ik,jk] = (-1)**(l1+l2) * r10
						self.H0[col5,row3,ik,jk] = (-1)**(l1+l2) * np.conjugate(r10)
						self.H0[row2,col3,ik,jk] = (-1)**(l1+l2) * r11
						self.H0[col3,row2,ik,jk] = (-1)**(l1+l2) * np.conjugate(r11)
						self.H0[row3,col2,ik,jk] = (-1)**(l1+l2) * r12
						self.H0[col2,row3,ik,jk] = (-1)**(l1+l2) * np.conjugate(r12)
						self.H0[row2,col5,ik,jk] = (-1)**(l1+l2) * r13
						self.H0[col5,row2,ik,jk] = (-1)**(l1+l2) * np.conjugate(r13)
						self.H0[row3,col4,ik,jk] = (-1)**(l1+l2) * r14
						self.H0[col4,row3,ik,jk] = (-1)**(l1+l2) * np.conjugate(r14)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r1 = 0.
						r2 = 0.
						r3 = 0.
						r4 = 0.
						r5 = 0.
						r6 = 0.
						r7 = 0.
						r8 = 0.
						r9 = 0.
						r10= 0.
						r11= 0.
						r12= 0.
						r13= 0.
						r14= 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r1 = 0.5*lcoef[2]*((3.*lcoef[2]**2-1.)*tpd0 + 2*np.sqrt(3)*(1.-lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r1
								r2 = -1./np.sqrt(2)*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r2
								r3 = 1/np.sqrt(2)*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*lcoef[2]**2 *tpd0 + (1.-2*lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r3
								r4 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r4
								r5 = lcoef[2]/(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0-2*tpd1)*cmath.exp(1j*kR)+r5
								r6 = -1./(2*np.sqrt(2))*(lcoef[0]-1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r6
								r7 = 1./(2*np.sqrt(2))*(lcoef[0]+1j*lcoef[1])*((3*lcoef[2]**2 -1.)*tpd0-2*np.sqrt(3)*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r7
								r8 = lcoef[2]/2*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*lcoef[2]**2 *tpd1)*cmath.exp(1j*kR)+r8
								r9 = -1./4*(lcoef[0]+1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r9
								r10= 1./4*(lcoef[0]-1j*lcoef[1])*(np.sqrt(3)*(1.-lcoef[2]**2)*tpd0 + 2*(1.+lcoef[2]**2)*tpd1)*cmath.exp(1j*kR)+r10
								r11= -0.5*lcoef[2]*(lcoef[0]-1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r11
								r12= -0.5*lcoef[2]*(lcoef[0]+1j*lcoef[1])**2 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r12
								r13= -1./4*(lcoef[0]-1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r13
								r14= 1./4*(lcoef[0]+1j*lcoef[1])**3 *(np.sqrt(3)*tpd0 - 2*tpd1)*cmath.exp(1j*kR)+r14
						for ms in [-0.5, 0.5]:
							row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
							row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
							col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
							col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
							self.H0[row1,col1,ik,jk,kk] = (-1)**(l1+l2) * r1
							self.H0[col1,row1,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r1)
							self.H0[row1,col2,ik,jk,kk] = (-1)**(l1+l2) * r2
							self.H0[col2,row1,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r2)
							self.H0[row1,col3,ik,jk,kk] = (-1)**(l1+l2) * r3
							self.H0[col3,row1,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r3)
							self.H0[row1,col4,ik,jk,kk] = (-1)**(l1+l2) * r4
							self.H0[col4,row1,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r4)
							self.H0[row1,col5,ik,jk,kk] = (-1)**(l1+l2) * r5
							self.H0[col5,row1,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r5)
							self.H0[row2,col1,ik,jk,kk] = (-1)**(l1+l2) * r6
							self.H0[col1,row2,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r6)
							self.H0[row3,col1,ik,jk,kk] = (-1)**(l1+l2) * r7
							self.H0[col1,row3,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r7)
							self.H0[row2,col2,ik,jk,kk] = (-1)**(l1+l2) * r8
							self.H0[col2,row2,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r8)
							self.H0[row3,col3,ik,jk,kk] = (-1)**(l1+l2) * r8
							self.H0[col3,row3,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r8)
							self.H0[row2,col4,ik,jk,kk] = (-1)**(l1+l2) * r9
							self.H0[col4,row2,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r9)
							self.H0[row3,col5,ik,jk,kk] = (-1)**(l1+l2) * r10
							self.H0[col5,row3,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r10)
							self.H0[row2,col3,ik,jk,kk] = (-1)**(l1+l2) * r11
							self.H0[col3,row2,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r11)
							self.H0[row3,col2,ik,jk,kk] = (-1)**(l1+l2) * r12
							self.H0[col2,row3,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r12)
							self.H0[row2,col5,ik,jk,kk] = (-1)**(l1+l2) * r13
							self.H0[col5,row2,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r13)
							self.H0[row3,col4,ik,jk,kk] = (-1)**(l1+l2) * r14
							self.H0[col4,row3,ik,jk,kk] = (-1)**(l1+l2) * np.conjugate(r14)
						# iterate iik
						iik = iik + 1
	# set dd SK integrals
	def set_dd_hopping_mtxel(self, Site1, Site2, tdd, siteslist, kg, Unitcell, MatrixEntry):
		# (d,d) hopping
		l1 = 2
		l2 = 2
		# hopping
		[tdd0, tdd1, tdd2] = tdd
		# dimension
		if kg.D == 0:
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r0 = 0.
			r1 = 0.
			r2 = 0.
			r3 = 0.
			r4 = 0.
			r56= 0.
			r7 = 0.
			r8 = 0.
			r9 = 0.
			r10= 0.
			r11= 0.
			r1213 = 0.
			r14 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r0 = 1./4*(3*lcoef[2]**2 -1)**2 *tdd0 + 3*lcoef[2]**2 *(1.-lcoef[2]**2)*tdd1 + 3./4*(1.-lcoef[2]**2)**2 *tdd2
					a1 = -0.5*lcoef[2]*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])
					a2 = 0.5*lcoef[2]*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])
					b12= (3*lcoef[2]**2 -1)*tdd0 + 2*(1.-2*lcoef[2]**2)*tdd1 - (1.-lcoef[2]**2)*tdd2
					r1 = a1*b12
					r2 = a2*b12
					a3 = 1./4*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2
					a4 = 1./4*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2
					b34= (3*lcoef[2]**2 -1.)*tdd0 - 4.*lcoef[2]**2 *tdd1 + (1+lcoef[2]**2)*tdd2
					r3 = a3*b34
					r4 = a4*b34
					r56= 1.5*lcoef[2]**2 *(1.-lcoef[2]**2)*tdd0 + 0.5*(4*lcoef[2]**4 -3*lcoef[2]**2 +1)*tdd1 + 0.5*(1-lcoef[2]**4)*tdd2
					a7 = -lcoef[2]/4 *(lcoef[0]+1j*lcoef[1])
					a8 = lcoef[2]/4 *(lcoef[0]-1j*lcoef[1])
					b78= 3*(1-lcoef[2]**2)*tdd0 + 4*lcoef[2]**2 *tdd1 - (3+lcoef[2]**2)*tdd2
					r7 = a7*b78
					r8 = a8*b78
					a9 = -0.5*(lcoef[0]-1j*lcoef[1])**2
					b9 = 3*lcoef[2]**2 *tdd0 + (1-4*lcoef[2]**2)*tdd1 + (lcoef[2]**2 -1)*tdd2
					r9 = a9*b9
					a10= -lcoef[2]/4 *(lcoef[0]-1j*lcoef[1])**3
					a11= lcoef[2]/4 *(lcoef[0]+1j*lcoef[1])**3
					b1011= 3*tdd0 - 4*tdd1 + tdd2
					r10 = a10*b1011
					r11 = a11*b1011
					r1213= 3./8*(1-lcoef[2]**2)**2 *tdd0 + 0.5*(1-lcoef[2]**4)*tdd1 + 1./8*(lcoef[2]**4 +6*lcoef[2]**2 +1)*tdd2
					r14= 1./8*(lcoef[0]-1j*lcoef[1])**4 *(3*tdd0 -4*tdd1 +tdd2)
			for ms in [-0.5, 0.5]:
				row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
				row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
				row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
				col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
				col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
				col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
				self.H0[row1,col1] = r0
				self.H0[col1,row1] = np.conjugate(r0)
				self.H0[row1,col2] = r1
				self.H0[col2,row1] = np.conjugate(r1)
				self.H0[row2,col1] = r1
				self.H0[col1,row2] = np.conjugate(r1)
				self.H0[row1,col3] = r2
				self.H0[col3,row1] = np.conjugate(r2)
				self.H0[row3,col1] = r2
				self.H0[col1,row3] = np.conjugate(r2)
				self.H0[row1,col4] = r3
				self.H0[col4,row1] = np.conjugate(r3)
				self.H0[row4,col1] = r3
				self.H0[col1,row4] = np.conjugate(r3)
				self.H0[row1,col5] = r4
				self.H0[col5,row1] = np.conjugate(r4)
				self.H0[row5,col1] = r4
				self.H0[col1,row5] = np.conjugate(r4)
				self.H0[row2,col2] = r56
				self.H0[col2,row2] = np.conjugate(r56)
				self.H0[row3,col3] = r56
				self.H0[col3,row3] = np.conjugate(r56)
				self.H0[row2,col4] = r7
				self.H0[col4,row2] = np.conjugate(r7)
				self.H0[row4,col2] = r7
				self.H0[col2,row4] = np.conjugate(r7)
				self.H0[row3,col5] = r8
				self.H0[col5,row3] = np.conjugate(r8)
				self.H0[row5,col3] = r8
				self.H0[col3,row5] = np.conjugate(r8)
				self.H0[row2,col3] = r9
				self.H0[col3,row2] = np.conjugate(r9)
				self.H0[row3,col2] = r9
				self.H0[col2,row3] = np.conjugate(r9)
				self.H0[row2,col5] = r10
				self.H0[col5,row2] = np.conjugate(r10)
				self.H0[row5,col2] = r10
				self.H0[col2,row5] = np.conjugate(r10)
				self.H0[row3,col4] = r11
				self.H0[col4,row3] = np.conjugate(r11)
				self.H0[row4,col3] = r11
				self.H0[col3,row4] = np.conjugate(r11)
				self.H0[row4,col4] = r1213
				self.H0[col4,row4] = np.conjugate(r1213)
				self.H0[row5,col5] = r1213
				self.H0[col5,row5] = np.conjugate(r1213)
				self.H0[row4,col5] = r14
				self.H0[col5,row4] = np.conjugate(r14)
				self.H0[row5,col4] = r14
				self.H0[col4,row5] = np.conjugate(r14)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r0 = 0.
				r1 = 0.
				r2 = 0.
				r3 = 0.
				r4 = 0.
				r56= 0.
				r7 = 0.
				r8 = 0.
				r9 = 0.
				r10= 0.
				r11= 0.
				r1213 = 0.
				r14 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r0 = (1./4*(3*lcoef[2]**2 -1)**2 *tdd0 + 3*lcoef[2]**2 *(1.-lcoef[2]**2)*tdd1 + 3./4*(1.-lcoef[2]**2)**2 *tdd2)*cmath.exp(1j*kR) + r0
						a1 = -0.5*lcoef[2]*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])
						a2 = 0.5*lcoef[2]*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])
						b12= (3*lcoef[2]**2 -1)*tdd0 + 2*(1.-2*lcoef[2]**2)*tdd1 - (1.-lcoef[2]**2)*tdd2
						r1 = a1*b12*cmath.exp(1j*kR) + r1
						r2 = a2*b12*cmath.exp(1j*kR) + r2
						a3 = 1./4*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2
						a4 = 1./4*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2
						b34= (3*lcoef[2]**2 -1.)*tdd0 - 4.*lcoef[2]**2 *tdd1 + (1+lcoef[2]**2)*tdd2
						r3 = a3*b34*cmath.exp(1j*kR) + r3
						r4 = a4*b34*cmath.exp(1j*kR) + r4
						r56= (1.5*lcoef[2]**2 *(1.-lcoef[2]**2)*tdd0 + 0.5*(4*lcoef[2]**4 -3*lcoef[2]**2 +1)*tdd1 + 0.5*(1-lcoef[2]**4)*tdd2)*cmath.exp(1j*kR) + r56
						a7 = -lcoef[2]/4 *(lcoef[0]+1j*lcoef[1])
						a8 = lcoef[2]/4 *(lcoef[0]-1j*lcoef[1])
						b78= 3*(1-lcoef[2]**2)*tdd0 + 4*lcoef[2]**2 *tdd1 - (3+lcoef[2]**2)*tdd2
						r7 = a7*b78*cmath.exp(1j*kR) + r7
						r8 = a8*b78*cmath.exp(1j*kR) + r8
						a9 = -0.5*(lcoef[0]-1j*lcoef[1])**2
						b9 = 3*lcoef[2]**2 *tdd0 + (1-4*lcoef[2]**2)*tdd1 + (lcoef[2]**2 -1)*tdd2
						r9 = a9*b9*cmath.exp(1j*kR) + r9
						a10= -lcoef[2]/4 *(lcoef[0]-1j*lcoef[1])**3
						a11= lcoef[2]/4 *(lcoef[0]+1j*lcoef[1])**3
						b1011= 3*tdd0 - 4*tdd1 + tdd2
						r10= a10*b1011*cmath.exp(1j*kR) + r10
						r11= a11*b1011*cmath.exp(1j*kR) + r11
						r1213= (3./8*(1-lcoef[2]**2)**2 *tdd0 + 0.5*(1-lcoef[2]**4)*tdd1 + 1./8*(lcoef[2]**4 +6*lcoef[2]**2 +1)*tdd2)*cmath.exp(1j*kR) + r1213
						r14= 1./8*(lcoef[0]-1j*lcoef[1])**4 *(3*tdd0 -4*tdd1 +tdd2)*cmath.exp(1j*kR) + r14
				for ms in [-0.5, 0.5]:
					row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
					row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
					row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
					col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
					col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
					col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
					self.H0[row1,col1,ik] = r0
					self.H0[col1,row1,ik] = np.conjugate(r0)
					self.H0[row1,col2,ik] = r1
					self.H0[col2,row1,ik] = np.conjugate(r1)
					self.H0[row2,col1,ik] = r1
					self.H0[col1,row2,ik] = np.conjugate(r1)
					self.H0[row1,col3,ik] = r2
					self.H0[col3,row1,ik] = np.conjugate(r2)
					self.H0[row3,col1,ik] = r2
					self.H0[col1,row3,ik] = np.conjugate(r2)
					self.H0[row1,col4,ik] = r3
					self.H0[col4,row1,ik] = np.conjugate(r3)
					self.H0[row4,col1,ik] = r3
					self.H0[col1,row4,ik] = np.conjugate(r3)
					self.H0[row1,col5,ik] = r4
					self.H0[col5,row1,ik] = np.conjugate(r4)
					self.H0[row5,col1,ik] = r4
					self.H0[col1,row5,ik] = np.conjugate(r4)
					self.H0[row2,col2,ik] = r56
					self.H0[col2,row2,ik] = np.conjugate(r56)
					self.H0[row3,col3,ik] = r56
					self.H0[col3,row3,ik] = np.conjugate(r56)
					self.H0[row2,col4,ik] = r7
					self.H0[col4,row2,ik] = np.conjugate(r7)
					self.H0[row4,col2,ik] = r7
					self.H0[col2,row4,ik] = np.conjugate(r7)
					self.H0[row3,col5,ik] = r8
					self.H0[col5,row3,ik] = np.conjugate(r8)
					self.H0[row5,col3,ik] = r8
					self.H0[col3,row5,ik] = np.conjugate(r8)
					self.H0[row2,col3,ik] = r9
					self.H0[col3,row2,ik] = np.conjugate(r9)
					self.H0[row3,col2,ik] = r9
					self.H0[col2,row3,ik] = np.conjugate(r9)
					self.H0[row2,col5,ik] = r10
					self.H0[col5,row2,ik] = np.conjugate(r10)
					self.H0[row5,col2,ik] = r10
					self.H0[col2,row5,ik] = np.conjugate(r10)
					self.H0[row3,col4,ik] = r11
					self.H0[col4,row3,ik] = np.conjugate(r11)
					self.H0[row4,col3,ik] = r11
					self.H0[col3,row4,ik] = np.conjugate(r11)
					self.H0[row4,col4,ik] = r1213
					self.H0[col4,row4,ik] = np.conjugate(r1213)
					self.H0[row5,col5,ik] = r1213
					self.H0[col5,row5,ik] = np.conjugate(r1213)
					self.H0[row4,col5,ik] = r14
					self.H0[col5,row4,ik] = np.conjugate(r14)
					self.H0[row5,col4,ik] = r14
					self.H0[col4,row5,ik] = np.conjugate(r14)
					if row1 == col1:
						self.H0[row1,col1,ik] = self.H0[row1,col1,ik].real
					if row2 == col2:
						self.H0[row2,col2,ik] = self.H0[row2,col2,ik].real
					if row3 == col3:
						self.H0[row3,col3,ik] = self.H0[row3,col3,ik].real
					if row4 == col4:
						self.H0[row4,col4,ik] = self.H0[row4,col4,ik].real
					if row5 == col5:
						self.H0[row5,col5,ik] = self.H0[row5,col5,ik].real
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r0 = 0.
					r1 = 0.
					r2 = 0.
					r3 = 0.
					r4 = 0.
					r56= 0.
					r7 = 0.
					r8 = 0.
					r9 = 0.
					r10= 0.
					r11= 0.
					r1213 = 0.
					r14 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r0 = (1./4*(3*lcoef[2]**2 -1)**2 *tdd0 + 3*lcoef[2]**2 *(1.-lcoef[2]**2)*tdd1 + 3./4*(1.-lcoef[2]**2)**2 *tdd2)*cmath.exp(1j*kR) + r0
							a1 = -0.5*lcoef[2]*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])
							a2 = 0.5*lcoef[2]*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])
							b12= (3*lcoef[2]**2 -1)*tdd0 + 2*(1.-2*lcoef[2]**2)*tdd1 - (1.-lcoef[2]**2)*tdd2
							r1 = a1*b12*cmath.exp(1j*kR) + r1
							r2 = a2*b12*cmath.exp(1j*kR) + r2
							a3 = 1./4*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2
							a4 = 1./4*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2
							b34= (3*lcoef[2]**2 -1.)*tdd0 - 4.*lcoef[2]**2 *tdd1 + (1+lcoef[2]**2)*tdd2
							r3 = a3*b34*cmath.exp(1j*kR) + r3
							r4 = a4*b34*cmath.exp(1j*kR) + r4
							r56= (1.5*lcoef[2]**2 *(1.-lcoef[2]**2)*tdd0 + 0.5*(4*lcoef[2]**4 -3*lcoef[2]**2 +1)*tdd1 + 0.5*(1-lcoef[2]**4)*tdd2)*cmath.exp(1j*kR) + r56
							a7 = -lcoef[2]/4 *(lcoef[0]+1j*lcoef[1])
							a8 = lcoef[2]/4 *(lcoef[0]-1j*lcoef[1])
							b78= 3*(1-lcoef[2]**2)*tdd0 + 4*lcoef[2]**2 *tdd1 - (3+lcoef[2]**2)*tdd2
							r7 = a7*b78*cmath.exp(1j*kR) + r7
							r8 = a8*b78*cmath.exp(1j*kR) + r8
							a9 = -0.5*(lcoef[0]-1j*lcoef[1])**2
							b9 = 3*lcoef[2]**2 *tdd0 + (1-4*lcoef[2]**2)*tdd1 + (lcoef[2]**2 -1)*tdd2
							r9 = a9*b9*cmath.exp(1j*kR) + r9
							a10= -lcoef[2]/4 *(lcoef[0]-1j*lcoef[1])**3
							a11= lcoef[2]/4 *(lcoef[0]+1j*lcoef[1])**3
							b1011= 3*tdd0 - 4*tdd1 + tdd2
							r10= a10*b1011*cmath.exp(1j*kR) + r10
							r11= a11*b1011*cmath.exp(1j*kR) + r11
							r1213= (3./8*(1-lcoef[2]**2)**2 *tdd0 + 0.5*(1-lcoef[2]**4)*tdd1 + 1./8*(lcoef[2]**4 +6*lcoef[2]**2 +1)*tdd2)*cmath.exp(1j*kR) + r1213
							r14= 1./8*(lcoef[0]-1j*lcoef[1])**4 *(3*tdd0 -4*tdd1 +tdd2)*cmath.exp(1j*kR) + r14
					for ms in [-0.5, 0.5]:
						row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
						row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
						row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
						col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
						col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
						col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
						self.H0[row1,col1,ik,jk] = r0
						self.H0[col1,row1,ik,jk] = np.conjugate(r0)
						self.H0[row1,col2,ik,jk] = r1
						self.H0[col2,row1,ik,jk] = np.conjugate(r1)
						self.H0[row2,col1,ik,jk] = r1
						self.H0[col1,row2,ik,jk] = np.conjugate(r1)
						self.H0[row1,col3,ik,jk] = r2
						self.H0[col3,row1,ik,jk] = np.conjugate(r2)
						self.H0[row3,col1,ik,jk] = r2
						self.H0[col1,row3,ik,jk] = np.conjugate(r2)
						self.H0[row1,col4,ik,jk] = r3
						self.H0[col4,row1,ik,jk] = np.conjugate(r3)
						self.H0[row4,col1,ik,jk] = r3
						self.H0[col1,row4,ik,jk] = np.conjugate(r3)
						self.H0[row1,col5,ik,jk] = r4
						self.H0[col5,row1,ik,jk] = np.conjugate(r4)
						self.H0[row5,col1,ik,jk] = r4
						self.H0[col1,row5,ik,jk] = np.conjugate(r4)
						self.H0[row2,col2,ik,jk] = r56
						self.H0[col2,row2,ik,jk] = np.conjugate(r56)
						self.H0[row3,col3,ik,jk] = r56
						self.H0[col3,row3,ik,jk] = np.conjugate(r56)
						self.H0[row2,col4,ik,jk] = r7
						self.H0[col4,row2,ik,jk] = np.conjugate(r7)
						self.H0[row4,col2,ik,jk] = r7
						self.H0[col2,row4,ik,jk] = np.conjugate(r7)
						self.H0[row3,col5,ik,jk] = r8
						self.H0[col5,row3,ik,jk] = np.conjugate(r8)
						self.H0[row5,col3,ik,jk] = r8
						self.H0[col3,row5,ik,jk] = np.conjugate(r8)
						self.H0[row2,col3,ik,jk] = r9
						self.H0[col3,row2,ik,jk] = np.conjugate(r9)
						self.H0[row3,col2,ik,jk] = r9
						self.H0[col2,row3,ik,jk] = np.conjugate(r9)
						self.H0[row2,col5,ik,jk] = r10
						self.H0[col5,row2,ik,jk] = np.conjugate(r10)
						self.H0[row5,col2,ik,jk] = r10
						self.H0[col2,row5,ik,jk] = np.conjugate(r10)
						self.H0[row3,col4,ik,jk] = r11
						self.H0[col4,row3,ik,jk] = np.conjugate(r11)
						self.H0[row4,col3,ik,jk] = r11
						self.H0[col3,row4,ik,jk] = np.conjugate(r11)
						self.H0[row4,col4,ik,jk] = r1213
						self.H0[col4,row4,ik,jk] = np.conjugate(r1213)
						self.H0[row5,col5,ik,jk] = r1213
						self.H0[col5,row5,ik,jk] = np.conjugate(r1213)
						self.H0[row4,col5,ik,jk] = r14
						self.H0[col5,row4,ik,jk] = np.conjugate(r14)
						self.H0[row5,col4,ik,jk] = r14
						self.H0[col4,row5,ik,jk] = np.conjugate(r14)
						if row1 == col1:
							self.H0[row1,col1,ik,jk] = self.H0[row1,col1,ik,jk].real
						if row2 == col2:
							self.H0[row2,col2,ik,jk] = self.H0[row2,col2,ik,jk].real
						if row3 == col3:
							self.H0[row3,col3,ik,jk] = self.H0[row3,col3,ik,jk].real
						if row4 == col4:
							self.H0[row4,col4,ik,jk] = self.H0[row4,col4,ik,jk].real
						if row5 == col5:
							self.H0[row5,col5,ik,jk] = self.H0[row5,col5,ik,jk].real
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r0 = 0.
						r1 = 0.
						r2 = 0.
						r3 = 0.
						r4 = 0.
						r56= 0.
						r7 = 0.
						r8 = 0.
						r9 = 0.
						r10= 0.
						r11= 0.
						r1213 = 0.
						r14 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r0 = (1./4*(3*lcoef[2]**2 -1)**2 *tdd0 + 3*lcoef[2]**2 *(1.-lcoef[2]**2)*tdd1 + 3./4*(1.-lcoef[2]**2)**2 *tdd2)*cmath.exp(1j*kR) + r0
								a1 = -0.5*lcoef[2]*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])
								a2 = 0.5*lcoef[2]*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])
								b12= (3*lcoef[2]**2 -1)*tdd0 + 2*(1.-2*lcoef[2]**2)*tdd1 - (1.-lcoef[2]**2)*tdd2
								r1 = a1*b12*cmath.exp(1j*kR) + r1
								r2 = a2*b12*cmath.exp(1j*kR) + r2
								a3 = 1./4*np.sqrt(3./2)*(lcoef[0]+1j*lcoef[1])**2
								a4 = 1./4*np.sqrt(3./2)*(lcoef[0]-1j*lcoef[1])**2
								b34= (3*lcoef[2]**2 -1.)*tdd0 - 4.*lcoef[2]**2 *tdd1 + (1+lcoef[2]**2)*tdd2
								r3 = a3*b34*cmath.exp(1j*kR) + r3
								r4 = a4*b34*cmath.exp(1j*kR) + r4
								r56= (1.5*lcoef[2]**2 *(1.-lcoef[2]**2)*tdd0 + 0.5*(4*lcoef[2]**4 -3*lcoef[2]**2 +1)*tdd1 + 0.5*(1-lcoef[2]**4)*tdd2)*cmath.exp(1j*kR) + r56
								a7 = -lcoef[2]/4 *(lcoef[0]+1j*lcoef[1])
								a8 = lcoef[2]/4 *(lcoef[0]-1j*lcoef[1])
								b78= 3*(1-lcoef[2]**2)*tdd0 + 4*lcoef[2]**2 *tdd1 - (3+lcoef[2]**2)*tdd2
								r7 = a7*b78*cmath.exp(1j*kR) + r7
								r8 = a8*b78*cmath.exp(1j*kR) + r8
								a9 = -0.5*(lcoef[0]-1j*lcoef[1])**2
								b9 = 3*lcoef[2]**2 *tdd0 + (1-4*lcoef[2]**2)*tdd1 + (lcoef[2]**2 -1)*tdd2
								r9 = a9*b9*cmath.exp(1j*kR) + r9
								a10= -lcoef[2]/4 *(lcoef[0]-1j*lcoef[1])**3
								a11= lcoef[2]/4 *(lcoef[0]+1j*lcoef[1])**3
								b1011= 3*tdd0 - 4*tdd1 + tdd2
								r10= a10*b1011*cmath.exp(1j*kR) + r10
								r11= a11*b1011*cmath.exp(1j*kR) + r11
								r1213= (3./8*(1-lcoef[2]**2)**2 *tdd0 + 0.5*(1-lcoef[2]**4)*tdd1 + 1./8*(lcoef[2]**4 +6*lcoef[2]**2 +1)*tdd2)*cmath.exp(1j*kR) + r1213
								r14= 1./8*(lcoef[0]-1j*lcoef[1])**4 *(3*tdd0 -4*tdd1 +tdd2)*cmath.exp(1j*kR) + r14
						for ms in [-0.5, 0.5]:
							row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
							row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
							row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
							col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
							col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
							col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
							self.H0[row1,col1,ik,jk,kk] = r0
							self.H0[col1,row1,ik,jk,kk] = np.conjugate(r0)
							self.H0[row1,col2,ik,jk,kk] = r1
							self.H0[col2,row1,ik,jk,kk] = np.conjugate(r1)
							self.H0[row2,col1,ik,jk,kk] = r1
							self.H0[col1,row2,ik,jk,kk] = np.conjugate(r1)
							self.H0[row1,col3,ik,jk,kk] = r2
							self.H0[col3,row1,ik,jk,kk] = np.conjugate(r2)
							self.H0[row3,col1,ik,jk,kk] = r2
							self.H0[col1,row3,ik,jk,kk] = np.conjugate(r2)
							self.H0[row1,col4,ik,jk,kk] = r3
							self.H0[col4,row1,ik,jk,kk] = np.conjugate(r3)
							self.H0[row4,col1,ik,jk,kk] = r3
							self.H0[col1,row4,ik,jk,kk] = np.conjugate(r3)
							self.H0[row1,col5,ik,jk,kk] = r4
							self.H0[col5,row1,ik,jk,kk] = np.conjugate(r4)
							self.H0[row5,col1,ik,jk,kk] = r4
							self.H0[col1,row5,ik,jk,kk] = np.conjugate(r4)
							self.H0[row2,col2,ik,jk,kk] = r56
							self.H0[col2,row2,ik,jk,kk] = np.conjugate(r56)
							self.H0[row3,col3,ik,jk,kk] = r56
							self.H0[col3,row3,ik,jk,kk] = np.conjugate(r56)
							self.H0[row2,col4,ik,jk,kk] = r7
							self.H0[col4,row2,ik,jk,kk] = np.conjugate(r7)
							self.H0[row4,col2,ik,jk,kk] = r7
							self.H0[col2,row4,ik,jk,kk] = np.conjugate(r7)
							self.H0[row3,col5,ik,jk,kk] = r8
							self.H0[col5,row3,ik,jk,kk] = np.conjugate(r8)
							self.H0[row5,col3,ik,jk,kk] = r8
							self.H0[col3,row5,ik,jk,kk] = np.conjugate(r8)
							self.H0[row2,col3,ik,jk,kk] = r9
							self.H0[col3,row2,ik,jk,kk] = np.conjugate(r9)
							self.H0[row3,col2,ik,jk,kk] = r9
							self.H0[col2,row3,ik,jk,kk] = np.conjugate(r9)
							self.H0[row2,col5,ik,jk,kk] = r10
							self.H0[col5,row2,ik,jk,kk] = np.conjugate(r10)
							self.H0[row5,col2,ik,jk,kk] = r10
							self.H0[col2,row5,ik,jk,kk] = np.conjugate(r10)
							self.H0[row3,col4,ik,jk,kk] = r11
							self.H0[col4,row3,ik,jk,kk] = np.conjugate(r11)
							self.H0[row4,col3,ik,jk,kk] = r11
							self.H0[col3,row4,ik,jk,kk] = np.conjugate(r11)
							self.H0[row4,col4,ik,jk,kk] = r1213
							self.H0[col4,row4,ik,jk,kk] = np.conjugate(r1213)
							self.H0[row5,col5,ik,jk,kk] = r1213
							self.H0[col5,row5,ik,jk,kk] = np.conjugate(r1213)
							self.H0[row4,col5,ik,jk,kk] = r14
							self.H0[col5,row4,ik,jk,kk] = np.conjugate(r14)
							self.H0[row5,col4,ik,jk,kk] = r14
							self.H0[col4,row5,ik,jk,kk] = np.conjugate(r14)
							if row1 == col1:
								self.H0[row1,col1,ik,jk,kk] = self.H0[row1,col1,ik,jk,kk].real
							if row2 == col2:
								self.H0[row2,col2,ik,jk,kk] = self.H0[row2,col2,ik,jk,kk].real
							if row3 == col3:
								self.H0[row3,col3,ik,jk,kk] = self.H0[row3,col3,ik,jk,kk].real
							if row4 == col4:
								self.H0[row4,col4,ik,jk,kk] = self.H0[row4,col4,ik,jk,kk].real
							if row5 == col5:
								self.H0[row5,col5,ik,jk,kk] = self.H0[row5,col5,ik,jk,kk].real
						# iterate iik
						iik = iik + 1
	# sf SK integrals
	def set_sf_hopping_mtxel(self, Site1, Site2, tsf, siteslist, kg, Unitcell, MatrixEntry):
		# (s,f) orbitals
		l1 = 0
		l2 = 3
		m = 0
		# dimensions
		if kg.D == 0:
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r0 = 0.
			r1 = 0.
			r2 = 0.
			r3 = 0.
			r4 = 0.
			r5 = 0.
			r6 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r0 = 0.5*lcoef[2]*(5*lcoef[2]**2 -3)*tsf
					r1 = 0.5*np.sqrt(3./2)*lcoef[0]*(5*lcoef[2]**2 -1)*tsf
					r2 = 0.5*np.sqrt(3./2)*lcoef[1]*(5*lcoef[2]**2 -1)*tsf
					r3 = 0.5*np.sqrt(15)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tsf
					r4 = np.sqrt(15)*lcoef[2]*lcoef[0]*lcoef[1]*tsf
					r5 = 0.5*np.sqrt(5./2)*lcoef[0]*(lcoef[0]**2 -3*lcoef[1]**2)*tsf
					r6 = 0.5*np.sqrt(5./2)*lcoef[1]*(3*lcoef[0]**2 -lcoef[1]**2)*tsf
			for ms in [-0.5, 0.5]:
				row0 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
				col0 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
				col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col2 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				col3 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
				col4 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
				col5 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
				col6 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
				self.H0[row0,col0] = r0
				self.H0[col0,row0] = np.conjugate(r0)
				self.H0[row0,col1] =-1/np.sqrt(2)*(r1+1j*r2)
				self.H0[col1,row0] =-1/np.sqrt(2)*(np.conjugate(r1)-1j*np.conjugate(r2))
				self.H0[row0,col2] = 1/np.sqrt(2)*(r1-1j*r2)
				self.H0[col2,row0] = 1/np.sqrt(2)*(np.conjugate(r1)+1j*np.conjugate(r2))
				self.H0[row0,col3] = 1/np.sqrt(2)*(r3+1j*r4)
				self.H0[col3,row0] = 1/np.sqrt(2)*(np.conjugate(r3)-1j*np.conjugate(r4))
				self.H0[row0,col4] = 1/np.sqrt(2)*(r3-1j*r4)
				self.H0[col4,row0] = 1/np.sqrt(2)*(np.conjugate(r3)+1j*np.conjugate(r4))
				self.H0[row0,col5] =-1/np.sqrt(2)*(r5+1j*r6)
				self.H0[col5,row0] =-1/np.sqrt(2)*(np.conjugate(r5)-1j*np.conjugate(r6))
				self.H0[row0,col6] = 1/np.sqrt(2)*(r5-1j*r6)
				self.H0[col6,row0] = 1/np.sqrt(2)*(np.conjugate(r5)+1j*np.conjugate(r6))
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r0 = 0.
				r1 = 0.
				r2 = 0.
				r3 = 0.
				r4 = 0.
				r5 = 0.
				r6 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r0 = 0.5*lcoef[2]*(5*lcoef[2]**2 -3)*tsf*cmath.exp(1j*kR) + r0
						r1 = 0.5*np.sqrt(3./2)*lcoef[0]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r1
						r2 = 0.5*np.sqrt(3./2)*lcoef[1]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r2
						r3 = 0.5*np.sqrt(15)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r3
						r4 = np.sqrt(15)*lcoef[2]*lcoef[0]*lcoef[1]*tsf*cmath.exp(1j*kR) + r4
						r5 = 0.5*np.sqrt(5./2)*lcoef[0]*(lcoef[0]**2 -3*lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r5
						r6 = 0.5*np.sqrt(5./2)*lcoef[1]*(3*lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r6
				for ms in [-0.5, 0.5]:
					row0 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
					col0 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
					col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col2 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					col3 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
					col4 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
					col5 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
					col6 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
					self.H0[row0,col0,ik] = r0
					self.H0[col0,row0,ik] = np.conjugate(r0)
					self.H0[row0,col1,ik] =-1/np.sqrt(2)*(r1+1j*r2)
					self.H0[col1,row0,ik] =-1/np.sqrt(2)*(np.conjugate(r1)-1j*np.conjugate(r2))
					self.H0[row0,col2,ik] = 1/np.sqrt(2)*(r1-1j*r2)
					self.H0[col2,row0,ik] = 1/np.sqrt(2)*(np.conjugate(r1)+1j*np.conjugate(r2))
					self.H0[row0,col3,ik] = 1/np.sqrt(2)*(r3+1j*r4)
					self.H0[col3,row0,ik] = 1/np.sqrt(2)*(np.conjugate(r3)-1j*np.conjugate(r4))
					self.H0[row0,col4,ik] = 1/np.sqrt(2)*(r3-1j*r4)
					self.H0[col4,row0,ik] = 1/np.sqrt(2)*(np.conjugate(r3)+1j*np.conjugate(r4))
					self.H0[row0,col5,ik] =-1/np.sqrt(2)*(r5+1j*r6)
					self.H0[col5,row0,ik] =-1/np.sqrt(2)*(np.conjugate(r5)-1j*np.conjugate(r6))
					self.H0[row0,col6,ik] = 1/np.sqrt(2)*(r5-1j*r6)
					self.H0[col6,row0,ik] = 1/np.sqrt(2)*(np.conjugate(r5)+1j*np.conjugate(r6))
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r0 = 0.
					r1 = 0.
					r2 = 0.
					r3 = 0.
					r4 = 0.
					r5 = 0.
					r6 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r0 = 0.5*lcoef[2]*(5*lcoef[2]**2 -3)*tsf*cmath.exp(1j*kR) + r0
							r1 = 0.5*np.sqrt(3./2)*lcoef[0]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r1
							r2 = 0.5*np.sqrt(3./2)*lcoef[1]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r2
							r3 = 0.5*np.sqrt(15)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r3
							r4 = np.sqrt(15)*lcoef[2]*lcoef[0]*lcoef[1]*tsf*cmath.exp(1j*kR) + r4
							r5 = 0.5*np.sqrt(5./2)*lcoef[0]*(lcoef[0]**2 -3*lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r5
							r6 = 0.5*np.sqrt(5./2)*lcoef[1]*(3*lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r6
					for ms in [-0.5, 0.5]:
						row0 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
						col0 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
						col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col2 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						col3 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
						col4 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
						col5 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
						col6 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
						self.H0[row0,col0,ik,jk] = r0
						self.H0[col0,row0,ik,jk] = np.conjugate(r0)
						self.H0[row0,col1,ik,jk] =-1/np.sqrt(2)*(r1+1j*r2)
						self.H0[col1,row0,ik,jk] =-1/np.sqrt(2)*(np.conjugate(r1)-1j*np.conjugate(r2))
						self.H0[row0,col2,ik,jk] = 1/np.sqrt(2)*(r1-1j*r2)
						self.H0[col2,row0,ik,jk] = 1/np.sqrt(2)*(np.conjugate(r1)+1j*np.conjugate(r2))
						self.H0[row0,col3,ik,jk] = 1/np.sqrt(2)*(r3+1j*r4)
						self.H0[col3,row0,ik,jk] = 1/np.sqrt(2)*(np.conjugate(r3)-1j*np.conjugate(r4))
						self.H0[row0,col4,ik,jk] = 1/np.sqrt(2)*(r3-1j*r4)
						self.H0[col4,row0,ik,jk] = 1/np.sqrt(2)*(np.conjugate(r3)+1j*np.conjugate(r4))
						self.H0[row0,col5,ik,jk] =-1/np.sqrt(2)*(r5+1j*r6)
						self.H0[col5,row0,ik,jk] =-1/np.sqrt(2)*(np.conjugate(r5)-1j*np.conjugate(r6))
						self.H0[row0,col6,ik,jk] = 1/np.sqrt(2)*(r5-1j*r6)
						self.H0[col6,row0,ik,jk] = 1/np.sqrt(2)*(np.conjugate(r5)+1j*np.conjugate(r6))
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r0 = 0.
						r1 = 0.
						r2 = 0.
						r3 = 0.
						r4 = 0.
						r5 = 0.
						r6 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r0 = 0.5*lcoef[2]*(5*lcoef[2]**2 -3)*tsf*cmath.exp(1j*kR) + r0
								r1 = 0.5*np.sqrt(3./2)*lcoef[0]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r1
								r2 = 0.5*np.sqrt(3./2)*lcoef[1]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r2
								r3 = 0.5*np.sqrt(15)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r3
								r4 = np.sqrt(15)*lcoef[2]*lcoef[0]*lcoef[1]*tsf*cmath.exp(1j*kR) + r4
								r5 = 0.5*np.sqrt(5./2)*lcoef[0]*(lcoef[0]**2 -3*lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r5
								r6 = 0.5*np.sqrt(5./2)*lcoef[1]*(3*lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r6
						for ms in [-0.5, 0.5]:
							row0 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
							col0 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
							col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col2 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							col3 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
							col4 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
							col5 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
							col6 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
							self.H0[row0,col0,ik,jk,kk] = r0
							self.H0[col0,row0,ik,jk,kk] = np.conjugate(r0)
							self.H0[row0,col1,ik,jk,kk] =-1/np.sqrt(2)*(r1+1j*r2)
							self.H0[col1,row0,ik,jk,kk] =-1/np.sqrt(2)*(np.conjugate(r1)-1j*np.conjugate(r2))
							self.H0[row0,col2,ik,jk,kk] = 1/np.sqrt(2)*(r1-1j*r2)
							self.H0[col2,row0,ik,jk,kk] = 1/np.sqrt(2)*(np.conjugate(r1)+1j*np.conjugate(r2))
							self.H0[row0,col3,ik,jk,kk] = 1/np.sqrt(2)*(r3+1j*r4)
							self.H0[col3,row0,ik,jk,kk] = 1/np.sqrt(2)*(np.conjugate(r3)-1j*np.conjugate(r4))
							self.H0[row0,col4,ik,jk,kk] = 1/np.sqrt(2)*(r3-1j*r4)
							self.H0[col4,row0,ik,jk,kk] = 1/np.sqrt(2)*(np.conjugate(r3)+1j*np.conjugate(r4))
							self.H0[row0,col5,ik,jk,kk] =-1/np.sqrt(2)*(r5+1j*r6)
							self.H0[col5,row0,ik,jk,kk] =-1/np.sqrt(2)*(np.conjugate(r5)-1j*np.conjugate(r6))
							self.H0[row0,col6,ik,jk,kk] = 1/np.sqrt(2)*(r5-1j*r6)
							self.H0[col6,row0,ik,jk,kk] = 1/np.sqrt(2)*(np.conjugate(r5)+1j*np.conjugate(r6))
						# iterate iik
						iik = iik + 1
	# fs SK integrals
	def set_fs_hopping_mtxel(self, Site1, Site2, tsf, siteslist, kg, Unitcell, MatrixEntry):
		# (f,s) orbitals
		l1 = 3
		l2 = 0
		m = 0
		# dimensions
		if kg.D == 0:
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r0 = 0.
			r1 = 0.
			r2 = 0.
			r3 = 0.
			r4 = 0.
			r5 = 0.
			r6 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r0 = 0.5*lcoef[2]*(5*lcoef[2]**2 -3)*tsf
					r1 = 0.5*np.sqrt(3./2)*lcoef[0]*(5*lcoef[2]**2 -1)*tsf
					r2 = 0.5*np.sqrt(3./2)*lcoef[1]*(5*lcoef[2]**2 -1)*tsf
					r3 = 0.5*np.sqrt(15)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tsf
					r4 = np.sqrt(15)*lcoef[2]*lcoef[0]*lcoef[1]*tsf
					r5 = 0.5*np.sqrt(5./2)*lcoef[0]*(lcoef[0]**2 -3*lcoef[1]**2)*tsf
					r6 = 0.5*np.sqrt(5./2)*lcoef[1]*(3*lcoef[0]**2 -lcoef[1]**2)*tsf
			for ms in [-0.5, 0.5]:
				row0 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
				row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row2 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				row3 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
				row4 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
				row5 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
				row6 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
				col0 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
				self.H0[row0,col0] = r0*(-1)**(l1+l2)
				self.H0[col0,row0] = np.conjugate(r0)*(-1)**(l1+l2)
				self.H0[row1,col0] =-1/np.sqrt(2)*(r1+1j*r2)*(-1)**(l1+l2)
				self.H0[col0,row1] =-1/np.sqrt(2)*(np.conjugate(r1)-1j*np.conjugate(r2))*(-1)**(l1+l2)
				self.H0[row2,col0] = 1/np.sqrt(2)*(r1-1j*r2)*(-1)**(l1+l2)
				self.H0[col0,row2] = 1/np.sqrt(2)*(np.conjugate(r1)+1j*np.conjugate(r2))*(-1)**(l1+l2)
				self.H0[row3,col0] = 1/np.sqrt(2)*(r3+1j*r4)*(-1)**(l1+l2)
				self.H0[col0,row3] = 1/np.sqrt(2)*(np.conjugate(r3)-1j*np.conjugate(r4))*(-1)**(l1+l2)
				self.H0[row4,col0] = 1/np.sqrt(2)*(r3-1j*r4)*(-1)**(l1+l2)
				self.H0[col0,row4] = 1/np.sqrt(2)*(np.conjugate(r3)+1j*np.conjugate(r4))*(-1)**(l1+l2)
				self.H0[row5,col0] =-1/np.sqrt(2)*(r5+1j*r6)*(-1)**(l1+l2)
				self.H0[col0,row5] =-1/np.sqrt(2)*(np.conjugate(r5)-1j*np.conjugate(r6))*(-1)**(l1+l2)
				self.H0[row6,col0] = 1/np.sqrt(2)*(r5-1j*r6)*(-1)**(l1+l2)
				self.H0[col0,row6] = 1/np.sqrt(2)*(np.conjugate(r5)+1j*np.conjugate(r6))*(-1)**(l1+l2)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r0 = 0.
				r1 = 0.
				r2 = 0.
				r3 = 0.
				r4 = 0.
				r5 = 0.
				r6 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r0 = 0.5*lcoef[2]*(5*lcoef[2]**2 -3)*tsf*cmath.exp(1j*kR) + r0
						r1 = 0.5*np.sqrt(3./2)*lcoef[0]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r1
						r2 = 0.5*np.sqrt(3./2)*lcoef[1]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r2
						r3 = 0.5*np.sqrt(15)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r3
						r4 = np.sqrt(15)*lcoef[2]*lcoef[0]*lcoef[1]*tsf*cmath.exp(1j*kR) + r4
						r5 = 0.5*np.sqrt(5./2)*lcoef[0]*(lcoef[0]**2 -3*lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r5
						r6 = 0.5*np.sqrt(5./2)*lcoef[1]*(3*lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r6
				for ms in [-0.5, 0.5]:
					row0 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
					row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row2 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					row3 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
					row4 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
					row5 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
					row6 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
					col0 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
					self.H0[row0,col0,ik] = r0*(-1)**(l1+l2)
					self.H0[col0,row0,ik] = np.conjugate(r0)*(-1)**(l1+l2)
					self.H0[row1,col0,ik] =-1/np.sqrt(2)*(r1+1j*r2)*(-1)**(l1+l2)
					self.H0[col0,row1,ik] =-1/np.sqrt(2)*(np.conjugate(r1)-1j*np.conjugate(r2))*(-1)**(l1+l2)
					self.H0[row2,col0,ik] = 1/np.sqrt(2)*(r1-1j*r2)*(-1)**(l1+l2)
					self.H0[col0,row2,ik] = 1/np.sqrt(2)*(np.conjugate(r1)+1j*np.conjugate(r2))*(-1)**(l1+l2)
					self.H0[row3,col0,ik] = 1/np.sqrt(2)*(r3+1j*r4)*(-1)**(l1+l2)
					self.H0[col0,row3,ik] = 1/np.sqrt(2)*(np.conjugate(r3)-1j*np.conjugate(r4))*(-1)**(l1+l2)
					self.H0[row4,col0,ik] = 1/np.sqrt(2)*(r3-1j*r4)*(-1)**(l1+l2)
					self.H0[col0,row4,ik] = 1/np.sqrt(2)*(np.conjugate(r3)+1j*np.conjugate(r4))*(-1)**(l1+l2)
					self.H0[row5,col0,ik] =-1/np.sqrt(2)*(r5+1j*r6)*(-1)**(l1+l2)
					self.H0[col0,row5,ik] =-1/np.sqrt(2)*(np.conjugate(r5)-1j*np.conjugate(r6))*(-1)**(l1+l2)
					self.H0[row6,col0,ik] = 1/np.sqrt(2)*(r5-1j*r6)*(-1)**(l1+l2)
					self.H0[col0,row6,ik] = 1/np.sqrt(2)*(np.conjugate(r5)+1j*np.conjugate(r6))*(-1)**(l1+l2)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r0 = 0.
					r1 = 0.
					r2 = 0.
					r3 = 0.
					r4 = 0.
					r5 = 0.
					r6 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r0 = 0.5*lcoef[2]*(5*lcoef[2]**2 -3)*tsf*cmath.exp(1j*kR) + r0
							r1 = 0.5*np.sqrt(3./2)*lcoef[0]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r1
							r2 = 0.5*np.sqrt(3./2)*lcoef[1]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r2
							r3 = 0.5*np.sqrt(15)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r3
							r4 = np.sqrt(15)*lcoef[2]*lcoef[0]*lcoef[1]*tsf*cmath.exp(1j*kR) + r4
							r5 = 0.5*np.sqrt(5./2)*lcoef[0]*(lcoef[0]**2 -3*lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r5
							r6 = 0.5*np.sqrt(5./2)*lcoef[1]*(3*lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r6
					for ms in [-0.5, 0.5]:
						row0 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
						row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row2 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						row3 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
						row4 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
						row5 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
						row6 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
						col0 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
						self.H0[row0,col0,ik,jk] = r0*(-1)**(l1+l2)
						self.H0[col0,row0,ik,jk] = np.conjugate(r0)*(-1)**(l1+l2)
						self.H0[row1,col0,ik,jk] =-1/np.sqrt(2)*(r1+1j*r2)*(-1)**(l1+l2)
						self.H0[col0,row1,ik,jk] =-1/np.sqrt(2)*(np.conjugate(r1)-1j*np.conjugate(r2))*(-1)**(l1+l2)
						self.H0[row2,col0,ik,jk] = 1/np.sqrt(2)*(r1-1j*r2)*(-1)**(l1+l2)
						self.H0[col0,row2,ik,jk] = 1/np.sqrt(2)*(np.conjugate(r1)+1j*np.conjugate(r2))*(-1)**(l1+l2)
						self.H0[row3,col0,ik,jk] = 1/np.sqrt(2)*(r3+1j*r4)*(-1)**(l1+l2)
						self.H0[col0,row3,ik,jk] = 1/np.sqrt(2)*(np.conjugate(r3)-1j*np.conjugate(r4))*(-1)**(l1+l2)
						self.H0[row4,col0,ik,jk] = 1/np.sqrt(2)*(r3-1j*r4)*(-1)**(l1+l2)
						self.H0[col0,row4,ik,jk] = 1/np.sqrt(2)*(np.conjugate(r3)+1j*np.conjugate(r4))*(-1)**(l1+l2)
						self.H0[row5,col0,ik,jk] =-1/np.sqrt(2)*(r5+1j*r6)*(-1)**(l1+l2)
						self.H0[col0,row5,ik,jk] =-1/np.sqrt(2)*(np.conjugate(r5)-1j*np.conjugate(r6))*(-1)**(l1+l2)
						self.H0[row6,col0,ik,jk] = 1/np.sqrt(2)*(r5-1j*r6)*(-1)**(l1+l2)
						self.H0[col0,row6,ik,jk] = 1/np.sqrt(2)*(np.conjugate(r5)+1j*np.conjugate(r6))*(-1)**(l1+l2)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r0 = 0.
						r1 = 0.
						r2 = 0.
						r3 = 0.
						r4 = 0.
						r5 = 0.
						r6 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r0 = 0.5*lcoef[2]*(5*lcoef[2]**2 -3)*tsf*cmath.exp(1j*kR) + r0
								r1 = 0.5*np.sqrt(3./2)*lcoef[0]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r1
								r2 = 0.5*np.sqrt(3./2)*lcoef[1]*(5*lcoef[2]**2 -1)*tsf*cmath.exp(1j*kR) + r2
								r3 = 0.5*np.sqrt(15)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r3
								r4 = np.sqrt(15)*lcoef[2]*lcoef[0]*lcoef[1]*tsf*cmath.exp(1j*kR) + r4
								r5 = 0.5*np.sqrt(5./2)*lcoef[0]*(lcoef[0]**2 -3*lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r5
								r6 = 0.5*np.sqrt(5./2)*lcoef[1]*(3*lcoef[0]**2 -lcoef[1]**2)*tsf*cmath.exp(1j*kR) + r6
						for ms in [-0.5, 0.5]:
							row0 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
							row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row2 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							row3 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
							row4 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
							row5 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
							row6 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
							col0 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
							self.H0[row0,col0,ik,jk,kk] = r0*(-1)**(l1+l2)
							self.H0[col0,row0,ik,jk,kk] = np.conjugate(r0)*(-1)**(l1+l2)
							self.H0[row1,col0,ik,jk,kk] =-1/np.sqrt(2)*(r1+1j*r2)*(-1)**(l1+l2)
							self.H0[col0,row1,ik,jk,kk] =-1/np.sqrt(2)*(np.conjugate(r1)-1j*np.conjugate(r2))*(-1)**(l1+l2)
							self.H0[row2,col0,ik,jk,kk] = 1/np.sqrt(2)*(r1-1j*r2)*(-1)**(l1+l2)
							self.H0[col0,row2,ik,jk,kk] = 1/np.sqrt(2)*(np.conjugate(r1)+1j*np.conjugate(r2))*(-1)**(l1+l2)
							self.H0[row3,col0,ik,jk,kk] = 1/np.sqrt(2)*(r3+1j*r4)*(-1)**(l1+l2)
							self.H0[col0,row3,ik,jk,kk] = 1/np.sqrt(2)*(np.conjugate(r3)-1j*np.conjugate(r4))*(-1)**(l1+l2)
							self.H0[row4,col0,ik,jk,kk] = 1/np.sqrt(2)*(r3-1j*r4)*(-1)**(l1+l2)
							self.H0[col0,row4,ik,jk,kk] = 1/np.sqrt(2)*(np.conjugate(r3)+1j*np.conjugate(r4))*(-1)**(l1+l2)
							self.H0[row5,col0,ik,jk,kk] =-1/np.sqrt(2)*(r5+1j*r6)*(-1)**(l1+l2)
							self.H0[col0,row5,ik,jk,kk] =-1/np.sqrt(2)*(np.conjugate(r5)-1j*np.conjugate(r6))*(-1)**(l1+l2)
							self.H0[row6,col0,ik,jk,kk] = 1/np.sqrt(2)*(r5-1j*r6)*(-1)**(l1+l2)
							self.H0[col0,row6,ik,jk,kk] = 1/np.sqrt(2)*(np.conjugate(r5)+1j*np.conjugate(r6))*(-1)**(l1+l2)
						# iterate iik
						iik = iik + 1
	# (p,f) SK integrals
	def set_pf_hopping_mtxel(self, Site1, Site2, tpf, siteslist, kg, Unitcell, MatrixEntry):
		# (p,f) orbitals
		l1 = 1
		l2 = 3
		# SK integrals
		[tpf0, tpf1] = tpf
		###   dimensions
		if kg.D == 0:
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r00 = 0.
			r10 = 0.
			r20 = 0.
			r01 = 0.
			r11 = 0.
			r21 = 0.
			r02 = 0.
			r12 = 0.
			r22 = 0.
			r03 = 0.
			r13 = 0.
			r23 = 0.
			r04 = 0.
			r14 = 0.
			r24 = 0.
			r05 = 0.
			r15 = 0.
			r25 = 0.
			r06 = 0.
			r16 = 0.
			r26 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r00 = 0.5*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*(5.*lcoef[2]**2 -1.)*(lcoef[2]**2 -1.)*tpf1
					r10 = 0.5*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1
					r20 = 0.5*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1
					r01 = np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1
					r11 = np.sqrt(3./8)*lcoef[0]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -1.)+2.*lcoef[0]**2)*tpf1
					r21 = np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1
					r02 = np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[1]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1
					r12 = np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1
					r22 = np.sqrt(3./8)*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -1.)+2.*lcoef[1]**2)*tpf1
					r03 = 0.5*np.sqrt(15.)*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tpf1
					r13 = 0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[0]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)-2.)*tpf1
					r23 = 0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)+2.)*tpf1
					r04 = np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]**2 *tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[1]*(3.*lcoef[2]**2 -1.)*tpf1
					r14 = np.sqrt(15.)*lcoef[0]**2 *lcoef[1]*lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -1.)*tpf1
					r24 = np.sqrt(15.)*lcoef[0]*lcoef[1]**2 *lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[2]*(3.*lcoef[1]**2 -1.)*tpf1
					r05 = np.sqrt(5./8)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf1
					r15 = np.sqrt(5./8)*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1
					r25 = np.sqrt(5./8)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2 +2.)*tpf1
					r06 = np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf1
					r16 = np.sqrt(5./8)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2 -2.)*tpf1
					r26 = np.sqrt(5./8)*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1
			for ms in [-0.5, 0.5]:
				row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
				row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
				col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
				col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
				col6 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
				col7 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
				# p0 coeffs
				self.H0[row1,col1] = r00
				self.H0[col1,row1] = np.conjugate(r00)
				self.H0[row1,col2] =-1./np.sqrt(2)*(r01+1j*r02)
				self.H0[col2,row1] =-1./np.sqrt(2)*(np.conjugate(r01)-1j*np.conjugate(r02))
				self.H0[row1,col3] = 1./np.sqrt(2)*(r01-1j*r02)
				self.H0[col3,row1] = 1./np.sqrt(2)*(np.conjugate(r01)+1j*np.conjugate(r02))
				self.H0[row1,col4] = 1./np.sqrt(2)*(r03+1j*r04)
				self.H0[col4,row1] = 1./np.sqrt(2)*(np.conjugate(r03)-1j*np.conjugate(r04))
				self.H0[row1,col5] = 1./np.sqrt(2)*(r03-1j*r04)
				self.H0[col5,row1] = 1./np.sqrt(2)*(np.conjugate(r03)+1j*np.conjugate(r04))
				self.H0[row1,col6] =-1./np.sqrt(2)*(r05+1j*r06)
				self.H0[col6,row1] =-1./np.sqrt(2)*(np.conjugate(r05)-1j*np.conjugate(r06))
				self.H0[row1,col7] = 1./np.sqrt(2)*(r05-1j*r06)
				self.H0[col7,row1] = 1./np.sqrt(2)*(np.conjugate(r05)+1j*np.conjugate(r06))
				# p1 coeffs
				self.H0[row2,col1] =-1./np.sqrt(2)*(r10-1j*r20)
				self.H0[col1,row2] =-1./np.sqrt(2)*(np.conjugate(r10)+1j*np.conjugate(r20))
				self.H0[row2,col2] = 0.5*(r11+r22+1j*(r12-r21))
				self.H0[col2,row2] = 0.5*(np.conjugate(r11+r22)-1j*np.conjugate(r12-r21))
				self.H0[row2,col3] =-0.5*(r11-r22-1j*(r12+r21))
				self.H0[col3,row2] =-0.5*(np.conjugate(r11-r22)+1j*np.conjugate(r12+r21))
				self.H0[row2,col4] =-0.5*(r13-1j*r23+1j*r14+r24)
				self.H0[col4,row2] =-0.5*(np.conjugate(r13)+1j*np.conjugate(r23)-1j*np.conjugate(r14)+np.conjugate(r24))
				self.H0[row2,col5] =-0.5*(r13-1j*r23-1j*r14-r24)
				self.H0[col5,row2] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)
				self.H0[row2,col6] = 0.5*(r15-1j*r25+1j*r16+r26)
				self.H0[col6,row2] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)
				self.H0[row2,col7] =-0.5*(r15-1j*r25-1j*r16-r26)
				self.H0[col7,row2] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)
				# p-1 coeffs
				self.H0[row3,col1] = 1./np.sqrt(2)*(r10+1j*r20)
				self.H0[col1,row3] = 1./np.sqrt(2)*np.conjugate(r10+1j*r20)
				self.H0[row3,col2] =-0.5*(r11-r22+1j*(r12+r21))
				self.H0[col2,row3] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))
				self.H0[row3,col3] = 0.5*(r11+r22+1j*(-r12+r21))
				self.H0[col3,row3] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))
				self.H0[row3,col4] = 0.5*(r13+1j*r23+1j*r14-r24)
				self.H0[col4,row3] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)
				self.H0[row3,col5] = 0.5*(r13+1j*r23-1j*r14+r24)
				self.H0[col5,row3] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)
				self.H0[row3,col6] =-0.5*(r15+1j*r25+1j*r16-r26)
				self.H0[col6,row3] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)
				self.H0[row3,col7] = 0.5*(r15+1j*r25-1j*r16+r26)
				self.H0[col7,row3] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r00 = 0.
				r10 = 0.
				r20 = 0.
				r01 = 0.
				r11 = 0.
				r21 = 0.
				r02 = 0.
				r12 = 0.
				r22 = 0.
				r03 = 0.
				r13 = 0.
				r23 = 0.
				r04 = 0.
				r14 = 0.
				r24 = 0.
				r05 = 0.
				r15 = 0.
				r25 = 0.
				r06 = 0.
				r16 = 0.
				r26 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r00 = (0.5*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*(5.*lcoef[2]**2 -1.)*(lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r00
						r10 = (0.5*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r10
						r20 = (0.5*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r20
						r01 = (np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r01
						r11 = (np.sqrt(3./8)*lcoef[0]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -1.)+2.*lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r11
						r21 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r21
						r02 = (np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[1]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r02
						r12 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r12
						r22 = (np.sqrt(3./8)*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -1.)+2.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r22
						r03 = (0.5*np.sqrt(15.)*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r03
						r13 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[0]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)-2.)*tpf1)*cmath.exp(1j*kR) + r13
						r23 = (0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)+2.)*tpf1)*cmath.exp(1j*kR) + r23
						r04 = (np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]**2 *tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[1]*(3.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r04
						r14 = (np.sqrt(15.)*lcoef[0]**2 *lcoef[1]*lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r14
						r24 = (np.sqrt(15.)*lcoef[0]*lcoef[1]**2 *lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[2]*(3.*lcoef[1]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r24
						r05 = (np.sqrt(5./8)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r05
						r15 = (np.sqrt(5./8)*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r15
						r25 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2 +2.)*tpf1)*cmath.exp(1j*kR) + r25
						r06 = (np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r06
						r16 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2 -2.)*tpf1)*cmath.exp(1j*kR) + r16
						r26 = (np.sqrt(5./8)*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r26
				for ms in [-0.5, 0.5]:
					row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
					row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
					col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
					col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
					col6 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
					col7 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
					# p0 coeffs
					self.H0[row1,col1,ik] = r00
					self.H0[col1,row1,ik] = np.conjugate(r00)
					self.H0[row1,col2,ik] =-1./np.sqrt(2)*(r01+1j*r02)
					self.H0[col2,row1,ik] =-1./np.sqrt(2)*(np.conjugate(r01)-1j*np.conjugate(r02))
					self.H0[row1,col3,ik] = 1./np.sqrt(2)*(r01-1j*r02)
					self.H0[col3,row1,ik] = 1./np.sqrt(2)*(np.conjugate(r01)+1j*np.conjugate(r02))
					self.H0[row1,col4,ik] = 1./np.sqrt(2)*(r03+1j*r04)
					self.H0[col4,row1,ik] = 1./np.sqrt(2)*(np.conjugate(r03)-1j*np.conjugate(r04))
					self.H0[row1,col5,ik] = 1./np.sqrt(2)*(r03-1j*r04)
					self.H0[col5,row1,ik] = 1./np.sqrt(2)*(np.conjugate(r03)+1j*np.conjugate(r04))
					self.H0[row1,col6,ik] =-1./np.sqrt(2)*(r05+1j*r06)
					self.H0[col6,row1,ik] =-1./np.sqrt(2)*(np.conjugate(r05)-1j*np.conjugate(r06))
					self.H0[row1,col7,ik] = 1./np.sqrt(2)*(r05-1j*r06)
					self.H0[col7,row1,ik] = 1./np.sqrt(2)*(np.conjugate(r05)+1j*np.conjugate(r06))
					# p1 coeffs
					self.H0[row2,col1,ik] =-1./np.sqrt(2)*(r10-1j*r20)
					self.H0[col1,row2,ik] =-1./np.sqrt(2)*(np.conjugate(r10)+1j*np.conjugate(r20))
					self.H0[row2,col2,ik] = 0.5*(r11+r22+1j*(r12-r21))
					self.H0[col2,row2,ik] = 0.5*(np.conjugate(r11+r22)-1j*np.conjugate(r12-r21))
					self.H0[row2,col3,ik] =-0.5*(r11-r22-1j*(r12+r21))
					self.H0[col3,row2,ik] =-0.5*(np.conjugate(r11-r22)+1j*np.conjugate(r12+r21))
					self.H0[row2,col4,ik] =-0.5*(r13-1j*r23+1j*r14+r24)
					self.H0[col4,row2,ik] =-0.5*(np.conjugate(r13)+1j*np.conjugate(r23)-1j*np.conjugate(r14)+np.conjugate(r24))
					self.H0[row2,col5,ik] =-0.5*(r13-1j*r23-1j*r14-r24)
					self.H0[col5,row2,ik] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)
					self.H0[row2,col6,ik] = 0.5*(r15-1j*r25+1j*r16+r26)
					self.H0[col6,row2,ik] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)
					self.H0[row2,col7,ik] =-0.5*(r15-1j*r25-1j*r16-r26)
					self.H0[col7,row2,ik] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)
					# p-1 coeffs
					self.H0[row3,col1,ik] = 1./np.sqrt(2)*(r10+1j*r20)
					self.H0[col1,row3,ik] = 1./np.sqrt(2)*np.conjugate(r10+1j*r20)
					self.H0[row3,col2,ik] =-0.5*(r11-r22+1j*(r12+r21))
					self.H0[col2,row3,ik] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))
					self.H0[row3,col3,ik] = 0.5*(r11+r22+1j*(-r12+r21))
					self.H0[col3,row3,ik] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))
					self.H0[row3,col4,ik] = 0.5*(r13+1j*r23+1j*r14-r24)
					self.H0[col4,row3,ik] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)
					self.H0[row3,col5,ik] = 0.5*(r13+1j*r23-1j*r14+r24)
					self.H0[col5,row3,ik] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)
					self.H0[row3,col6,ik] =-0.5*(r15+1j*r25+1j*r16-r26)
					self.H0[col6,row3,ik] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)
					self.H0[row3,col7,ik] = 0.5*(r15+1j*r25-1j*r16+r26)
					self.H0[col7,row3,ik] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r00 = 0.
					r10 = 0.
					r20 = 0.
					r01 = 0.
					r11 = 0.
					r21 = 0.
					r02 = 0.
					r12 = 0.
					r22 = 0.
					r03 = 0.
					r13 = 0.
					r23 = 0.
					r04 = 0.
					r14 = 0.
					r24 = 0.
					r05 = 0.
					r15 = 0.
					r25 = 0.
					r06 = 0.
					r16 = 0.
					r26 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r00 = (0.5*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*(5.*lcoef[2]**2 -1.)*(lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r00
							r10 = (0.5*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r10
							r20 = (0.5*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r20
							r01 = (np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r01
							r11 = (np.sqrt(3./8)*lcoef[0]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -1.)+2.*lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r11
							r21 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r21
							r02 = (np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[1]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r02
							r12 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r12
							r22 = (np.sqrt(3./8)*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -1.)+2.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r22
							r03 = (0.5*np.sqrt(15.)*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r03
							r13 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[0]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)-2.)*tpf1)*cmath.exp(1j*kR) + r13
							r23 = (0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)+2.)*tpf1)*cmath.exp(1j*kR) + r23
							r04 = (np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]**2 *tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[1]*(3.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r04
							r14 = (np.sqrt(15.)*lcoef[0]**2 *lcoef[1]*lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r14
							r24 = (np.sqrt(15.)*lcoef[0]*lcoef[1]**2 *lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[2]*(3.*lcoef[1]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r24
							r05 = (np.sqrt(5./8)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r05
							r15 = (np.sqrt(5./8)*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r15
							r25 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2 +2.)*tpf1)*cmath.exp(1j*kR) + r25
							r06 = (np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r06
							r16 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2 -2.)*tpf1)*cmath.exp(1j*kR) + r16
							r26 = (np.sqrt(5./8)*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r26
					for ms in [-0.5, 0.5]:
						row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
						row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
						col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
						col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
						col6 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
						col7 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
						# p0 coeffs
						self.H0[row1,col1,ik,jk] = r00
						self.H0[col1,row1,ik,jk] = np.conjugate(r00)
						self.H0[row1,col2,ik,jk] =-1./np.sqrt(2)*(r01+1j*r02)
						self.H0[col2,row1,ik,jk] =-1./np.sqrt(2)*(np.conjugate(r01)-1j*np.conjugate(r02))
						self.H0[row1,col3,ik,jk] = 1./np.sqrt(2)*(r01-1j*r02)
						self.H0[col3,row1,ik,jk] = 1./np.sqrt(2)*(np.conjugate(r01)+1j*np.conjugate(r02))
						self.H0[row1,col4,ik,jk] = 1./np.sqrt(2)*(r03+1j*r04)
						self.H0[col4,row1,ik,jk] = 1./np.sqrt(2)*(np.conjugate(r03)-1j*np.conjugate(r04))
						self.H0[row1,col5,ik,jk] = 1./np.sqrt(2)*(r03-1j*r04)
						self.H0[col5,row1,ik,jk] = 1./np.sqrt(2)*(np.conjugate(r03)+1j*np.conjugate(r04))
						self.H0[row1,col6,ik,jk] =-1./np.sqrt(2)*(r05+1j*r06)
						self.H0[col6,row1,ik,jk] =-1./np.sqrt(2)*(np.conjugate(r05)-1j*np.conjugate(r06))
						self.H0[row1,col7,ik,jk] = 1./np.sqrt(2)*(r05-1j*r06)
						self.H0[col7,row1,ik,jk] = 1./np.sqrt(2)*(np.conjugate(r05)+1j*np.conjugate(r06))
						# p1 coeffs
						self.H0[row2,col1,ik,jk] =-1./np.sqrt(2)*(r10-1j*r20)
						self.H0[col1,row2,ik,jk] =-1./np.sqrt(2)*(np.conjugate(r10)+1j*np.conjugate(r20))
						self.H0[row2,col2,ik,jk] = 0.5*(r11+r22+1j*(r12-r21))
						self.H0[col2,row2,ik,jk] = 0.5*(np.conjugate(r11+r22)-1j*np.conjugate(r12-r21))
						self.H0[row2,col3,ik,jk] =-0.5*(r11-r22-1j*(r12+r21))
						self.H0[col3,row2,ik,jk] =-0.5*(np.conjugate(r11-r22)+1j*np.conjugate(r12+r21))
						self.H0[row2,col4,ik,jk] =-0.5*(r13-1j*r23+1j*r14+r24)
						self.H0[col4,row2,ik,jk] =-0.5*(np.conjugate(r13)+1j*np.conjugate(r23)-1j*np.conjugate(r14)+np.conjugate(r24))
						self.H0[row2,col5,ik,jk] =-0.5*(r13-1j*r23-1j*r14-r24)
						self.H0[col5,row2,ik,jk] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)
						self.H0[row2,col6,ik,jk] = 0.5*(r15-1j*r25+1j*r16+r26)
						self.H0[col6,row2,ik,jk] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)
						self.H0[row2,col7,ik,jk] =-0.5*(r15-1j*r25-1j*r16-r26)
						self.H0[col7,row2,ik,jk] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)
						# p-1 coeffs
						self.H0[row3,col1,ik,jk] = 1./np.sqrt(2)*(r10+1j*r20)
						self.H0[col1,row3,ik,jk] = 1./np.sqrt(2)*np.conjugate(r10+1j*r20)
						self.H0[row3,col2,ik,jk] =-0.5*(r11-r22+1j*(r12+r21))
						self.H0[col2,row3,ik,jk] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))
						self.H0[row3,col3,ik,jk] = 0.5*(r11+r22+1j*(-r12+r21))
						self.H0[col3,row3,ik,jk] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))
						self.H0[row3,col4,ik,jk] = 0.5*(r13+1j*r23+1j*r14-r24)
						self.H0[col4,row3,ik,jk] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)
						self.H0[row3,col5,ik,jk] = 0.5*(r13+1j*r23-1j*r14+r24)
						self.H0[col5,row3,ik,jk] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)
						self.H0[row3,col6,ik,jk] =-0.5*(r15+1j*r25+1j*r16-r26)
						self.H0[col6,row3,ik,jk] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)
						self.H0[row3,col7,ik,jk] = 0.5*(r15+1j*r25-1j*r16+r26)
						self.H0[col7,row3,ik,jk] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r00 = 0.
						r10 = 0.
						r20 = 0.
						r01 = 0.
						r11 = 0.
						r21 = 0.
						r02 = 0.
						r12 = 0.
						r22 = 0.
						r03 = 0.
						r13 = 0.
						r23 = 0.
						r04 = 0.
						r14 = 0.
						r24 = 0.
						r05 = 0.
						r15 = 0.
						r25 = 0.
						r06 = 0.
						r16 = 0.
						r26 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r00 = (0.5*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*(5.*lcoef[2]**2 -1.)*(lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r00
								r10 = (0.5*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r10
								r20 = (0.5*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r20
								r01 = (np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r01
								r11 = (np.sqrt(3./8)*lcoef[0]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -1.)+2.*lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r11
								r21 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r21
								r02 = (np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[1]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r02
								r12 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r12
								r22 = (np.sqrt(3./8)*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -1.)+2.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r22
								r03 = (0.5*np.sqrt(15.)*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r03
								r13 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[0]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)-2.)*tpf1)*cmath.exp(1j*kR) + r13
								r23 = (0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)+2.)*tpf1)*cmath.exp(1j*kR) + r23
								r04 = (np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]**2 *tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[1]*(3.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r04
								r14 = (np.sqrt(15.)*lcoef[0]**2 *lcoef[1]*lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r14
								r24 = (np.sqrt(15.)*lcoef[0]*lcoef[1]**2 *lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[2]*(3.*lcoef[1]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r24
								r05 = (np.sqrt(5./8)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r05
								r15 = (np.sqrt(5./8)*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r15
								r25 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2 +2.)*tpf1)*cmath.exp(1j*kR) + r25
								r06 = (np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r06
								r16 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2 -2.)*tpf1)*cmath.exp(1j*kR) + r16
								r26 = (np.sqrt(5./8)*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r26
						for ms in [-0.5, 0.5]:
							row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
							row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
							col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
							col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
							col6 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
							col7 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
							# p0 coeffs
							self.H0[row1,col1,ik,jk,kk] = r00
							self.H0[col1,row1,ik,jk,kk] = np.conjugate(r00)
							self.H0[row1,col2,ik,jk,kk] =-1./np.sqrt(2)*(r01+1j*r02)
							self.H0[col2,row1,ik,jk,kk] =-1./np.sqrt(2)*(np.conjugate(r01)-1j*np.conjugate(r02))
							self.H0[row1,col3,ik,jk,kk] = 1./np.sqrt(2)*(r01-1j*r02)
							self.H0[col3,row1,ik,jk,kk] = 1./np.sqrt(2)*(np.conjugate(r01)+1j*np.conjugate(r02))
							self.H0[row1,col4,ik,jk,kk] = 1./np.sqrt(2)*(r03+1j*r04)
							self.H0[col4,row1,ik,jk,kk] = 1./np.sqrt(2)*(np.conjugate(r03)-1j*np.conjugate(r04))
							self.H0[row1,col5,ik,jk,kk] = 1./np.sqrt(2)*(r03-1j*r04)
							self.H0[col5,row1,ik,jk,kk] = 1./np.sqrt(2)*(np.conjugate(r03)+1j*np.conjugate(r04))
							self.H0[row1,col6,ik,jk,kk] =-1./np.sqrt(2)*(r05+1j*r06)
							self.H0[col6,row1,ik,jk,kk] =-1./np.sqrt(2)*(np.conjugate(r05)-1j*np.conjugate(r06))
							self.H0[row1,col7,ik,jk,kk] = 1./np.sqrt(2)*(r05-1j*r06)
							self.H0[col7,row1,ik,jk,kk] = 1./np.sqrt(2)*(np.conjugate(r05)+1j*np.conjugate(r06))
							# p1 coeffs
							self.H0[row2,col1,ik,jk,kk] =-1./np.sqrt(2)*(r10-1j*r20)
							self.H0[col1,row2,ik,jk,kk] =-1./np.sqrt(2)*(np.conjugate(r10)+1j*np.conjugate(r20))
							self.H0[row2,col2,ik,jk,kk] = 0.5*(r11+r22+1j*(r12-r21))
							self.H0[col2,row2,ik,jk,kk] = 0.5*(np.conjugate(r11+r22)-1j*np.conjugate(r12-r21))
							self.H0[row2,col3,ik,jk,kk] =-0.5*(r11-r22-1j*(r12+r21))
							self.H0[col3,row2,ik,jk,kk] =-0.5*(np.conjugate(r11-r22)+1j*np.conjugate(r12+r21))
							self.H0[row2,col4,ik,jk,kk] =-0.5*(r13-1j*r23+1j*r14+r24)
							self.H0[col4,row2,ik,jk,kk] =-0.5*(np.conjugate(r13)+1j*np.conjugate(r23)-1j*np.conjugate(r14)+np.conjugate(r24))
							self.H0[row2,col5,ik,jk,kk] =-0.5*(r13-1j*r23-1j*r14-r24)
							self.H0[col5,row2,ik,jk,kk] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)
							self.H0[row2,col6,ik,jk,kk] = 0.5*(r15-1j*r25+1j*r16+r26)
							self.H0[col6,row2,ik,jk,kk] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)
							self.H0[row2,col7,ik,jk,kk] =-0.5*(r15-1j*r25-1j*r16-r26)
							self.H0[col7,row2,ik,jk,kk] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)
							# p-1 coeffs
							self.H0[row3,col1,ik,jk,kk] = 1./np.sqrt(2)*(r10+1j*r20)
							self.H0[col1,row3,ik,jk,kk] = 1./np.sqrt(2)*np.conjugate(r10+1j*r20)
							self.H0[row3,col2,ik,jk,kk] =-0.5*(r11-r22+1j*(r12+r21))
							self.H0[col2,row3,ik,jk,kk] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))
							self.H0[row3,col3,ik,jk,kk] = 0.5*(r11+r22+1j*(-r12+r21))
							self.H0[col3,row3,ik,jk,kk] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))
							self.H0[row3,col4,ik,jk,kk] = 0.5*(r13+1j*r23+1j*r14-r24)
							self.H0[col4,row3,ik,jk,kk] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)
							self.H0[row3,col5,ik,jk,kk] = 0.5*(r13+1j*r23-1j*r14+r24)
							self.H0[col5,row3,ik,jk,kk] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)
							self.H0[row3,col6,ik,jk,kk] =-0.5*(r15+1j*r25+1j*r16-r26)
							self.H0[col6,row3,ik,jk,kk] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)
							self.H0[row3,col7,ik,jk,kk] = 0.5*(r15+1j*r25-1j*r16+r26)
							self.H0[col7,row3,ik,jk,kk] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)
						# iterate iik
						iik = iik + 1
	# (f,p) SK integrals
	def set_fp_hopping_mtxel(self, Site1, Site2, tpf, siteslist, kg, Unitcell, MatrixEntry):
		# (f,p) orbitals
		l1 = 3
		l2 = 1
		# SK integrals
		[tpf0, tpf1] = tpf
		###   dimensions
		if kg.D == 0:
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r00 = 0.
			r10 = 0.
			r20 = 0.
			r01 = 0.
			r11 = 0.
			r21 = 0.
			r02 = 0.
			r12 = 0.
			r22 = 0.
			r03 = 0.
			r13 = 0.
			r23 = 0.
			r04 = 0.
			r14 = 0.
			r24 = 0.
			r05 = 0.
			r15 = 0.
			r25 = 0.
			r06 = 0.
			r16 = 0.
			r26 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r00 = 0.5*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*(5.*lcoef[2]**2 -1.)*(lcoef[2]**2 -1.)*tpf1
					r10 = 0.5*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1
					r20 = 0.5*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1
					r01 = np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1
					r11 = np.sqrt(3./8)*lcoef[0]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -1.)+2.*lcoef[0]**2)*tpf1
					r21 = np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1
					r02 = np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[1]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1
					r12 = np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1
					r22 = np.sqrt(3./8)*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -1.)+2.*lcoef[1]**2)*tpf1
					r03 = 0.5*np.sqrt(15.)*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tpf1
					r13 = 0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[0]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)-2.)*tpf1
					r23 = 0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)+2.)*tpf1
					r04 = np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]**2 *tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[1]*(3.*lcoef[2]**2 -1.)*tpf1
					r14 = np.sqrt(15.)*lcoef[0]**2 *lcoef[1]*lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -1.)*tpf1
					r24 = np.sqrt(15.)*lcoef[0]*lcoef[1]**2 *lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[2]*(3.*lcoef[1]**2 -1.)*tpf1
					r05 = np.sqrt(5./8)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf1
					r15 = np.sqrt(5./8)*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1
					r25 = np.sqrt(5./8)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2 +2.)*tpf1
					r06 = np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf1
					r16 = np.sqrt(5./8)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2 -2.)*tpf1
					r26 = np.sqrt(5./8)*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1
			for ms in [-0.5, 0.5]:
				row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
				row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
				row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
				row6 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
				row7 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
				col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
				col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				# p0 coeffs
				self.H0[row1,col1] = r00*(-1)**(l1+l2)
				self.H0[col1,row1] = np.conjugate(r00)*(-1)**(l1+l2)
				self.H0[row2,col1] =-1./np.sqrt(2)*(r01+1j*r02)*(-1)**(l1+l2)
				self.H0[col1,row2] =-1./np.sqrt(2)*(np.conjugate(r01)-1j*np.conjugate(r02))*(-1)**(l1+l2)
				self.H0[row3,col1] = 1./np.sqrt(2)*(r01-1j*r02)*(-1)**(l1+l2)
				self.H0[col1,row3] = 1./np.sqrt(2)*(np.conjugate(r01)+1j*np.conjugate(r02))*(-1)**(l1+l2)
				self.H0[row4,col1] = 1./np.sqrt(2)*(r03+1j*r04)*(-1)**(l1+l2)
				self.H0[col1,row4] = 1./np.sqrt(2)*(np.conjugate(r03)-1j*np.conjugate(r04))*(-1)**(l1+l2)
				self.H0[row5,col1] = 1./np.sqrt(2)*(r03-1j*r04)*(-1)**(l1+l2)
				self.H0[col1,row5] = 1./np.sqrt(2)*(np.conjugate(r03)+1j*np.conjugate(r04))*(-1)**(l1+l2)
				self.H0[row6,col1] =-1./np.sqrt(2)*(r05+1j*r06)*(-1)**(l1+l2)
				self.H0[col1,row6] =-1./np.sqrt(2)*(np.conjugate(r05)-1j*np.conjugate(r06))*(-1)**(l1+l2)
				self.H0[row7,col1] = 1./np.sqrt(2)*(r05-1j*r06)*(-1)**(l1+l2)
				self.H0[col1,row7] = 1./np.sqrt(2)*(np.conjugate(r05)+1j*np.conjugate(r06))*(-1)**(l1+l2)
				# p1 coeffs
				self.H0[row1,col2] =-1./np.sqrt(2)*(r10-1j*r20)*(-1)**(l1+l2)
				self.H0[col2,row1] =-1./np.sqrt(2)*(np.conjugate(r10)+1j*np.conjugate(r20))*(-1)**(l1+l2)
				self.H0[row2,col2] = 0.5*(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
				self.H0[col2,row2] = 0.5*(np.conjugate(r11+r22)-1j*np.conjugate(r12-r21))*(-1)**(l1+l2)
				self.H0[row3,col2] =-0.5*(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
				self.H0[col2,row3] =-0.5*(np.conjugate(r11-r22)+1j*np.conjugate(r12+r21))*(-1)**(l1+l2)
				self.H0[row4,col2] =-0.5*(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
				self.H0[col2,row4] =-0.5*(np.conjugate(r13)+1j*np.conjugate(r23)-1j*np.conjugate(r14)+np.conjugate(r24))*(-1)**(l1+l2)
				self.H0[row5,col2] =-0.5*(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
				self.H0[col2,row5] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
				self.H0[row6,col2] = 0.5*(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
				self.H0[col2,row6] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
				self.H0[row7,col2] =-0.5*(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
				self.H0[col2,row7] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
				# p-1 coeffs
				self.H0[row1,col3] = 1./np.sqrt(2)*(r10+1j*r20)*(-1)**(l1+l2)
				self.H0[col3,row1] = 1./np.sqrt(2)*np.conjugate(r10+1j*r20)*(-1)**(l1+l2)
				self.H0[row2,col3] =-0.5*(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
				self.H0[col3,row2] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
				self.H0[row3,col3] = 0.5*(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
				self.H0[col3,row3] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
				self.H0[row4,col3] = 0.5*(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
				self.H0[col3,row4] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
				self.H0[row5,col3] = 0.5*(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
				self.H0[col3,row5] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
				self.H0[row6,col3] =-0.5*(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
				self.H0[col3,row6] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
				self.H0[row7,col3] = 0.5*(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
				self.H0[col3,row7] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r00 = 0.
				r10 = 0.
				r20 = 0.
				r01 = 0.
				r11 = 0.
				r21 = 0.
				r02 = 0.
				r12 = 0.
				r22 = 0.
				r03 = 0.
				r13 = 0.
				r23 = 0.
				r04 = 0.
				r14 = 0.
				r24 = 0.
				r05 = 0.
				r15 = 0.
				r25 = 0.
				r06 = 0.
				r16 = 0.
				r26 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r00 = (0.5*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*(5.*lcoef[2]**2 -1.)*(lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r00
						r10 = (0.5*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r10
						r20 = (0.5*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r20
						r01 = (np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r01
						r11 = (np.sqrt(3./8)*lcoef[0]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -1.)+2.*lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r11
						r21 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r21
						r02 = (np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[1]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r02
						r12 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r12
						r22 = (np.sqrt(3./8)*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -1.)+2.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r22
						r03 = (0.5*np.sqrt(15.)*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r03
						r13 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[0]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)-2.)*tpf1)*cmath.exp(1j*kR) + r13
						r23 = (0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)+2.)*tpf1)*cmath.exp(1j*kR) + r23
						r04 = (np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]**2 *tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[1]*(3.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r04
						r14 = (np.sqrt(15.)*lcoef[0]**2 *lcoef[1]*lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r14
						r24 = (np.sqrt(15.)*lcoef[0]*lcoef[1]**2 *lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[2]*(3.*lcoef[1]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r24
						r05 = (np.sqrt(5./8)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r05
						r15 = (np.sqrt(5./8)*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r15
						r25 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2 +2.)*tpf1)*cmath.exp(1j*kR) + r25
						r06 = (np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r06
						r16 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2 -2.)*tpf1)*cmath.exp(1j*kR) + r16
						r26 = (np.sqrt(5./8)*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r26
				for ms in [-0.5, 0.5]:
					row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
					row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
					row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
					row6 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
					row7 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
					col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
					col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					# p0 coeffs
					self.H0[row1,col1,ik] = r00*(-1)**(l1+l2)
					self.H0[col1,row1,ik] = np.conjugate(r00)*(-1)**(l1+l2)
					self.H0[row2,col1,ik] =-1./np.sqrt(2)*(r01+1j*r02)*(-1)**(l1+l2)
					self.H0[col1,row2,ik] =-1./np.sqrt(2)*(np.conjugate(r01)-1j*np.conjugate(r02))*(-1)**(l1+l2)
					self.H0[row3,col1,ik] = 1./np.sqrt(2)*(r01-1j*r02)*(-1)**(l1+l2)
					self.H0[col1,row3,ik] = 1./np.sqrt(2)*(np.conjugate(r01)+1j*np.conjugate(r02))*(-1)**(l1+l2)
					self.H0[row4,col1,ik] = 1./np.sqrt(2)*(r03+1j*r04)*(-1)**(l1+l2)
					self.H0[col1,row4,ik] = 1./np.sqrt(2)*(np.conjugate(r03)-1j*np.conjugate(r04))*(-1)**(l1+l2)
					self.H0[row5,col1,ik] = 1./np.sqrt(2)*(r03-1j*r04)*(-1)**(l1+l2)
					self.H0[col1,row5,ik] = 1./np.sqrt(2)*(np.conjugate(r03)+1j*np.conjugate(r04))*(-1)**(l1+l2)
					self.H0[row6,col1,ik] =-1./np.sqrt(2)*(r05+1j*r06)*(-1)**(l1+l2)
					self.H0[col1,row6,ik] =-1./np.sqrt(2)*(np.conjugate(r05)-1j*np.conjugate(r06))*(-1)**(l1+l2)
					self.H0[row7,col1,ik] = 1./np.sqrt(2)*(r05-1j*r06)*(-1)**(l1+l2)
					self.H0[col1,row7,ik] = 1./np.sqrt(2)*(np.conjugate(r05)+1j*np.conjugate(r06))*(-1)**(l1+l2)
					# p1 coeffs
					self.H0[row1,col2,ik] =-1./np.sqrt(2)*(r10-1j*r20)*(-1)**(l1+l2)
					self.H0[col2,row1,ik] =-1./np.sqrt(2)*(np.conjugate(r10)+1j*np.conjugate(r20))*(-1)**(l1+l2)
					self.H0[row2,col2,ik] = 0.5*(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
					self.H0[col2,row2,ik] = 0.5*(np.conjugate(r11+r22)-1j*np.conjugate(r12-r21))*(-1)**(l1+l2)
					self.H0[row3,col2,ik] =-0.5*(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
					self.H0[col2,row3,ik] =-0.5*(np.conjugate(r11-r22)+1j*np.conjugate(r12+r21))*(-1)**(l1+l2)
					self.H0[row4,col2,ik] =-0.5*(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
					self.H0[col2,row4,ik] =-0.5*(np.conjugate(r13)+1j*np.conjugate(r23)-1j*np.conjugate(r14)+np.conjugate(r24))*(-1)**(l1+l2)
					self.H0[row5,col2,ik] =-0.5*(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
					self.H0[col2,row5,ik] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
					self.H0[row6,col2,ik] = 0.5*(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
					self.H0[col2,row6,ik] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
					self.H0[row7,col2,ik] =-0.5*(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
					self.H0[col2,row7,ik] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
					# p-1 coeffs
					self.H0[row1,col3,ik] = 1./np.sqrt(2)*(r10+1j*r20)*(-1)**(l1+l2)
					self.H0[col3,row1,ik] = 1./np.sqrt(2)*np.conjugate(r10+1j*r20)*(-1)**(l1+l2)
					self.H0[row2,col3,ik] =-0.5*(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
					self.H0[col3,row2,ik] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
					self.H0[row3,col3,ik] = 0.5*(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
					self.H0[col3,row3,ik] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
					self.H0[row4,col3,ik] = 0.5*(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
					self.H0[col3,row4,ik] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
					self.H0[row5,col3,ik] = 0.5*(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
					self.H0[col3,row5,ik] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
					self.H0[row6,col3,ik] =-0.5*(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
					self.H0[col3,row6,ik] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
					self.H0[row7,col3,ik] = 0.5*(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
					self.H0[col3,row7,ik] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r00 = 0.
					r10 = 0.
					r20 = 0.
					r01 = 0.
					r11 = 0.
					r21 = 0.
					r02 = 0.
					r12 = 0.
					r22 = 0.
					r03 = 0.
					r13 = 0.
					r23 = 0.
					r04 = 0.
					r14 = 0.
					r24 = 0.
					r05 = 0.
					r15 = 0.
					r25 = 0.
					r06 = 0.
					r16 = 0.
					r26 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r00 = (0.5*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*(5.*lcoef[2]**2 -1.)*(lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r00
							r10 = (0.5*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r10
							r20 = (0.5*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r20
							r01 = (np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r01
							r11 = (np.sqrt(3./8)*lcoef[0]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -1.)+2.*lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r11
							r21 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r21
							r02 = (np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[1]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r02
							r12 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r12
							r22 = (np.sqrt(3./8)*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -1.)+2.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r22
							r03 = (0.5*np.sqrt(15.)*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r03
							r13 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[0]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)-2.)*tpf1)*cmath.exp(1j*kR) + r13
							r23 = (0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)+2.)*tpf1)*cmath.exp(1j*kR) + r23
							r04 = (np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]**2 *tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[1]*(3.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r04
							r14 = (np.sqrt(15.)*lcoef[0]**2 *lcoef[1]*lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r14
							r24 = (np.sqrt(15.)*lcoef[0]*lcoef[1]**2 *lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[2]*(3.*lcoef[1]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r24
							r05 = (np.sqrt(5./8)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r05
							r15 = (np.sqrt(5./8)*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r15
							r25 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2 +2.)*tpf1)*cmath.exp(1j*kR) + r25
							r06 = (np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r06
							r16 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2 -2.)*tpf1)*cmath.exp(1j*kR) + r16
							r26 = (np.sqrt(5./8)*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r26
					for ms in [-0.5, 0.5]:
						row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
						row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
						row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
						row6 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
						row7 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
						col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
						col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						# p0 coeffs
						self.H0[row1,col1,ik,jk] = r00*(-1)**(l1+l2)
						self.H0[col1,row1,ik,jk] = np.conjugate(r00)*(-1)**(l1+l2)
						self.H0[row2,col1,ik,jk] =-1./np.sqrt(2)*(r01+1j*r02)*(-1)**(l1+l2)
						self.H0[col1,row2,ik,jk] =-1./np.sqrt(2)*(np.conjugate(r01)-1j*np.conjugate(r02))*(-1)**(l1+l2)
						self.H0[row3,col1,ik,jk] = 1./np.sqrt(2)*(r01-1j*r02)*(-1)**(l1+l2)
						self.H0[col1,row3,ik,jk] = 1./np.sqrt(2)*(np.conjugate(r01)+1j*np.conjugate(r02))*(-1)**(l1+l2)
						self.H0[row4,col1,ik,jk] = 1./np.sqrt(2)*(r03+1j*r04)*(-1)**(l1+l2)
						self.H0[col1,row4,ik,jk] = 1./np.sqrt(2)*(np.conjugate(r03)-1j*np.conjugate(r04))*(-1)**(l1+l2)
						self.H0[row5,col1,ik,jk] = 1./np.sqrt(2)*(r03-1j*r04)*(-1)**(l1+l2)
						self.H0[col1,row5,ik,jk] = 1./np.sqrt(2)*(np.conjugate(r03)+1j*np.conjugate(r04))*(-1)**(l1+l2)
						self.H0[row6,col1,ik,jk] =-1./np.sqrt(2)*(r05+1j*r06)*(-1)**(l1+l2)
						self.H0[col1,row6,ik,jk] =-1./np.sqrt(2)*(np.conjugate(r05)-1j*np.conjugate(r06))*(-1)**(l1+l2)
						self.H0[row7,col1,ik,jk] = 1./np.sqrt(2)*(r05-1j*r06)*(-1)**(l1+l2)
						self.H0[col1,row7,ik,jk] = 1./np.sqrt(2)*(np.conjugate(r05)+1j*np.conjugate(r06))*(-1)**(l1+l2)
						# p1 coeffs
						self.H0[row1,col2,ik,jk] =-1./np.sqrt(2)*(r10-1j*r20)*(-1)**(l1+l2)
						self.H0[col2,row1,ik,jk] =-1./np.sqrt(2)*(np.conjugate(r10)+1j*np.conjugate(r20))*(-1)**(l1+l2)
						self.H0[row2,col2,ik,jk] = 0.5*(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
						self.H0[col2,row2,ik,jk] = 0.5*(np.conjugate(r11+r22)-1j*np.conjugate(r12-r21))*(-1)**(l1+l2)
						self.H0[row3,col2,ik,jk] =-0.5*(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
						self.H0[col2,row3,ik,jk] =-0.5*(np.conjugate(r11-r22)+1j*np.conjugate(r12+r21))*(-1)**(l1+l2)
						self.H0[row4,col2,ik,jk] =-0.5*(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
						self.H0[col2,row4,ik,jk] =-0.5*(np.conjugate(r13)+1j*np.conjugate(r23)-1j*np.conjugate(r14)+np.conjugate(r24))*(-1)**(l1+l2)
						self.H0[row5,col2,ik,jk] =-0.5*(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
						self.H0[col2,row5,ik,jk] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
						self.H0[row6,col2,ik,jk] = 0.5*(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
						self.H0[col2,row6,ik,jk] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
						self.H0[row7,col2,ik,jk] =-0.5*(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
						self.H0[col2,row7,ik,jk] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
						# p-1 coeffs
						self.H0[row1,col3,ik,jk] = 1./np.sqrt(2)*(r10+1j*r20)*(-1)**(l1+l2)
						self.H0[col3,row1,ik,jk] = 1./np.sqrt(2)*np.conjugate(r10+1j*r20)*(-1)**(l1+l2)
						self.H0[row2,col3,ik,jk] =-0.5*(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
						self.H0[col3,row2,ik,jk] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
						self.H0[row3,col3,ik,jk] = 0.5*(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
						self.H0[col3,row3,ik,jk] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
						self.H0[row4,col3,ik,jk] = 0.5*(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
						self.H0[col3,row4,ik,jk] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
						self.H0[row5,col3,ik,jk] = 0.5*(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
						self.H0[col3,row5,ik,jk] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
						self.H0[row6,col3,ik,jk] =-0.5*(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
						self.H0[col3,row6,ik,jk] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
						self.H0[row7,col3,ik,jk] = 0.5*(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
						self.H0[col3,row7,ik,jk] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r00 = 0.
						r10 = 0.
						r20 = 0.
						r01 = 0.
						r11 = 0.
						r21 = 0.
						r02 = 0.
						r12 = 0.
						r22 = 0.
						r03 = 0.
						r13 = 0.
						r23 = 0.
						r04 = 0.
						r14 = 0.
						r24 = 0.
						r05 = 0.
						r15 = 0.
						r25 = 0.
						r06 = 0.
						r16 = 0.
						r26 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r00 = (0.5*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*(5.*lcoef[2]**2 -1.)*(lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r00
								r10 = (0.5*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r10
								r20 = (0.5*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tpf0-np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r20
								r01 = (np.sqrt(3./8)*lcoef[0]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r01
								r11 = (np.sqrt(3./8)*lcoef[0]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -1.)+2.*lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r11
								r21 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r21
								r02 = (np.sqrt(3./8)*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[1]*lcoef[2]*(15.*lcoef[2]**2 -11.)*tpf1)*cmath.exp(1j*kR) + r02
								r12 = (np.sqrt(3./8)*lcoef[0]*lcoef[1]*(5.*lcoef[2]**2 -1.)*tpf0-1./4*lcoef[0]*lcoef[1]*(15.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r12
								r22 = (np.sqrt(3./8)*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tpf0-1./4*((5.*lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -1.)+2.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r22
								r03 = (0.5*np.sqrt(15.)*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r03
								r13 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[0]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)-2.)*tpf1)*cmath.exp(1j*kR) + r13
								r23 = (0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tpf0-np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)+2.)*tpf1)*cmath.exp(1j*kR) + r23
								r04 = (np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]**2 *tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[1]*(3.*lcoef[2]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r04
								r14 = (np.sqrt(15.)*lcoef[0]**2 *lcoef[1]*lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r14
								r24 = (np.sqrt(15.)*lcoef[0]*lcoef[1]**2 *lcoef[2]*tpf0-np.sqrt(5./2)*lcoef[0]*lcoef[2]*(3.*lcoef[1]**2 -1.)*tpf1)*cmath.exp(1j*kR) + r24
								r05 = (np.sqrt(5./8)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r05
								r15 = (np.sqrt(5./8)*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r15
								r25 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2 +2.)*tpf1)*cmath.exp(1j*kR) + r25
								r06 = (np.sqrt(5./8)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf1)*cmath.exp(1j*kR) + r06
								r16 = (np.sqrt(5./8)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*lcoef[0]*lcoef[1]*(3.*lcoef[0]**2 -lcoef[1]**2 -2.)*tpf1)*cmath.exp(1j*kR) + r16
								r26 = (np.sqrt(5./8)*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tpf0-1./4*np.sqrt(15.)*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)+lcoef[1]**2 -lcoef[0]**2)*tpf1)*cmath.exp(1j*kR) + r26
						for ms in [-0.5, 0.5]:
							row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
							row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
							row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
							row6 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
							row7 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
							col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
							col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							# p0 coeffs
							self.H0[row1,col1,ik,jk,kk] = r00*(-1)**(l1+l2)
							self.H0[col1,row1,ik,jk,kk] = np.conjugate(r00)*(-1)**(l1+l2)
							self.H0[row2,col1,ik,jk,kk] =-1./np.sqrt(2)*(r01+1j*r02)*(-1)**(l1+l2)
							self.H0[col1,row2,ik,jk,kk] =-1./np.sqrt(2)*(np.conjugate(r01)-1j*np.conjugate(r02))*(-1)**(l1+l2)
							self.H0[row3,col1,ik,jk,kk] = 1./np.sqrt(2)*(r01-1j*r02)*(-1)**(l1+l2)
							self.H0[col1,row3,ik,jk,kk] = 1./np.sqrt(2)*(np.conjugate(r01)+1j*np.conjugate(r02))*(-1)**(l1+l2)
							self.H0[row4,col1,ik,jk,kk] = 1./np.sqrt(2)*(r03+1j*r04)*(-1)**(l1+l2)
							self.H0[col1,row4,ik,jk,kk] = 1./np.sqrt(2)*(np.conjugate(r03)-1j*np.conjugate(r04))*(-1)**(l1+l2)
							self.H0[row5,col1,ik,jk,kk] = 1./np.sqrt(2)*(r03-1j*r04)*(-1)**(l1+l2)
							self.H0[col1,row5,ik,jk,kk] = 1./np.sqrt(2)*(np.conjugate(r03)+1j*np.conjugate(r04))*(-1)**(l1+l2)
							self.H0[row6,col1,ik,jk,kk] =-1./np.sqrt(2)*(r05+1j*r06)*(-1)**(l1+l2)
							self.H0[col1,row6,ik,jk,kk] =-1./np.sqrt(2)*(np.conjugate(r05)-1j*np.conjugate(r06))*(-1)**(l1+l2)
							self.H0[row7,col1,ik,jk,kk] = 1./np.sqrt(2)*(r05-1j*r06)*(-1)**(l1+l2)
							self.H0[col1,row7,ik,jk,kk] = 1./np.sqrt(2)*(np.conjugate(r05)+1j*np.conjugate(r06))*(-1)**(l1+l2)
							# p1 coeffs
							self.H0[row1,col2,ik,jk,kk] =-1./np.sqrt(2)*(r10-1j*r20)*(-1)**(l1+l2)
							self.H0[col2,row1,ik,jk,kk] =-1./np.sqrt(2)*(np.conjugate(r10)+1j*np.conjugate(r20))*(-1)**(l1+l2)
							self.H0[row2,col2,ik,jk,kk] = 0.5*(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
							self.H0[col2,row2,ik,jk,kk] = 0.5*(np.conjugate(r11+r22)-1j*np.conjugate(r12-r21))*(-1)**(l1+l2)
							self.H0[row3,col2,ik,jk,kk] =-0.5*(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
							self.H0[col2,row3,ik,jk,kk] =-0.5*(np.conjugate(r11-r22)+1j*np.conjugate(r12+r21))*(-1)**(l1+l2)
							self.H0[row4,col2,ik,jk,kk] =-0.5*(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
							self.H0[col2,row4,ik,jk,kk] =-0.5*(np.conjugate(r13)+1j*np.conjugate(r23)-1j*np.conjugate(r14)+np.conjugate(r24))*(-1)**(l1+l2)
							self.H0[row5,col2,ik,jk,kk] =-0.5*(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
							self.H0[col2,row5,ik,jk,kk] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
							self.H0[row6,col2,ik,jk,kk] = 0.5*(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
							self.H0[col2,row6,ik,jk,kk] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
							self.H0[row7,col2,ik,jk,kk] =-0.5*(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
							self.H0[col2,row7,ik,jk,kk] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
							# p-1 coeffs
							self.H0[row1,col3,ik,jk,kk] = 1./np.sqrt(2)*(r10+1j*r20)*(-1)**(l1+l2)
							self.H0[col3,row1,ik,jk,kk] = 1./np.sqrt(2)*np.conjugate(r10+1j*r20)*(-1)**(l1+l2)
							self.H0[row2,col3,ik,jk,kk] =-0.5*(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
							self.H0[col3,row2,ik,jk,kk] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
							self.H0[row3,col3,ik,jk,kk] = 0.5*(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
							self.H0[col3,row3,ik,jk,kk] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
							self.H0[row4,col3,ik,jk,kk] = 0.5*(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
							self.H0[col3,row4,ik,jk,kk] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
							self.H0[row5,col3,ik,jk,kk] = 0.5*(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
							self.H0[col3,row5,ik,jk,kk] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
							self.H0[row6,col3,ik,jk,kk] =-0.5*(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
							self.H0[col3,row6,ik,jk,kk] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
							self.H0[row7,col3,ik,jk,kk] = 0.5*(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
							self.H0[col3,row7,ik,jk,kk] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
						# iterate iik
						iik = iik + 1
	# (d,f) SK integrals
	def set_df_hopping_mtxel(self, Site1, Site2, tdf, siteslist, kg, Unitcell, MatrixEntry):
		# (d,f) orbitals
		l1 = 2
		l2 = 3
		# SK integrals
		[tdf0, tdf1, tdf2] = tdf
		###   dimensions
		if kg.D == 0:
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r00 = 0.
			r10 = 0.
			r20 = 0.
			r30 = 0.
			r40 = 0.
			r01 = 0.
			r11 = 0.
			r21 = 0.
			r31 = 0.
			r41 = 0.
			r02 = 0.
			r12 = 0.
			r22 = 0.
			r32 = 0.
			r42 = 0.
			r03 = 0.
			r13 = 0.
			r23 = 0.
			r33 = 0.
			r43 = 0.
			r04 = 0.
			r14 = 0.
			r24 = 0.
			r34 = 0.
			r44 = 0.
			r05 = 0.
			r15 = 0.
			r25 = 0.
			r35 = 0.
			r45 = 0.
			r06 = 0.
			r16 = 0.
			r26 = 0.
			r36 = 0.
			r46 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r00 = 1./4*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(5.*lcoef[2]**2 - 3.)*tdf0 - 3./np.sqrt(8)*lcoef[2]*(5.*lcoef[2]**2 - 1.)*(lcoef[2]**2 - 1.)*tdf1 + 1./4*np.sqrt(45)*lcoef[2]*(lcoef[2]**2 -1.)**2 *tdf2
					r10 = 0.5*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(5.*lcoef[2]**2 - 3.)*tdf0 - np.sqrt(3./8)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2
					r20 = 0.5*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./8)*lcoef[1]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2
					r30 = 1./4*np.sqrt(3.)*lcoef[2]*(5.*lcoef[2]**2 -3.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(3./8)*lcoef[2]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1./4*np.sqrt(15.)*lcoef[2]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2
					r40 = 0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./2)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 +1.)*tdf2
					r01 = np.sqrt(3./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[0]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2
					r11 = 3./np.sqrt(8.)*lcoef[0]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[0]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -2.)*tdf2
					r21 = 3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2
					r31 = 3./np.sqrt(32.)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*lcoef[0]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[1]**2 -4.*lcoef[2]**2)*tdf1 + np.sqrt(5./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[2]**2 +1.) - 4.*lcoef[2]**2)*tdf2
					r41 = 3./np.sqrt(8.)*lcoef[0]**2 *lcoef[1]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*((6.*lcoef[0]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[1]*(lcoef[0]**2 *(3.*lcoef[2]**2 +1.) - 2.*lcoef[2]**2)*tdf2
					r02 = np.sqrt(3./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[1]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2
					r12 = 3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2
					r22 = 3./np.sqrt(8.)*lcoef[1]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[1]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -2.)*tdf2
					r32 = 3./np.sqrt(32.)*lcoef[1]*(lcoef[0]**2 - lcoef[1]**2)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 4.*lcoef[2]**2 - 2.*lcoef[0]**2)*tdf1 + np.sqrt(5./32)*lcoef[1]*((lcoef[0]**2 - lcoef[1]**2)*(3.*lcoef[2]**2 +1.) + 4.*lcoef[2]**2)*tdf2
					r42 = 3./np.sqrt(8.)*lcoef[0]*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[0]*((6.*lcoef[1]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[2]**2 + 1.) - 2.*lcoef[2]**2)*tdf2
					r03 = 1./4*np.sqrt(15.)*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf0 - np.sqrt(15./8)*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + np.sqrt(3.)/4*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2
					r13 = 0.5*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[0]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) - 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[0]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[1]**2 - 2.*lcoef[2]**2)*tdf2
					r23 = 0.5*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[1]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[1]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) - 4.*lcoef[0]**2 + 2.*lcoef[2]**2)*tdf2
					r33 = 1./4*np.sqrt(45.)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)**2 *tdf0 - np.sqrt(5./8)*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 2.*lcoef[2]**2 - 2.)*tdf1 + 1./4*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 8.*lcoef[2]**2 -4.)*tdf2
					r43 = 0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2
					r04 = 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf0 - np.sqrt(15./2)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf2
					r14 = np.sqrt(45.)*lcoef[0]**2 *lcoef[1]*lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[1]*(6.*lcoef[0]**2 *lcoef[2]**2 + lcoef[1]**2 -1.)*tdf1 + lcoef[1]*(3.*lcoef[0]**2 *lcoef[2]**2 + 2.*lcoef[1]**2 -1.)*tdf2
					r24 = np.sqrt(45.)*lcoef[0]*lcoef[1]**2 *lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[0]*(6.*lcoef[1]**2 *lcoef[2]**2 + lcoef[0]**2 -1.)*tdf1 + lcoef[0]*(3.*lcoef[1]**2 *lcoef[2]**2 +2.*lcoef[0]**2 -1.)*tdf2
					r34 = 0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2
					r44 = np.sqrt(45.)*lcoef[0]**2 *lcoef[1]**2 *lcoef[2]*tdf0 - np.sqrt(5./2)*lcoef[2]*(6.*lcoef[0]**2 *lcoef[1]**2 +lcoef[2]**2 -1.)*tdf1 + lcoef[2]*(3.*lcoef[0]**2 *lcoef[1]**2 +2.*lcoef[2]**2 -1.)*tdf2
					r05 = np.sqrt(5./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[0]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf2
					r15 = np.sqrt(15./8)*lcoef[0]**2 *lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[0]**2 *(lcoef[0]**2 - 3.*lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[0]**2 +2.*lcoef[1]**2)*tdf2
					r25 = np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +4.)*tdf2
					r35 = np.sqrt(15./32)*lcoef[0]*(lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*((lcoef[0]**2 - lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) - lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) +4.*lcoef[2]**2)*tdf2
					r45 = np.sqrt(15./8)*lcoef[0]**2 *lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*(2.*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[1]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[2]**2)*tdf2
					r06 = np.sqrt(5./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[1]*(lcoef[2]**2 +1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf2
					r16 = np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -4.)*tdf2
					r26 = np.sqrt(15./8)*lcoef[1]**2 *lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[0]**2 + 2.*lcoef[1]**2)*tdf2
					r36 = np.sqrt(15./32)*lcoef[1]*(lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./32)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[2]**2)*tdf2
					r46 = np.sqrt(15./8)*lcoef[0]*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[2]**2)*tdf2
			for ms in [-0.5, 0.5]:
				row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
				row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
				row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
				col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
				col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
				col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
				col6 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
				col7 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
				########### d0 COEFFICIENTS #############################
				self.H0[row1,col1] = r00
				self.H0[col1,row1] = np.conjugate(r00)
				self.H0[row1,col2] =-1./np.sqrt(2.)*(r01+1j*r02)
				self.H0[col2,row1] =-1./np.sqrt(2.)*np.conjugate(r01+1j*r02)
				self.H0[row1,col3] = 1./np.sqrt(2.)*(r01-1j*r02)
				self.H0[col3,row1] = 1./np.sqrt(2.)*np.conjugate(r01-1j*r02)
				self.H0[row1,col4] = 1./np.sqrt(2.)*(r03+1j*r04)
				self.H0[col4,row1] = 1./np.sqrt(2.)*np.conjugate(r03+1j*r04)
				self.H0[row1,col5] = 1./np.sqrt(2.)*(r03-1j*r04)
				self.H0[col5,row1] = 1./np.sqrt(2.)*np.conjugate(r03-1j*r04)
				self.H0[row1,col6] =-1./np.sqrt(2.)*(r05+1j*r06)
				self.H0[col6,row1] =-1./np.sqrt(2.)*np.conjugate(r05+1j*r06)
				self.H0[row1,col7] = 1./np.sqrt(2.)*(r05-1j*r06)
				self.H0[col7,row1] = 1./np.sqrt(2.)*np.conjugate(r05-1j*r06)
				########### d1 COEFFICIENTS ##############################
				self.H0[row2,col1] =-1./np.sqrt(2.)*(r10-1j*r20)
				self.H0[col1,row2] =-1./np.sqrt(2.)*np.conjugate(r10-1j*r20)
				self.H0[row2,col2] = 0.5*(r11+r22+1j*(r12-r21))
				self.H0[col2,row2] = 0.5*np.conjugate(r11+r22+1j*(r12-r21))
				self.H0[row2,col3] =-0.5*(r11-r22-1j*(r12+r21))
				self.H0[col3,row2] =-0.5*np.conjugate(r11-r22-1j*(r12+r21))
				self.H0[row2,col4] =-0.5*(r13-1j*r23+1j*r14+r24)
				self.H0[col4,row2] =-0.5*np.conjugate(r13-1j*r23+1j*r14+r24)
				self.H0[row2,col5] =-0.5*(r13-1j*r23-1j*r14-r24)
				self.H0[col5,row2] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)
				self.H0[row2,col6] = 0.5*(r15-1j*r25+1j*r16+r26)
				self.H0[col6,row2] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)
				self.H0[row2,col7] =-0.5*(r15-1j*r25-1j*r16-r26)
				self.H0[col7,row2] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)
				########### d-1 COEFFICIENTS ##############################
				self.H0[row3,col1] = 1./np.sqrt(2.)*(r10+1j*r20)
				self.H0[col1,row3] = 1./np.sqrt(2.)*np.conjugate(r10+1j*r20)
				self.H0[row3,col2] =-0.5*(r11-r22+1j*(r12+r21))
				self.H0[col2,row3] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))
				self.H0[row3,col3] = 0.5*(r11+r22+1j*(-r12+r21))
				self.H0[col3,row3] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))
				self.H0[row3,col4] = 0.5*(r13+1j*r23+1j*r14-r24)
				self.H0[col4,row3] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)
				self.H0[row3,col5] = 0.5*(r13+1j*r23-1j*r14+r24)
				self.H0[col5,row3] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)
				self.H0[row3,col6] =-0.5*(r15+1j*r25+1j*r16-r26)
				self.H0[col6,row3] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)
				self.H0[row3,col7] = 0.5*(r15+1j*r25-1j*r16+r26)
				self.H0[col7,row3] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)
				########### d2 COEFFICIENTS ###############################
				self.H0[row4,col1] = 1./np.sqrt(2.)*(r30-1j*r40)
				self.H0[col1,row4] = 1./np.sqrt(2.)*np.conjugate(r30-1j*r40)
				self.H0[row4,col2] =-0.5*(r31+r42+1j*(r32-r41))
				self.H0[col2,row4] =-0.5*np.conjugate(r31+r42+1j*(r32-r41))
				self.H0[row4,col3] = 0.5*(r31-r32-1j*(r32+r41))
				self.H0[col3,row4] = 0.5*np.conjugate(r31-r32-1j*(r32+r41))
				self.H0[row4,col4] = 0.5*(r33-1j*r43+1j*r34+r44)
				self.H0[col4,row4] = 0.5*np.conjugate(r33-1j*r43+1j*r34+r44)
				self.H0[row4,col5] = 0.5*(r33-1j*r43-1j*r34-r44)
				self.H0[col5,row4] = 0.5*np.conjugate(r33-1j*r43-1j*r34-r44)
				self.H0[row4,col6] =-0.5*(r35-1j*r45+1j*r36+r46)
				self.H0[col6,row4] =-0.5*np.conjugate(r35-1j*r45+1j*r36+r46)
				self.H0[row4,col7] = 0.5*(r35-1j*r45-1j*r36-r46)
				self.H0[col7,row4] = 0.5*np.conjugate(r35-1j*r45-1j*r36-r46)
				########### d-2 COEFFICIENTS ##############################
				self.H0[row5,col1] = 1./np.sqrt(2.)*(r30+1j*r40)
				self.H0[col1,row5] = 1./np.sqrt(2.)*np.conjugate(r30+1j*r40)
				self.H0[row5,col2] =-0.5*(r31-r42+1j*(r32+r41))
				self.H0[col2,row5] =-0.5*np.conjugate(r31-r42+1j*(r32+r41))
				self.H0[row5,col3] = 0.5*(r31+r42+1j*(-r32+r41))
				self.H0[col3,row5] = 0.5*np.conjugate(r31+r42+1j*(-r32+r41))
				self.H0[row5,col4] = 0.5*(r33+1j*r43+1j*r34-r44)
				self.H0[col4,row5] = 0.5*np.conjugate(r33+1j*r43+1j*r34-r44)
				self.H0[row5,col5] = 0.5*(r33+1j*r43-1j*r34+r44)
				self.H0[col5,row5] = 0.5*np.conjugate(r33+1j*r43-1j*r34+r44)
				self.H0[row5,col6] =-0.5*(r35+1j*r45+1j*r36-r46)
				self.H0[col6,row5] =-0.5*np.conjugate(r35+1j*r45+1j*r36-r46)
				self.H0[row5,col7] = 0.5*(r35+1j*r45-1j*r36+r46)
				self.H0[col7,row5] = 0.5*np.conjugate(r35+1j*r45-1j*r36+r46)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r00 = 0.
				r10 = 0.
				r20 = 0.
				r30 = 0.
				r40 = 0.
				r01 = 0.
				r11 = 0.
				r21 = 0.
				r31 = 0.
				r41 = 0.
				r02 = 0.
				r12 = 0.
				r22 = 0.
				r32 = 0.
				r42 = 0.
				r03 = 0.
				r13 = 0.
				r23 = 0.
				r33 = 0.
				r43 = 0.
				r04 = 0.
				r14 = 0.
				r24 = 0.
				r34 = 0.
				r44 = 0.
				r05 = 0.
				r15 = 0.
				r25 = 0.
				r35 = 0.
				r45 = 0.
				r06 = 0.
				r16 = 0.
				r26 = 0.
				r36 = 0.
				r46 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r00 = (1./4*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(5.*lcoef[2]**2 - 3.)*tdf0 - 3./np.sqrt(8)*lcoef[2]*(5.*lcoef[2]**2 - 1.)*(lcoef[2]**2 - 1.)*tdf1 + 1./4*np.sqrt(45)*lcoef[2]*(lcoef[2]**2 -1.)**2 *tdf2)*cmath.exp(1j*kR) + r00
						r10 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(5.*lcoef[2]**2 - 3.)*tdf0 - np.sqrt(3./8)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r10
						r20 = (0.5*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./8)*lcoef[1]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r20
						r30 = (1./4*np.sqrt(3.)*lcoef[2]*(5.*lcoef[2]**2 -3.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(3./8)*lcoef[2]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1./4*np.sqrt(15.)*lcoef[2]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r30
						r40 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./2)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 +1.)*tdf2)*cmath.exp(1j*kR) + r40
						r01 = (np.sqrt(3./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[0]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r01
						r11 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[0]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r11
						r21 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r21
						r31 = (3./np.sqrt(32.)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*lcoef[0]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[1]**2 -4.*lcoef[2]**2)*tdf1 + np.sqrt(5./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[2]**2 +1.) - 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r31
						r41 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[1]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*((6.*lcoef[0]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[1]*(lcoef[0]**2 *(3.*lcoef[2]**2 +1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r41
						r02 = (np.sqrt(3./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[1]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r02
						r12 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r12
						r22 = (3./np.sqrt(8.)*lcoef[1]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[1]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r22
						r32 = (3./np.sqrt(32.)*lcoef[1]*(lcoef[0]**2 - lcoef[1]**2)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 4.*lcoef[2]**2 - 2.*lcoef[0]**2)*tdf1 + np.sqrt(5./32)*lcoef[1]*((lcoef[0]**2 - lcoef[1]**2)*(3.*lcoef[2]**2 +1.) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r32
						r42 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[0]*((6.*lcoef[1]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[2]**2 + 1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r42
						r03 = (1./4*np.sqrt(15.)*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf0 - np.sqrt(15./8)*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + np.sqrt(3.)/4*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r03
						r13 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[0]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) - 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[0]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[1]**2 - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r13
						r23 = (0.5*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[1]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[1]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) - 4.*lcoef[0]**2 + 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r23
						r33 = (1./4*np.sqrt(45.)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)**2 *tdf0 - np.sqrt(5./8)*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 2.*lcoef[2]**2 - 2.)*tdf1 + 1./4*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 8.*lcoef[2]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r33
						r43 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r43
						r04 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf0 - np.sqrt(15./2)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r04
						r14 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]*lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[1]*(6.*lcoef[0]**2 *lcoef[2]**2 + lcoef[1]**2 -1.)*tdf1 + lcoef[1]*(3.*lcoef[0]**2 *lcoef[2]**2 + 2.*lcoef[1]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r14
						r24 = (np.sqrt(45.)*lcoef[0]*lcoef[1]**2 *lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[0]*(6.*lcoef[1]**2 *lcoef[2]**2 + lcoef[0]**2 -1.)*tdf1 + lcoef[0]*(3.*lcoef[1]**2 * lcoef[2]**2 +2.*lcoef[0]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r24
						r34 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r34
						r44 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]**2 *lcoef[2]*tdf0 - np.sqrt(5./2)*lcoef[2]*(6.*lcoef[0]**2 *lcoef[1]**2 +lcoef[2]**2 -1.)*tdf1 + lcoef[2]*(3.*lcoef[0]**2 *lcoef[1]**2 +2.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r44
						r05 = (np.sqrt(5./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[0]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r05
						r15 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[0]**2 *(lcoef[0]**2 - 3.*lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[0]**2 +2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r15
						r25 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +4.)*tdf2)*cmath.exp(1j*kR) + r25
						r35 = (np.sqrt(15./32)*lcoef[0]*(lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*((lcoef[0]**2 - lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) - lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) +4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r35
						r45 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*(2.*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[1]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r45
						r06 = (np.sqrt(5./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[1]*(lcoef[2]**2 +1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r06
						r16 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r16
						r26 = (np.sqrt(15./8)*lcoef[1]**2 *lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[0]**2 + 2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r26
						r36 = (np.sqrt(15./32)*lcoef[1]*(lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./32)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r36
						r46 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r46
				for ms in [-0.5, 0.5]:
					row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
					row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
					row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
					col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
					col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
					col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
					col6 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
					col7 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
					########### d0 COEFFICIENTS #############################
					self.H0[row1,col1,ik] = r00
					self.H0[col1,row1,ik] = np.conjugate(r00)
					self.H0[row1,col2,ik] =-1./np.sqrt(2.)*(r01+1j*r02)
					self.H0[col2,row1,ik] =-1./np.sqrt(2.)*np.conjugate(r01+1j*r02)
					self.H0[row1,col3,ik] = 1./np.sqrt(2.)*(r01-1j*r02)
					self.H0[col3,row1,ik] = 1./np.sqrt(2.)*np.conjugate(r01-1j*r02)
					self.H0[row1,col4,ik] = 1./np.sqrt(2.)*(r03+1j*r04)
					self.H0[col4,row1,ik] = 1./np.sqrt(2.)*np.conjugate(r03+1j*r04)
					self.H0[row1,col5,ik] = 1./np.sqrt(2.)*(r03-1j*r04)
					self.H0[col5,row1,ik] = 1./np.sqrt(2.)*np.conjugate(r03-1j*r04)
					self.H0[row1,col6,ik] =-1./np.sqrt(2.)*(r05+1j*r06)
					self.H0[col6,row1,ik] =-1./np.sqrt(2.)*np.conjugate(r05+1j*r06)
					self.H0[row1,col7,ik] = 1./np.sqrt(2.)*(r05-1j*r06)
					self.H0[col7,row1,ik] = 1./np.sqrt(2.)*np.conjugate(r05-1j*r06)
					########### d1 COEFFICIENTS ##############################
					self.H0[row2,col1,ik] =-1./np.sqrt(2.)*(r10-1j*r20)
					self.H0[col1,row2,ik] =-1./np.sqrt(2.)*np.conjugate(r10-1j*r20)
					self.H0[row2,col2,ik] = 0.5*(r11+r22+1j*(r12-r21))
					self.H0[col2,row2,ik] = 0.5*np.conjugate(r11+r22+1j*(r12-r21))
					self.H0[row2,col3,ik] =-0.5*(r11-r22-1j*(r12+r21))
					self.H0[col3,row2,ik] =-0.5*np.conjugate(r11-r22-1j*(r12+r21))
					self.H0[row2,col4,ik] =-0.5*(r13-1j*r23+1j*r14+r24)
					self.H0[col4,row2,ik] =-0.5*np.conjugate(r13-1j*r23+1j*r14+r24)
					self.H0[row2,col5,ik] =-0.5*(r13-1j*r23-1j*r14-r24)
					self.H0[col5,row2,ik] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)
					self.H0[row2,col6,ik] = 0.5*(r15-1j*r25+1j*r16+r26)
					self.H0[col6,row2,ik] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)
					self.H0[row2,col7,ik] =-0.5*(r15-1j*r25-1j*r16-r26)
					self.H0[col7,row2,ik] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)
					########### d-1 COEFFICIENTS ##############################
					self.H0[row3,col1,ik] = 1./np.sqrt(2.)*(r10+1j*r20)
					self.H0[col1,row3,ik] = 1./np.sqrt(2.)*np.conjugate(r10+1j*r20)
					self.H0[row3,col2,ik] =-0.5*(r11-r22+1j*(r12+r21))
					self.H0[col2,row3,ik] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))
					self.H0[row3,col3,ik] = 0.5*(r11+r22+1j*(-r12+r21))
					self.H0[col3,row3,ik] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))
					self.H0[row3,col4,ik] = 0.5*(r13+1j*r23+1j*r14-r24)
					self.H0[col4,row3,ik] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)
					self.H0[row3,col5,ik] = 0.5*(r13+1j*r23-1j*r14+r24)
					self.H0[col5,row3,ik] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)
					self.H0[row3,col6,ik] =-0.5*(r15+1j*r25+1j*r16-r26)
					self.H0[col6,row3,ik] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)
					self.H0[row3,col7,ik] = 0.5*(r15+1j*r25-1j*r16+r26)
					self.H0[col7,row3,ik] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)
					########### d2 COEFFICIENTS ###############################
					self.H0[row4,col1,ik] = 1./np.sqrt(2.)*(r30-1j*r40)
					self.H0[col1,row4,ik] = 1./np.sqrt(2.)*np.conjugate(r30-1j*r40)
					self.H0[row4,col2,ik] =-0.5*(r31+r42+1j*(r32-r41))
					self.H0[col2,row4,ik] =-0.5*np.conjugate(r31+r42+1j*(r32-r41))
					self.H0[row4,col3,ik] = 0.5*(r31-r32-1j*(r32+r41))
					self.H0[col3,row4,ik] = 0.5*np.conjugate(r31-r32-1j*(r32+r41))
					self.H0[row4,col4,ik] = 0.5*(r33-1j*r43+1j*r34+r44)
					self.H0[col4,row4,ik] = 0.5*np.conjugate(r33-1j*r43+1j*r34+r44)
					self.H0[row4,col5,ik] = 0.5*(r33-1j*r43-1j*r34-r44)
					self.H0[col5,row4,ik] = 0.5*np.conjugate(r33-1j*r43-1j*r34-r44)
					self.H0[row4,col6,ik] =-0.5*(r35-1j*r45+1j*r36+r46)
					self.H0[col6,row4,ik] =-0.5*np.conjugate(r35-1j*r45+1j*r36+r46)
					self.H0[row4,col7,ik] = 0.5*(r35-1j*r45-1j*r36-r46)
					self.H0[col7,row4,ik] = 0.5*np.conjugate(r35-1j*r45-1j*r36-r46)
					########### d-2 COEFFICIENTS ##############################
					self.H0[row5,col1,ik] = 1./np.sqrt(2.)*(r30+1j*r40)
					self.H0[col1,row5,ik] = 1./np.sqrt(2.)*np.conjugate(r30+1j*r40)
					self.H0[row5,col2,ik] =-0.5*(r31-r42+1j*(r32+r41))
					self.H0[col2,row5,ik] =-0.5*np.conjugate(r31-r42+1j*(r32+r41))
					self.H0[row5,col3,ik] = 0.5*(r31+r42+1j*(-r32+r41))
					self.H0[col3,row5,ik] = 0.5*np.conjugate(r31+r42+1j*(-r32+r41))
					self.H0[row5,col4,ik] = 0.5*(r33+1j*r43+1j*r34-r44)
					self.H0[col4,row5,ik] = 0.5*np.conjugate(r33+1j*r43+1j*r34-r44)
					self.H0[row5,col5,ik] = 0.5*(r33+1j*r43-1j*r34+r44)
					self.H0[col5,row5,ik] = 0.5*np.conjugate(r33+1j*r43-1j*r34+r44)
					self.H0[row5,col6,ik] =-0.5*(r35+1j*r45+1j*r36-r46)
					self.H0[col6,row5,ik] =-0.5*np.conjugate(r35+1j*r45+1j*r36-r46)
					self.H0[row5,col7,ik] = 0.5*(r35+1j*r45-1j*r36+r46)
					self.H0[col7,row5,ik] = 0.5*np.conjugate(r35+1j*r45-1j*r36+r46)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r00 = 0.
					r10 = 0.
					r20 = 0.
					r30 = 0.
					r40 = 0.
					r01 = 0.
					r11 = 0.
					r21 = 0.
					r31 = 0.
					r41 = 0.
					r02 = 0.
					r12 = 0.
					r22 = 0.
					r32 = 0.
					r42 = 0.
					r03 = 0.
					r13 = 0.
					r23 = 0.
					r33 = 0.
					r43 = 0.
					r04 = 0.
					r14 = 0.
					r24 = 0.
					r34 = 0.
					r44 = 0.
					r05 = 0.
					r15 = 0.
					r25 = 0.
					r35 = 0.
					r45 = 0.
					r06 = 0.
					r16 = 0.
					r26 = 0.
					r36 = 0.
					r46 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r00 = (1./4*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(5.*lcoef[2]**2 - 3.)*tdf0 - 3./np.sqrt(8)*lcoef[2]*(5.*lcoef[2]**2 - 1.)*(lcoef[2]**2 - 1.)*tdf1 + 1./4*np.sqrt(45)*lcoef[2]*(lcoef[2]**2 -1.)**2 *tdf2)*cmath.exp(1j*kR) + r00
							r10 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(5.*lcoef[2]**2 - 3.)*tdf0 - np.sqrt(3./8)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r10
							r20 = (0.5*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./8)*lcoef[1]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r20
							r30 = (1./4*np.sqrt(3.)*lcoef[2]*(5.*lcoef[2]**2 -3.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(3./8)*lcoef[2]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1./4*np.sqrt(15.)*lcoef[2]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r30
							r40 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./2)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 +1.)*tdf2)*cmath.exp(1j*kR) + r40
							r01 = (np.sqrt(3./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[0]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r01
							r11 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[0]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r11
							r21 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r21
							r31 = (3./np.sqrt(32.)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*lcoef[0]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[1]**2 -4.*lcoef[2]**2)*tdf1 + np.sqrt(5./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[2]**2 +1.) - 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r31
							r41 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[1]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*((6.*lcoef[0]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[1]*(lcoef[0]**2 *(3.*lcoef[2]**2 +1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r41
							r02 = (np.sqrt(3./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[1]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r02
							r12 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r12
							r22 = (3./np.sqrt(8.)*lcoef[1]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[1]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r22
							r32 = (3./np.sqrt(32.)*lcoef[1]*(lcoef[0]**2 - lcoef[1]**2)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 4.*lcoef[2]**2 - 2.*lcoef[0]**2)*tdf1 + np.sqrt(5./32)*lcoef[1]*((lcoef[0]**2 - lcoef[1]**2)*(3.*lcoef[2]**2 +1.) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r32
							r42 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[0]*((6.*lcoef[1]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[2]**2 + 1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r42
							r03 = (1./4*np.sqrt(15.)*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf0 - np.sqrt(15./8)*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + np.sqrt(3.)/4*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r03
							r13 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[0]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) - 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[0]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[1]**2 - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r13
							r23 = (0.5*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[1]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[1]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) - 4.*lcoef[0]**2 + 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r23
							r33 = (1./4*np.sqrt(45.)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)**2 *tdf0 - np.sqrt(5./8)*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 2.*lcoef[2]**2 - 2.)*tdf1 + 1./4*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 8.*lcoef[2]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r33
							r43 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r43
							r04 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf0 - np.sqrt(15./2)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r04
							r14 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]*lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[1]*(6.*lcoef[0]**2 *lcoef[2]**2 + lcoef[1]**2 -1.)*tdf1 + lcoef[1]*(3.*lcoef[0]**2 *lcoef[2]**2 + 2.*lcoef[1]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r14
							r24 = (np.sqrt(45.)*lcoef[0]*lcoef[1]**2 *lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[0]*(6.*lcoef[1]**2 *lcoef[2]**2 + lcoef[0]**2 -1.)*tdf1 + lcoef[0]*(3.*lcoef[1]**2 *lcoef[2]**2 +2.*lcoef[0]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r24
							r34 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r34
							r44 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]**2 *lcoef[2]*tdf0 - np.sqrt(5./2)*lcoef[2]*(6.*lcoef[0]**2 *lcoef[1]**2 +lcoef[2]**2 -1.)*tdf1 + lcoef[2]*(3.*lcoef[0]**2 *lcoef[1]**2 +2.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r44
							r05 = (np.sqrt(5./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[0]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r05
							r15 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[0]**2 *(lcoef[0]**2 - 3.*lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[0]**2 +2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r15
							r25 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +4.)*tdf2)*cmath.exp(1j*kR) + r25
							r35 = (np.sqrt(15./32)*lcoef[0]*(lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*((lcoef[0]**2 - lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) - lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) +4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r35
							r45 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*(2.*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[1]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r45
							r06 = (np.sqrt(5./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[1]*(lcoef[2]**2 +1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r06
							r16 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r16
							r26 = (np.sqrt(15./8)*lcoef[1]**2 *lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[0]**2 + 2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r26
							r36 = (np.sqrt(15./32)*lcoef[1]*(lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./32)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r36
							r46 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r46
					for ms in [-0.5, 0.5]:
						row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
						row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
						row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
						col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
						col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
						col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
						col6 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
						col7 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
						########### d0 COEFFICIENTS #############################
						self.H0[row1,col1,ik,jk] = r00
						self.H0[col1,row1,ik,jk] = np.conjugate(r00)
						self.H0[row1,col2,ik,jk] =-1./np.sqrt(2.)*(r01+1j*r02)
						self.H0[col2,row1,ik,jk] =-1./np.sqrt(2.)*np.conjugate(r01+1j*r02)
						self.H0[row1,col3,ik,jk] = 1./np.sqrt(2.)*(r01-1j*r02)
						self.H0[col3,row1,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r01-1j*r02)
						self.H0[row1,col4,ik,jk] = 1./np.sqrt(2.)*(r03+1j*r04)
						self.H0[col4,row1,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r03+1j*r04)
						self.H0[row1,col5,ik,jk] = 1./np.sqrt(2.)*(r03-1j*r04)
						self.H0[col5,row1,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r03-1j*r04)
						self.H0[row1,col6,ik,jk] =-1./np.sqrt(2.)*(r05+1j*r06)
						self.H0[col6,row1,ik,jk] =-1./np.sqrt(2.)*np.conjugate(r05+1j*r06)
						self.H0[row1,col7,ik,jk] = 1./np.sqrt(2.)*(r05-1j*r06)
						self.H0[col7,row1,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r05-1j*r06)
						########### d1 COEFFICIENTS ##############################
						self.H0[row2,col1,ik,jk] =-1./np.sqrt(2.)*(r10-1j*r20)
						self.H0[col1,row2,ik,jk] =-1./np.sqrt(2.)*np.conjugate(r10-1j*r20)
						self.H0[row2,col2,ik,jk] = 0.5*(r11+r22+1j*(r12-r21))
						self.H0[col2,row2,ik,jk] = 0.5*np.conjugate(r11+r22+1j*(r12-r21))
						self.H0[row2,col3,ik,jk] =-0.5*(r11-r22-1j*(r12+r21))
						self.H0[col3,row2,ik,jk] =-0.5*np.conjugate(r11-r22-1j*(r12+r21))
						self.H0[row2,col4,ik,jk] =-0.5*(r13-1j*r23+1j*r14+r24)
						self.H0[col4,row2,ik,jk] =-0.5*np.conjugate(r13-1j*r23+1j*r14+r24)
						self.H0[row2,col5,ik,jk] =-0.5*(r13-1j*r23-1j*r14-r24)
						self.H0[col5,row2,ik,jk] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)
						self.H0[row2,col6,ik,jk] = 0.5*(r15-1j*r25+1j*r16+r26)
						self.H0[col6,row2,ik,jk] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)
						self.H0[row2,col7,ik,jk] =-0.5*(r15-1j*r25-1j*r16-r26)
						self.H0[col7,row2,ik,jk] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)
						########### d-1 COEFFICIENTS ##############################
						self.H0[row3,col1,ik,jk] = 1./np.sqrt(2.)*(r10+1j*r20)
						self.H0[col1,row3,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r10+1j*r20)
						self.H0[row3,col2,ik,jk] =-0.5*(r11-r22+1j*(r12+r21))
						self.H0[col2,row3,ik,jk] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))
						self.H0[row3,col3,ik,jk] = 0.5*(r11+r22+1j*(-r12+r21))
						self.H0[col3,row3,ik,jk] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))
						self.H0[row3,col4,ik,jk] = 0.5*(r13+1j*r23+1j*r14-r24)
						self.H0[col4,row3,ik,jk] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)
						self.H0[row3,col5,ik,jk] = 0.5*(r13+1j*r23-1j*r14+r24)
						self.H0[col5,row3,ik,jk] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)
						self.H0[row3,col6,ik,jk] =-0.5*(r15+1j*r25+1j*r16-r26)
						self.H0[col6,row3,ik,jk] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)
						self.H0[row3,col7,ik,jk] = 0.5*(r15+1j*r25-1j*r16+r26)
						self.H0[col7,row3,ik,jk] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)
						########### d2 COEFFICIENTS ###############################
						self.H0[row4,col1,ik,jk] = 1./np.sqrt(2.)*(r30-1j*r40)
						self.H0[col1,row4,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r30-1j*r40)
						self.H0[row4,col2,ik,jk] =-0.5*(r31+r42+1j*(r32-r41))
						self.H0[col2,row4,ik,jk] =-0.5*np.conjugate(r31+r42+1j*(r32-r41))
						self.H0[row4,col3,ik,jk] = 0.5*(r31-r32-1j*(r32+r41))
						self.H0[col3,row4,ik,jk] = 0.5*np.conjugate(r31-r32-1j*(r32+r41))
						self.H0[row4,col4,ik,jk] = 0.5*(r33-1j*r43+1j*r34+r44)
						self.H0[col4,row4,ik,jk] = 0.5*np.conjugate(r33-1j*r43+1j*r34+r44)
						self.H0[row4,col5,ik,jk] = 0.5*(r33-1j*r43-1j*r34-r44)
						self.H0[col5,row4,ik,jk] = 0.5*np.conjugate(r33-1j*r43-1j*r34-r44)
						self.H0[row4,col6,ik,jk] =-0.5*(r35-1j*r45+1j*r36+r46)
						self.H0[col6,row4,ik,jk] =-0.5*np.conjugate(r35-1j*r45+1j*r36+r46)
						self.H0[row4,col7,ik,jk] = 0.5*(r35-1j*r45-1j*r36-r46)
						self.H0[col7,row4,ik,jk] = 0.5*np.conjugate(r35-1j*r45-1j*r36-r46)
						########### d-2 COEFFICIENTS ##############################
						self.H0[row5,col1,ik,jk] = 1./np.sqrt(2.)*(r30+1j*r40)
						self.H0[col1,row5,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r30+1j*r40)
						self.H0[row5,col2,ik,jk] =-0.5*(r31-r42+1j*(r32+r41))
						self.H0[col2,row5,ik,jk] =-0.5*np.conjugate(r31-r42+1j*(r32+r41))
						self.H0[row5,col3,ik,jk] = 0.5*(r31+r42+1j*(-r32+r41))
						self.H0[col3,row5,ik,jk] = 0.5*np.conjugate(r31+r42+1j*(-r32+r41))
						self.H0[row5,col4,ik,jk] = 0.5*(r33+1j*r43+1j*r34-r44)
						self.H0[col4,row5,ik,jk] = 0.5*np.conjugate(r33+1j*r43+1j*r34-r44)
						self.H0[row5,col5,ik,jk] = 0.5*(r33+1j*r43-1j*r34+r44)
						self.H0[col5,row5,ik,jk] = 0.5*np.conjugate(r33+1j*r43-1j*r34+r44)
						self.H0[row5,col6,ik,jk] =-0.5*(r35+1j*r45+1j*r36-r46)
						self.H0[col6,row5,ik,jk] =-0.5*np.conjugate(r35+1j*r45+1j*r36-r46)
						self.H0[row5,col7,ik,jk] = 0.5*(r35+1j*r45-1j*r36+r46)
						self.H0[col7,row5,ik,jk] = 0.5*np.conjugate(r35+1j*r45-1j*r36+r46)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r00 = 0.
						r10 = 0.
						r20 = 0.
						r30 = 0.
						r40 = 0.
						r01 = 0.
						r11 = 0.
						r21 = 0.
						r31 = 0.
						r41 = 0.
						r02 = 0.
						r12 = 0.
						r22 = 0.
						r32 = 0.
						r42 = 0.
						r03 = 0.
						r13 = 0.
						r23 = 0.
						r33 = 0.
						r43 = 0.
						r04 = 0.
						r14 = 0.
						r24 = 0.
						r34 = 0.
						r44 = 0.
						r05 = 0.
						r15 = 0.
						r25 = 0.
						r35 = 0.
						r45 = 0.
						r06 = 0.
						r16 = 0.
						r26 = 0.
						r36 = 0.
						r46 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r00 = (1./4*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(5.*lcoef[2]**2 - 3.)*tdf0 - 3./np.sqrt(8)*lcoef[2]*(5.*lcoef[2]**2 - 1.)*(lcoef[2]**2 - 1.)*tdf1 + 1./4*np.sqrt(45)*lcoef[2]*(lcoef[2]**2 -1.)**2 *tdf2)*cmath.exp(1j*kR) + r00
								r10 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(5.*lcoef[2]**2 - 3.)*tdf0 - np.sqrt(3./8)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r10
								r20 = (0.5*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./8)*lcoef[1]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r20
								r30 = (1./4*np.sqrt(3.)*lcoef[2]*(5.*lcoef[2]**2 -3.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(3./8)*lcoef[2]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1./4*np.sqrt(15.)*lcoef[2]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r30
								r40 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./2)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 +1.)*tdf2)*cmath.exp(1j*kR) + r40
								r01 = (np.sqrt(3./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[0]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r01
								r11 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[0]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r11
								r21 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r21
								r31 = (3./np.sqrt(32.)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*lcoef[0]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[1]**2 -4.*lcoef[2]**2)*tdf1 + np.sqrt(5./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[2]**2 +1.) - 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r31
								r41 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[1]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*((6.*lcoef[0]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[1]*(lcoef[0]**2 *(3.*lcoef[2]**2 +1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r41
								r02 = (np.sqrt(3./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[1]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r02
								r12 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r12
								r22 = (3./np.sqrt(8.)*lcoef[1]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[1]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r22
								r32 = (3./np.sqrt(32.)*lcoef[1]*(lcoef[0]**2 - lcoef[1]**2)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 4.*lcoef[2]**2 - 2.*lcoef[0]**2)*tdf1 + np.sqrt(5./32)*lcoef[1]*((lcoef[0]**2 - lcoef[1]**2)*(3.*lcoef[2]**2 +1.) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r32
								r42 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[0]*((6.*lcoef[1]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[2]**2 + 1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r42
								r03 = (1./4*np.sqrt(15.)*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf0 - np.sqrt(15./8)*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + np.sqrt(3.)/4*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r03
								r13 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[0]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) - 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[0]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[1]**2 - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r13
								r23 = (0.5*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[1]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[1]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) - 4.*lcoef[0]**2 + 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r23
								r33 = (1./4*np.sqrt(45.)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)**2 *tdf0 - np.sqrt(5./8)*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 2.*lcoef[2]**2 - 2.)*tdf1 + 1./4*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 8.*lcoef[2]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r33
								r43 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r43
								r04 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf0 - np.sqrt(15./2)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r04
								r14 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]*lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[1]*(6.*lcoef[0]**2 *lcoef[2]**2 + lcoef[1]**2 -1.)*tdf1 + lcoef[1]*(3.*lcoef[0]**2 *lcoef[2]**2 + 2.*lcoef[1]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r14
								r24 = (np.sqrt(45.)*lcoef[0]*lcoef[1]**2 *lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[0]*(6.*lcoef[1]**2 *lcoef[2]**2 + lcoef[0]**2 -1.)*tdf1 + lcoef[0]*(3.*lcoef[1]**2 *lcoef[2]**2 +2.*lcoef[0]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r24
								r34 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r34
								r44 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]**2 *lcoef[2]*tdf0 - np.sqrt(5./2)*lcoef[2]*(6.*lcoef[0]**2 *lcoef[1]**2 +lcoef[2]**2 -1.)*tdf1 + lcoef[2]*(3.*lcoef[0]**2 *lcoef[1]**2 +2.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r44
								r05 = (np.sqrt(5./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[0]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r05
								r15 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[0]**2 *(lcoef[0]**2 - 3.*lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[0]**2 +2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r15
								r25 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +4.)*tdf2)*cmath.exp(1j*kR) + r25
								r35 = (np.sqrt(15./32)*lcoef[0]*(lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*((lcoef[0]**2 - lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) - lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) +4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r35
								r45 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*(2.*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[1]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r45
								r06 = (np.sqrt(5./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[1]*(lcoef[2]**2 +1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r06
								r16 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r16
								r26 = (np.sqrt(15./8)*lcoef[1]**2 *lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[0]**2 + 2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r26
								r36 = (np.sqrt(15./32)*lcoef[1]*(lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./32)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r36
								r46 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r46
						for ms in [-0.5, 0.5]:
							row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
							row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
							row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
							col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
							col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
							col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
							col6 = MatrixEntry(siteslist.Atomslist, Site2, l2, 3, ms)
							col7 = MatrixEntry(siteslist.Atomslist, Site2, l2,-3, ms)
							########### d0 COEFFICIENTS #############################
							self.H0[row1,col1,ik,jk,kk] = r00
							self.H0[col1,row1,ik,jk,kk] = np.conjugate(r00)
							self.H0[row1,col2,ik,jk,kk] =-1./np.sqrt(2.)*(r01+1j*r02)
							self.H0[col2,row1,ik,jk,kk] =-1./np.sqrt(2.)*np.conjugate(r01+1j*r02)
							self.H0[row1,col3,ik,jk,kk] = 1./np.sqrt(2.)*(r01-1j*r02)
							self.H0[col3,row1,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r01-1j*r02)
							self.H0[row1,col4,ik,jk,kk] = 1./np.sqrt(2.)*(r03+1j*r04)
							self.H0[col4,row1,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r03+1j*r04)
							self.H0[row1,col5,ik,jk,kk] = 1./np.sqrt(2.)*(r03-1j*r04)
							self.H0[col5,row1,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r03-1j*r04)
							self.H0[row1,col6,ik,jk,kk] =-1./np.sqrt(2.)*(r05+1j*r06)
							self.H0[col6,row1,ik,jk,kk] =-1./np.sqrt(2.)*np.conjugate(r05+1j*r06)
							self.H0[row1,col7,ik,jk,kk] = 1./np.sqrt(2.)*(r05-1j*r06)
							self.H0[col7,row1,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r05-1j*r06)
							########### d1 COEFFICIENTS ##############################
							self.H0[row2,col1,ik,jk,kk] =-1./np.sqrt(2.)*(r10-1j*r20)
							self.H0[col1,row2,ik,jk,kk] =-1./np.sqrt(2.)*np.conjugate(r10-1j*r20)
							self.H0[row2,col2,ik,jk,kk] = 0.5*(r11+r22+1j*(r12-r21))
							self.H0[col2,row2,ik,jk,kk] = 0.5*np.conjugate(r11+r22+1j*(r12-r21))
							self.H0[row2,col3,ik,jk,kk] =-0.5*(r11-r22-1j*(r12+r21))
							self.H0[col3,row2,ik,jk,kk] =-0.5*np.conjugate(r11-r22-1j*(r12+r21))
							self.H0[row2,col4,ik,jk,kk] =-0.5*(r13-1j*r23+1j*r14+r24)
							self.H0[col4,row2,ik,jk,kk] =-0.5*np.conjugate(r13-1j*r23+1j*r14+r24)
							self.H0[row2,col5,ik,jk,kk] =-0.5*(r13-1j*r23-1j*r14-r24)
							self.H0[col5,row2,ik,jk,kk] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)
							self.H0[row2,col6,ik,jk,kk] = 0.5*(r15-1j*r25+1j*r16+r26)
							self.H0[col6,row2,ik,jk,kk] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)
							self.H0[row2,col7,ik,jk,kk] =-0.5*(r15-1j*r25-1j*r16-r26)
							self.H0[col7,row2,ik,jk,kk] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)
							########### d-1 COEFFICIENTS ##############################
							self.H0[row3,col1,ik,jk,kk] = 1./np.sqrt(2.)*(r10+1j*r20)
							self.H0[col1,row3,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r10+1j*r20)
							self.H0[row3,col2,ik,jk,kk] =-0.5*(r11-r22+1j*(r12+r21))
							self.H0[col2,row3,ik,jk,kk] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))
							self.H0[row3,col3,ik,jk,kk] = 0.5*(r11+r22+1j*(-r12+r21))
							self.H0[col3,row3,ik,jk,kk] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))
							self.H0[row3,col4,ik,jk,kk] = 0.5*(r13+1j*r23+1j*r14-r24)
							self.H0[col4,row3,ik,jk,kk] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)
							self.H0[row3,col5,ik,jk,kk] = 0.5*(r13+1j*r23-1j*r14+r24)
							self.H0[col5,row3,ik,jk,kk] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)
							self.H0[row3,col6,ik,jk,kk] =-0.5*(r15+1j*r25+1j*r16-r26)
							self.H0[col6,row3,ik,jk,kk] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)
							self.H0[row3,col7,ik,jk,kk] = 0.5*(r15+1j*r25-1j*r16+r26)
							self.H0[col7,row3,ik,jk,kk] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)
							########### d2 COEFFICIENTS ###############################
							self.H0[row4,col1,ik,jk,kk] = 1./np.sqrt(2.)*(r30-1j*r40)
							self.H0[col1,row4,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r30-1j*r40)
							self.H0[row4,col2,ik,jk,kk] =-0.5*(r31+r42+1j*(r32-r41))
							self.H0[col2,row4,ik,jk,kk] =-0.5*np.conjugate(r31+r42+1j*(r32-r41))
							self.H0[row4,col3,ik,jk,kk] = 0.5*(r31-r32-1j*(r32+r41))
							self.H0[col3,row4,ik,jk,kk] = 0.5*np.conjugate(r31-r32-1j*(r32+r41))
							self.H0[row4,col4,ik,jk,kk] = 0.5*(r33-1j*r43+1j*r34+r44)
							self.H0[col4,row4,ik,jk,kk] = 0.5*np.conjugate(r33-1j*r43+1j*r34+r44)
							self.H0[row4,col5,ik,jk,kk] = 0.5*(r33-1j*r43-1j*r34-r44)
							self.H0[col5,row4,ik,jk,kk] = 0.5*np.conjugate(r33-1j*r43-1j*r34-r44)
							self.H0[row4,col6,ik,jk,kk] =-0.5*(r35-1j*r45+1j*r36+r46)
							self.H0[col6,row4,ik,jk,kk] =-0.5*np.conjugate(r35-1j*r45+1j*r36+r46)
							self.H0[row4,col7,ik,jk,kk] = 0.5*(r35-1j*r45-1j*r36-r46)
							self.H0[col7,row4,ik,jk,kk] = 0.5*np.conjugate(r35-1j*r45-1j*r36-r46)
							########### d-2 COEFFICIENTS ##############################
							self.H0[row5,col1,ik,jk,kk] = 1./np.sqrt(2.)*(r30+1j*r40)
							self.H0[col1,row5,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r30+1j*r40)
							self.H0[row5,col2,ik,jk,kk] =-0.5*(r31-r42+1j*(r32+r41))
							self.H0[col2,row5,ik,jk,kk] =-0.5*np.conjugate(r31-r42+1j*(r32+r41))
							self.H0[row5,col3,ik,jk,kk] = 0.5*(r31+r42+1j*(-r32+r41))
							self.H0[col3,row5,ik,jk,kk] = 0.5*np.conjugate(r31+r42+1j*(-r32+r41))
							self.H0[row5,col4,ik,jk,kk] = 0.5*(r33+1j*r43+1j*r34-r44)
							self.H0[col4,row5,ik,jk,kk] = 0.5*np.conjugate(r33+1j*r43+1j*r34-r44)
							self.H0[row5,col5,ik,jk,kk] = 0.5*(r33+1j*r43-1j*r34+r44)
							self.H0[col5,row5,ik,jk,kk] = 0.5*np.conjugate(r33+1j*r43-1j*r34+r44)
							self.H0[row5,col6,ik,jk,kk] =-0.5*(r35+1j*r45+1j*r36-r46)
							self.H0[col6,row5,ik,jk,kk] =-0.5*np.conjugate(r35+1j*r45+1j*r36-r46)
							self.H0[row5,col7,ik,jk,kk] = 0.5*(r35+1j*r45-1j*r36+r46)
							self.H0[col7,row5,ik,jk,kk] = 0.5*np.conjugate(r35+1j*r45-1j*r36+r46)
						# iterate iik
						iik = iik + 1
	# (f,d) SK integrals
	def set_fd_hopping_mtxel(self, Site1, Site2, tdf, siteslist, kg, Unitcell, MatrixEntry):
		# (f,d) orbitals
		l1 = 3
		l2 = 2
		# SK integrals
		[tdf0, tdf1, tdf2] = tdf
		###   dimensions
		if kg.D == 0:
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r00 = 0.
			r10 = 0.
			r20 = 0.
			r30 = 0.
			r40 = 0.
			r01 = 0.
			r11 = 0.
			r21 = 0.
			r31 = 0.
			r41 = 0.
			r02 = 0.
			r12 = 0.
			r22 = 0.
			r32 = 0.
			r42 = 0.
			r03 = 0.
			r13 = 0.
			r23 = 0.
			r33 = 0.
			r43 = 0.
			r04 = 0.
			r14 = 0.
			r24 = 0.
			r34 = 0.
			r44 = 0.
			r05 = 0.
			r15 = 0.
			r25 = 0.
			r35 = 0.
			r45 = 0.
			r06 = 0.
			r16 = 0.
			r26 = 0.
			r36 = 0.
			r46 = 0.
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					r00 = 1./4*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(5.*lcoef[2]**2 - 3.)*tdf0 - 3./np.sqrt(8)*lcoef[2]*(5.*lcoef[2]**2 - 1.)*(lcoef[2]**2 - 1.)*tdf1 + 1./4*np.sqrt(45)*lcoef[2]*(lcoef[2]**2 -1.)**2 *tdf2
					r10 = 0.5*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(5.*lcoef[2]**2 - 3.)*tdf0 - np.sqrt(3./8)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2
					r20 = 0.5*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./8)*lcoef[1]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2
					r30 = 1./4*np.sqrt(3.)*lcoef[2]*(5.*lcoef[2]**2 -3.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(3./8)*lcoef[2]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1./4*np.sqrt(15.)*lcoef[2]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2
					r40 = 0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./2)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 +1.)*tdf2
					r01 = np.sqrt(3./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[0]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2
					r11 = 3./np.sqrt(8.)*lcoef[0]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[0]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -2.)*tdf2
					r21 = 3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2
					r31 = 3./np.sqrt(32.)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*lcoef[0]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[1]**2 -4.*lcoef[2]**2)*tdf1 + np.sqrt(5./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[2]**2 +1.) - 4.*lcoef[2]**2)*tdf2
					r41 = 3./np.sqrt(8.)*lcoef[0]**2 *lcoef[1]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*((6.*lcoef[0]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[1]*(lcoef[0]**2 *(3.*lcoef[2]**2 +1.) - 2.*lcoef[2]**2)*tdf2
					r02 = np.sqrt(3./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[1]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2
					r12 = 3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2
					r22 = 3./np.sqrt(8.)*lcoef[1]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[1]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -2.)*tdf2
					r32 = 3./np.sqrt(32.)*lcoef[1]*(lcoef[0]**2 - lcoef[1]**2)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 4.*lcoef[2]**2 - 2.*lcoef[0]**2)*tdf1 + np.sqrt(5./32)*lcoef[1]*((lcoef[0]**2 - lcoef[1]**2)*(3.*lcoef[2]**2 +1.) + 4.*lcoef[2]**2)*tdf2
					r42 = 3./np.sqrt(8.)*lcoef[0]*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[0]*((6.*lcoef[1]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[2]**2 + 1.) - 2.*lcoef[2]**2)*tdf2
					r03 = 1./4*np.sqrt(15.)*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf0 - np.sqrt(15./8)*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + np.sqrt(3.)/4*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2
					r13 = 0.5*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[0]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) - 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[0]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[1]**2 - 2.*lcoef[2]**2)*tdf2
					r23 = 0.5*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[1]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[1]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) - 4.*lcoef[0]**2 + 2.*lcoef[2]**2)*tdf2
					r33 = 1./4*np.sqrt(45.)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)**2 *tdf0 - np.sqrt(5./8)*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 2.*lcoef[2]**2 - 2.)*tdf1 + 1./4*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 8.*lcoef[2]**2 -4.)*tdf2
					r43 = 0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2
					r04 = 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf0 - np.sqrt(15./2)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf2
					r14 = np.sqrt(45.)*lcoef[0]**2 *lcoef[1]*lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[1]*(6.*lcoef[0]**2 *lcoef[2]**2 + lcoef[1]**2 -1.)*tdf1 + lcoef[1]*(3.*lcoef[0]**2 *lcoef[2]**2 + 2.*lcoef[1]**2 -1.)*tdf2
					r24 = np.sqrt(45.)*lcoef[0]*lcoef[1]**2 *lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[0]*(6.*lcoef[1]**2 *lcoef[2]**2 + lcoef[0]**2 -1.)*tdf1 + lcoef[0]*(3.*lcoef[1]**2 *lcoef[2]**2 +2.*lcoef[0]**2 -1.)*tdf2
					r34 = 0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2
					r44 = np.sqrt(45.)*lcoef[0]**2 *lcoef[1]**2 *lcoef[2]*tdf0 - np.sqrt(5./2)*lcoef[2]*(6.*lcoef[0]**2 *lcoef[1]**2 +lcoef[2]**2 -1.)*tdf1 + lcoef[2]*(3.*lcoef[0]**2 *lcoef[1]**2 +2.*lcoef[2]**2 -1.)*tdf2
					r05 = np.sqrt(5./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[0]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf2
					r15 = np.sqrt(15./8)*lcoef[0]**2 *lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[0]**2 *(lcoef[0]**2 - 3.*lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[0]**2 +2.*lcoef[1]**2)*tdf2
					r25 = np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +4.)*tdf2
					r35 = np.sqrt(15./32)*lcoef[0]*(lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*((lcoef[0]**2 - lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) - lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) +4.*lcoef[2]**2)*tdf2
					r45 = np.sqrt(15./8)*lcoef[0]**2 *lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*(2.*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[1]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[2]**2)*tdf2
					r06 = np.sqrt(5./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[1]*(lcoef[2]**2 +1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf2
					r16 = np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -4.)*tdf2
					r26 = np.sqrt(15./8)*lcoef[1]**2 *lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[0]**2 + 2.*lcoef[1]**2)*tdf2
					r36 = np.sqrt(15./32)*lcoef[1]*(lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./32)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[2]**2)*tdf2
					r46 = np.sqrt(15./8)*lcoef[0]*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[2]**2)*tdf2
			for ms in [-0.5, 0.5]:
				row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
				row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
				row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
				row6 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
				row7 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
				col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
				col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
				col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
				########### d0 COEFFICIENTS #############################
				self.H0[row1,col1] = r00*(-1)**(l1+l2)
				self.H0[col1,row1] = np.conjugate(r00)*(-1)**(l1+l2)
				self.H0[row2,col1] =-1./np.sqrt(2.)*(r01+1j*r02)*(-1)**(l1+l2)
				self.H0[col1,row2] =-1./np.sqrt(2.)*np.conjugate(r01+1j*r02)*(-1)**(l1+l2)
				self.H0[row3,col1] = 1./np.sqrt(2.)*(r01-1j*r02)*(-1)**(l1+l2)
				self.H0[col1,row3] = 1./np.sqrt(2.)*np.conjugate(r01-1j*r02)*(-1)**(l1+l2)
				self.H0[row4,col1] = 1./np.sqrt(2.)*(r03+1j*r04)*(-1)**(l1+l2)
				self.H0[col1,row4] = 1./np.sqrt(2.)*np.conjugate(r03+1j*r04)*(-1)**(l1+l2)
				self.H0[row5,col1] = 1./np.sqrt(2.)*(r03-1j*r04)*(-1)**(l1+l2)
				self.H0[col1,row5] = 1./np.sqrt(2.)*np.conjugate(r03-1j*r04)*(-1)**(l1+l2)
				self.H0[row6,col1] =-1./np.sqrt(2.)*(r05+1j*r06)*(-1)**(l1+l2)
				self.H0[col1,row6] =-1./np.sqrt(2.)*np.conjugate(r05+1j*r06)*(-1)**(l1+l2)
				self.H0[row7,col1] = 1./np.sqrt(2.)*(r05-1j*r06)*(-1)**(l1+l2)
				self.H0[col1,row7] = 1./np.sqrt(2.)*np.conjugate(r05-1j*r06)*(-1)**(l1+l2)
				########### d1 COEFFICIENTS ##############################
				self.H0[row1,col2] =-1./np.sqrt(2.)*(r10-1j*r20)*(-1)**(l1+l2)
				self.H0[col2,row1] =-1./np.sqrt(2.)*np.conjugate(r10-1j*r20)*(-1)**(l1+l2)
				self.H0[row2,col2] = 0.5*(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
				self.H0[col2,row2] = 0.5*np.conjugate(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
				self.H0[row3,col2] =-0.5*(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
				self.H0[col2,row3] =-0.5*np.conjugate(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
				self.H0[row4,col2] =-0.5*(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
				self.H0[col2,row4] =-0.5*np.conjugate(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
				self.H0[row5,col2] =-0.5*(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
				self.H0[col2,row5] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
				self.H0[row6,col2] = 0.5*(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
				self.H0[col2,row6] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
				self.H0[row7,col2] =-0.5*(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
				self.H0[col2,row7] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
				########### d-1 COEFFICIENTS ##############################
				self.H0[row1,col3] = 1./np.sqrt(2.)*(r10+1j*r20)*(-1)**(l1+l2)
				self.H0[col3,row1] = 1./np.sqrt(2.)*np.conjugate(r10+1j*r20)*(-1)**(l1+l2)
				self.H0[row2,col3] =-0.5*(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
				self.H0[col3,row2] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
				self.H0[row3,col3] = 0.5*(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
				self.H0[col3,row3] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
				self.H0[row4,col3] = 0.5*(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
				self.H0[col3,row4] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
				self.H0[row5,col3] = 0.5*(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
				self.H0[col3,row5] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
				self.H0[row6,col3] =-0.5*(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
				self.H0[col3,row6] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
				self.H0[row7,col3] = 0.5*(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
				self.H0[col3,row7] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
				########### d2 COEFFICIENTS ###############################
				self.H0[row1,col4] = 1./np.sqrt(2.)*(r30-1j*r40)*(-1)**(l1+l2)
				self.H0[col4,row1] = 1./np.sqrt(2.)*np.conjugate(r30-1j*r40)*(-1)**(l1+l2)
				self.H0[row2,col4] =-0.5*(r31+r42+1j*(r32-r41))*(-1)**(l1+l2)
				self.H0[col4,row2] =-0.5*np.conjugate(r31+r42+1j*(r32-r41))*(-1)**(l1+l2)
				self.H0[row3,col4] = 0.5*(r31-r32-1j*(r32+r41))*(-1)**(l1+l2)
				self.H0[col4,row3] = 0.5*np.conjugate(r31-r32-1j*(r32+r41))*(-1)**(l1+l2)
				self.H0[row4,col4] = 0.5*(r33-1j*r43+1j*r34+r44)*(-1)**(l1+l2)
				self.H0[col4,row4] = 0.5*np.conjugate(r33-1j*r43+1j*r34+r44)*(-1)**(l1+l2)
				self.H0[row5,col4] = 0.5*(r33-1j*r43-1j*r34-r44)*(-1)**(l1+l2)
				self.H0[col4,row5] = 0.5*np.conjugate(r33-1j*r43-1j*r34-r44)*(-1)**(l1+l2)
				self.H0[row6,col4] =-0.5*(r35-1j*r45+1j*r36+r46)*(-1)**(l1+l2)
				self.H0[col4,row6] =-0.5*np.conjugate(r35-1j*r45+1j*r36+r46)*(-1)**(l1+l2)
				self.H0[row7,col4] = 0.5*(r35-1j*r45-1j*r36-r46)*(-1)**(l1+l2)
				self.H0[col4,row7] = 0.5*np.conjugate(r35-1j*r45-1j*r36-r46)*(-1)**(l1+l2)
				########### d-2 COEFFICIENTS ##############################
				self.H0[row1,col5] = 1./np.sqrt(2.)*(r30+1j*r40)*(-1)**(l1+l2)
				self.H0[col5,row1] = 1./np.sqrt(2.)*np.conjugate(r30+1j*r40)*(-1)**(l1+l2)
				self.H0[row2,col5] =-0.5*(r31-r42+1j*(r32+r41))*(-1)**(l1+l2)
				self.H0[col5,row2] =-0.5*np.conjugate(r31-r42+1j*(r32+r41))*(-1)**(l1+l2)
				self.H0[row3,col5] = 0.5*(r31+r42+1j*(-r32+r41))*(-1)**(l1+l2)
				self.H0[col5,row3] = 0.5*np.conjugate(r31+r42+1j*(-r32+r41))*(-1)**(l1+l2)
				self.H0[row4,col5] = 0.5*(r33+1j*r43+1j*r34-r44)*(-1)**(l1+l2)
				self.H0[col5,row4] = 0.5*np.conjugate(r33+1j*r43+1j*r34-r44)*(-1)**(l1+l2)
				self.H0[row5,col5] = 0.5*(r33+1j*r43-1j*r34+r44)*(-1)**(l1+l2)
				self.H0[col5,row5] = 0.5*np.conjugate(r33+1j*r43-1j*r34+r44)*(-1)**(l1+l2)
				self.H0[row6,col5] =-0.5*(r35+1j*r45+1j*r36-r46)*(-1)**(l1+l2)
				self.H0[col5,row6] =-0.5*np.conjugate(r35+1j*r45+1j*r36-r46)*(-1)**(l1+l2)
				self.H0[row7,col5] = 0.5*(r35+1j*r45-1j*r36+r46)*(-1)**(l1+l2)
				self.H0[col5,row7] = 0.5*np.conjugate(r35+1j*r45-1j*r36+r46)*(-1)**(l1+l2)
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r00 = 0.
				r10 = 0.
				r20 = 0.
				r30 = 0.
				r40 = 0.
				r01 = 0.
				r11 = 0.
				r21 = 0.
				r31 = 0.
				r41 = 0.
				r02 = 0.
				r12 = 0.
				r22 = 0.
				r32 = 0.
				r42 = 0.
				r03 = 0.
				r13 = 0.
				r23 = 0.
				r33 = 0.
				r43 = 0.
				r04 = 0.
				r14 = 0.
				r24 = 0.
				r34 = 0.
				r44 = 0.
				r05 = 0.
				r15 = 0.
				r25 = 0.
				r35 = 0.
				r45 = 0.
				r06 = 0.
				r16 = 0.
				r26 = 0.
				r36 = 0.
				r46 = 0.
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						r00 = (1./4*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(5.*lcoef[2]**2 - 3.)*tdf0 - 3./np.sqrt(8)*lcoef[2]*(5.*lcoef[2]**2 - 1.)*(lcoef[2]**2 - 1.)*tdf1 + 1./4*np.sqrt(45)*lcoef[2]*(lcoef[2]**2 -1.)**2 *tdf2)*cmath.exp(1j*kR) + r00
						r10 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(5.*lcoef[2]**2 - 3.)*tdf0 - np.sqrt(3./8)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r10
						r20 = (0.5*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./8)*lcoef[1]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r20
						r30 = (1./4*np.sqrt(3.)*lcoef[2]*(5.*lcoef[2]**2 -3.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(3./8)*lcoef[2]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1./4*np.sqrt(15.)*lcoef[2]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r30
						r40 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./2)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 +1.)*tdf2)*cmath.exp(1j*kR) + r40
						r01 = (np.sqrt(3./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[0]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r01
						r11 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[0]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r11
						r21 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r21
						r31 = (3./np.sqrt(32.)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*lcoef[0]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[1]**2 -4.*lcoef[2]**2)*tdf1 + np.sqrt(5./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[2]**2 +1.) - 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r31
						r41 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[1]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*((6.*lcoef[0]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[1]*(lcoef[0]**2 *(3.*lcoef[2]**2 +1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r41
						r02 = (np.sqrt(3./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[1]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r02
						r12 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r12
						r22 = (3./np.sqrt(8.)*lcoef[1]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[1]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r22
						r32 = (3./np.sqrt(32.)*lcoef[1]*(lcoef[0]**2 - lcoef[1]**2)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 4.*lcoef[2]**2 - 2.*lcoef[0]**2)*tdf1 + np.sqrt(5./32)*lcoef[1]*((lcoef[0]**2 - lcoef[1]**2)*(3.*lcoef[2]**2 +1.) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r32
						r42 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[0]*((6.*lcoef[1]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[2]**2 + 1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r42
						r03 = (1./4*np.sqrt(15.)*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf0 - np.sqrt(15./8)*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + np.sqrt(3.)/4*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r03
						r13 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[0]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) - 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[0]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[1]**2 - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r13
						r23 = (0.5*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[1]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[1]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) - 4.*lcoef[0]**2 + 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r23
						r33 = (1./4*np.sqrt(45.)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)**2 *tdf0 - np.sqrt(5./8)*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 2.*lcoef[2]**2 - 2.)*tdf1 + 1./4*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 8.*lcoef[2]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r33
						r43 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r43
						r04 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf0 - np.sqrt(15./2)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r04
						r14 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]*lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[1]*(6.*lcoef[0]**2 *lcoef[2]**2 + lcoef[1]**2 -1.)*tdf1 + lcoef[1]*(3.*lcoef[0]**2 *lcoef[2]**2 + 2.*lcoef[1]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r14
						r24 = (np.sqrt(45.)*lcoef[0]*lcoef[1]**2 *lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[0]*(6.*lcoef[1]**2 *lcoef[2]**2 + lcoef[0]**2 -1.)*tdf1 + lcoef[0]*(3.*lcoef[1]**2 * lcoef[2]**2 +2.*lcoef[0]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r24
						r34 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r34
						r44 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]**2 *lcoef[2]*tdf0 - np.sqrt(5./2)*lcoef[2]*(6.*lcoef[0]**2 *lcoef[1]**2 +lcoef[2]**2 -1.)*tdf1 + lcoef[2]*(3.*lcoef[0]**2 *lcoef[1]**2 +2.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r44
						r05 = (np.sqrt(5./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[0]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r05
						r15 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[0]**2 *(lcoef[0]**2 - 3.*lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[0]**2 +2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r15
						r25 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +4.)*tdf2)*cmath.exp(1j*kR) + r25
						r35 = (np.sqrt(15./32)*lcoef[0]*(lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*((lcoef[0]**2 - lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) - lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) +4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r35
						r45 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*(2.*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[1]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r45
						r06 = (np.sqrt(5./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[1]*(lcoef[2]**2 +1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r06
						r16 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r16
						r26 = (np.sqrt(15./8)*lcoef[1]**2 *lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[0]**2 + 2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r26
						r36 = (np.sqrt(15./32)*lcoef[1]*(lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./32)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r36
						r46 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r46
				for ms in [-0.5, 0.5]:
					row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
					row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
					row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
					row6 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
					row7 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
					col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
					col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
					col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
					########### d0 COEFFICIENTS #############################
					self.H0[row1,col1,ik] = r00*(-1)**(l1+l2)
					self.H0[col1,row1,ik] = np.conjugate(r00)*(-1)**(l1+l2)
					self.H0[row2,col1,ik] =-1./np.sqrt(2.)*(r01+1j*r02)*(-1)**(l1+l2)
					self.H0[col1,row2,ik] =-1./np.sqrt(2.)*np.conjugate(r01+1j*r02)*(-1)**(l1+l2)
					self.H0[row3,col1,ik] = 1./np.sqrt(2.)*(r01-1j*r02)*(-1)**(l1+l2)
					self.H0[col1,row3,ik] = 1./np.sqrt(2.)*np.conjugate(r01-1j*r02)*(-1)**(l1+l2)
					self.H0[row4,col1,ik] = 1./np.sqrt(2.)*(r03+1j*r04)*(-1)**(l1+l2)
					self.H0[col1,row4,ik] = 1./np.sqrt(2.)*np.conjugate(r03+1j*r04)*(-1)**(l1+l2)
					self.H0[row5,col1,ik] = 1./np.sqrt(2.)*(r03-1j*r04)*(-1)**(l1+l2)
					self.H0[col1,row5,ik] = 1./np.sqrt(2.)*np.conjugate(r03-1j*r04)*(-1)**(l1+l2)
					self.H0[row6,col1,ik] =-1./np.sqrt(2.)*(r05+1j*r06)*(-1)**(l1+l2)
					self.H0[col1,row6,ik] =-1./np.sqrt(2.)*np.conjugate(r05+1j*r06)*(-1)**(l1+l2)
					self.H0[row7,col1,ik] = 1./np.sqrt(2.)*(r05-1j*r06)*(-1)**(l1+l2)
					self.H0[col1,row7,ik] = 1./np.sqrt(2.)*np.conjugate(r05-1j*r06)*(-1)**(l1+l2)
					########### d1 COEFFICIENTS ##############################
					self.H0[row1,col2,ik] =-1./np.sqrt(2.)*(r10-1j*r20)*(-1)**(l1+l2)
					self.H0[col2,row1,ik] =-1./np.sqrt(2.)*np.conjugate(r10-1j*r20)*(-1)**(l1+l2)
					self.H0[row2,col2,ik] = 0.5*(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
					self.H0[col2,row2,ik] = 0.5*np.conjugate(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
					self.H0[row3,col2,ik] =-0.5*(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
					self.H0[col2,row3,ik] =-0.5*np.conjugate(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
					self.H0[row4,col2,ik] =-0.5*(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
					self.H0[col2,row4,ik] =-0.5*np.conjugate(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
					self.H0[row5,col2,ik] =-0.5*(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
					self.H0[col2,row5,ik] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
					self.H0[row6,col2,ik] = 0.5*(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
					self.H0[col2,row6,ik] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
					self.H0[row7,col2,ik] =-0.5*(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
					self.H0[col2,row7,ik] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
					########### d-1 COEFFICIENTS ##############################
					self.H0[row1,col3,ik] = 1./np.sqrt(2.)*(r10+1j*r20)*(-1)**(l1+l2)
					self.H0[col3,row1,ik] = 1./np.sqrt(2.)*np.conjugate(r10+1j*r20)*(-1)**(l1+l2)
					self.H0[row2,col3,ik] =-0.5*(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
					self.H0[col3,row2,ik] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
					self.H0[row3,col3,ik] = 0.5*(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
					self.H0[col3,row3,ik] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
					self.H0[row4,col3,ik] = 0.5*(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
					self.H0[col3,row4,ik] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
					self.H0[row5,col3,ik] = 0.5*(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
					self.H0[col3,row5,ik] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
					self.H0[row6,col3,ik] =-0.5*(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
					self.H0[col3,row6,ik] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
					self.H0[row7,col3,ik] = 0.5*(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
					self.H0[col3,row7,ik] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
					########### d2 COEFFICIENTS ###############################
					self.H0[row1,col4,ik] = 1./np.sqrt(2.)*(r30-1j*r40)*(-1)**(l1+l2)
					self.H0[col4,row1,ik] = 1./np.sqrt(2.)*np.conjugate(r30-1j*r40)*(-1)**(l1+l2)
					self.H0[row2,col4,ik] =-0.5*(r31+r42+1j*(r32-r41))*(-1)**(l1+l2)
					self.H0[col4,row2,ik] =-0.5*np.conjugate(r31+r42+1j*(r32-r41))*(-1)**(l1+l2)
					self.H0[row3,col4,ik] = 0.5*(r31-r32-1j*(r32+r41))*(-1)**(l1+l2)
					self.H0[col4,row3,ik] = 0.5*np.conjugate(r31-r32-1j*(r32+r41))*(-1)**(l1+l2)
					self.H0[row4,col4,ik] = 0.5*(r33-1j*r43+1j*r34+r44)*(-1)**(l1+l2)
					self.H0[col4,row4,ik] = 0.5*np.conjugate(r33-1j*r43+1j*r34+r44)*(-1)**(l1+l2)
					self.H0[row5,col4,ik] = 0.5*(r33-1j*r43-1j*r34-r44)*(-1)**(l1+l2)
					self.H0[col4,row5,ik] = 0.5*np.conjugate(r33-1j*r43-1j*r34-r44)*(-1)**(l1+l2)
					self.H0[row6,col4,ik] =-0.5*(r35-1j*r45+1j*r36+r46)*(-1)**(l1+l2)
					self.H0[col4,row6,ik] =-0.5*np.conjugate(r35-1j*r45+1j*r36+r46)*(-1)**(l1+l2)
					self.H0[row7,col4,ik] = 0.5*(r35-1j*r45-1j*r36-r46)*(-1)**(l1+l2)
					self.H0[col4,row7,ik] = 0.5*np.conjugate(r35-1j*r45-1j*r36-r46)*(-1)**(l1+l2)
					########### d-2 COEFFICIENTS ##############################
					self.H0[row1,col5,ik] = 1./np.sqrt(2.)*(r30+1j*r40)*(-1)**(l1+l2)
					self.H0[col5,row1,ik] = 1./np.sqrt(2.)*np.conjugate(r30+1j*r40)*(-1)**(l1+l2)
					self.H0[row2,col5,ik] =-0.5*(r31-r42+1j*(r32+r41))*(-1)**(l1+l2)
					self.H0[col5,row2,ik] =-0.5*np.conjugate(r31-r42+1j*(r32+r41))*(-1)**(l1+l2)
					self.H0[row3,col5,ik] = 0.5*(r31+r42+1j*(-r32+r41))*(-1)**(l1+l2)
					self.H0[col5,row3,ik] = 0.5*np.conjugate(r31+r42+1j*(-r32+r41))*(-1)**(l1+l2)
					self.H0[row4,col5,ik] = 0.5*(r33+1j*r43+1j*r34-r44)*(-1)**(l1+l2)
					self.H0[col5,row4,ik] = 0.5*np.conjugate(r33+1j*r43+1j*r34-r44)*(-1)**(l1+l2)
					self.H0[row5,col5,ik] = 0.5*(r33+1j*r43-1j*r34+r44)*(-1)**(l1+l2)
					self.H0[col5,row5,ik] = 0.5*np.conjugate(r33+1j*r43-1j*r34+r44)*(-1)**(l1+l2)
					self.H0[row6,col5,ik] =-0.5*(r35+1j*r45+1j*r36-r46)*(-1)**(l1+l2)
					self.H0[col5,row6,ik] =-0.5*np.conjugate(r35+1j*r45+1j*r36-r46)*(-1)**(l1+l2)
					self.H0[row7,col5,ik] = 0.5*(r35+1j*r45-1j*r36+r46)*(-1)**(l1+l2)
					self.H0[col5,row7,ik] = 0.5*np.conjugate(r35+1j*r45-1j*r36+r46)*(-1)**(l1+l2)
		elif kg.D == 2:
			e = Unitcell.rcv
			if kg.nkpts[0] == 0:
				nk1 = kg.nkpts[1]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[1] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[2]
			elif kg.nkpts[2] == 0:
				nk1 = kg.nkpts[0]
				nk2 = kg.nkpts[1]
			else:
				print("wrong nkpts")
				sys.exit(1)
			# run over k pts
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					kpt = kg.kgrid[iik]
					k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
					# nn data site1
					nndata_s1 = Unitcell.NNlist[Site1-1]
					r00 = 0.
					r10 = 0.
					r20 = 0.
					r30 = 0.
					r40 = 0.
					r01 = 0.
					r11 = 0.
					r21 = 0.
					r31 = 0.
					r41 = 0.
					r02 = 0.
					r12 = 0.
					r22 = 0.
					r32 = 0.
					r42 = 0.
					r03 = 0.
					r13 = 0.
					r23 = 0.
					r33 = 0.
					r43 = 0.
					r04 = 0.
					r14 = 0.
					r24 = 0.
					r34 = 0.
					r44 = 0.
					r05 = 0.
					r15 = 0.
					r25 = 0.
					r35 = 0.
					r45 = 0.
					r06 = 0.
					r16 = 0.
					r26 = 0.
					r36 = 0.
					r46 = 0.
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							r00 = (1./4*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(5.*lcoef[2]**2 - 3.)*tdf0 - 3./np.sqrt(8)*lcoef[2]*(5.*lcoef[2]**2 - 1.)*(lcoef[2]**2 - 1.)*tdf1 + 1./4*np.sqrt(45)*lcoef[2]*(lcoef[2]**2 -1.)**2 *tdf2)*cmath.exp(1j*kR) + r00
							r10 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(5.*lcoef[2]**2 - 3.)*tdf0 - np.sqrt(3./8)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r10
							r20 = (0.5*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./8)*lcoef[1]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r20
							r30 = (1./4*np.sqrt(3.)*lcoef[2]*(5.*lcoef[2]**2 -3.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(3./8)*lcoef[2]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1./4*np.sqrt(15.)*lcoef[2]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r30
							r40 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./2)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 +1.)*tdf2)*cmath.exp(1j*kR) + r40
							r01 = (np.sqrt(3./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[0]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r01
							r11 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[0]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r11
							r21 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r21
							r31 = (3./np.sqrt(32.)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*lcoef[0]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[1]**2 -4.*lcoef[2]**2)*tdf1 + np.sqrt(5./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[2]**2 +1.) - 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r31
							r41 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[1]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*((6.*lcoef[0]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[1]*(lcoef[0]**2 *(3.*lcoef[2]**2 +1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r41
							r02 = (np.sqrt(3./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[1]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r02
							r12 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r12
							r22 = (3./np.sqrt(8.)*lcoef[1]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[1]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r22
							r32 = (3./np.sqrt(32.)*lcoef[1]*(lcoef[0]**2 - lcoef[1]**2)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 4.*lcoef[2]**2 - 2.*lcoef[0]**2)*tdf1 + np.sqrt(5./32)*lcoef[1]*((lcoef[0]**2 - lcoef[1]**2)*(3.*lcoef[2]**2 +1.) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r32
							r42 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[0]*((6.*lcoef[1]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[2]**2 + 1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r42
							r03 = (1./4*np.sqrt(15.)*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf0 - np.sqrt(15./8)*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + np.sqrt(3.)/4*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r03
							r13 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[0]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) - 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[0]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[1]**2 - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r13
							r23 = (0.5*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[1]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[1]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) - 4.*lcoef[0]**2 + 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r23
							r33 = (1./4*np.sqrt(45.)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)**2 *tdf0 - np.sqrt(5./8)*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 2.*lcoef[2]**2 - 2.)*tdf1 + 1./4*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 8.*lcoef[2]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r33
							r43 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r43
							r04 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf0 - np.sqrt(15./2)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r04
							r14 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]*lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[1]*(6.*lcoef[0]**2 *lcoef[2]**2 + lcoef[1]**2 -1.)*tdf1 + lcoef[1]*(3.*lcoef[0]**2 *lcoef[2]**2 + 2.*lcoef[1]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r14
							r24 = (np.sqrt(45.)*lcoef[0]*lcoef[1]**2 *lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[0]*(6.*lcoef[1]**2 *lcoef[2]**2 + lcoef[0]**2 -1.)*tdf1 + lcoef[0]*(3.*lcoef[1]**2 *lcoef[2]**2 +2.*lcoef[0]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r24
							r34 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r34
							r44 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]**2 *lcoef[2]*tdf0 - np.sqrt(5./2)*lcoef[2]*(6.*lcoef[0]**2 *lcoef[1]**2 +lcoef[2]**2 -1.)*tdf1 + lcoef[2]*(3.*lcoef[0]**2 *lcoef[1]**2 +2.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r44
							r05 = (np.sqrt(5./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[0]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r05
							r15 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[0]**2 *(lcoef[0]**2 - 3.*lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[0]**2 +2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r15
							r25 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +4.)*tdf2)*cmath.exp(1j*kR) + r25
							r35 = (np.sqrt(15./32)*lcoef[0]*(lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*((lcoef[0]**2 - lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) - lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) +4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r35
							r45 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*(2.*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[1]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r45
							r06 = (np.sqrt(5./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[1]*(lcoef[2]**2 +1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r06
							r16 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r16
							r26 = (np.sqrt(15./8)*lcoef[1]**2 *lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[0]**2 + 2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r26
							r36 = (np.sqrt(15./32)*lcoef[1]*(lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./32)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r36
							r46 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r46
					for ms in [-0.5, 0.5]:
						row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
						row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
						row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
						row6 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
						row7 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
						col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
						col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
						col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
						########### d0 COEFFICIENTS #############################
						self.H0[row1,col1,ik,jk] = r00*(-1)**(l1+l2)
						self.H0[col1,row1,ik,jk] = np.conjugate(r00)*(-1)**(l1+l2)
						self.H0[row2,col1,ik,jk] =-1./np.sqrt(2.)*(r01+1j*r02)*(-1)**(l1+l2)
						self.H0[col1,row2,ik,jk] =-1./np.sqrt(2.)*np.conjugate(r01+1j*r02)*(-1)**(l1+l2)
						self.H0[row3,col1,ik,jk] = 1./np.sqrt(2.)*(r01-1j*r02)*(-1)**(l1+l2)
						self.H0[col1,row3,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r01-1j*r02)*(-1)**(l1+l2)
						self.H0[row4,col1,ik,jk] = 1./np.sqrt(2.)*(r03+1j*r04)*(-1)**(l1+l2)
						self.H0[col1,row4,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r03+1j*r04)*(-1)**(l1+l2)
						self.H0[row5,col1,ik,jk] = 1./np.sqrt(2.)*(r03-1j*r04)*(-1)**(l1+l2)
						self.H0[col1,row5,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r03-1j*r04)*(-1)**(l1+l2)
						self.H0[row6,col1,ik,jk] =-1./np.sqrt(2.)*(r05+1j*r06)*(-1)**(l1+l2)
						self.H0[col1,row6,ik,jk] =-1./np.sqrt(2.)*np.conjugate(r05+1j*r06)*(-1)**(l1+l2)
						self.H0[row7,col1,ik,jk] = 1./np.sqrt(2.)*(r05-1j*r06)*(-1)**(l1+l2)
						self.H0[col1,row7,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r05-1j*r06)*(-1)**(l1+l2)
						########### d1 COEFFICIENTS ##############################
						self.H0[row1,col2,ik,jk] =-1./np.sqrt(2.)*(r10-1j*r20)*(-1)**(l1+l2)
						self.H0[col2,row1,ik,jk] =-1./np.sqrt(2.)*np.conjugate(r10-1j*r20)*(-1)**(l1+l2)
						self.H0[row2,col2,ik,jk] = 0.5*(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
						self.H0[col2,row2,ik,jk] = 0.5*np.conjugate(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
						self.H0[row3,col2,ik,jk] =-0.5*(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
						self.H0[col2,row3,ik,jk] =-0.5*np.conjugate(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
						self.H0[row4,col2,ik,jk] =-0.5*(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
						self.H0[col2,row4,ik,jk] =-0.5*np.conjugate(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
						self.H0[row5,col2,ik,jk] =-0.5*(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
						self.H0[col2,row5,ik,jk] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
						self.H0[row6,col2,ik,jk] = 0.5*(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
						self.H0[col2,row6,ik,jk] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
						self.H0[row7,col2,ik,jk] =-0.5*(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
						self.H0[col2,row7,ik,jk] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
						########### d-1 COEFFICIENTS ##############################
						self.H0[row1,col3,ik,jk] = 1./np.sqrt(2.)*(r10+1j*r20)*(-1)**(l1+l2)
						self.H0[col3,row1,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r10+1j*r20)*(-1)**(l1+l2)
						self.H0[row2,col3,ik,jk] =-0.5*(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
						self.H0[col3,row2,ik,jk] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
						self.H0[row3,col3,ik,jk] = 0.5*(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
						self.H0[col3,row3,ik,jk] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
						self.H0[row4,col3,ik,jk] = 0.5*(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
						self.H0[col3,row4,ik,jk] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
						self.H0[row5,col3,ik,jk] = 0.5*(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
						self.H0[col3,row5,ik,jk] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
						self.H0[row6,col3,ik,jk] =-0.5*(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
						self.H0[col3,row6,ik,jk] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
						self.H0[row7,col3,ik,jk] = 0.5*(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
						self.H0[col3,row7,ik,jk] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
						########### d2 COEFFICIENTS ###############################
						self.H0[row1,col4,ik,jk] = 1./np.sqrt(2.)*(r30-1j*r40)*(-1)**(l1+l2)
						self.H0[col4,row1,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r30-1j*r40)*(-1)**(l1+l2)
						self.H0[row2,col4,ik,jk] =-0.5*(r31+r42+1j*(r32-r41))*(-1)**(l1+l2)
						self.H0[col4,row2,ik,jk] =-0.5*np.conjugate(r31+r42+1j*(r32-r41))*(-1)**(l1+l2)
						self.H0[row3,col4,ik,jk] = 0.5*(r31-r32-1j*(r32+r41))*(-1)**(l1+l2)
						self.H0[col4,row3,ik,jk] = 0.5*np.conjugate(r31-r32-1j*(r32+r41))*(-1)**(l1+l2)
						self.H0[row4,col4,ik,jk] = 0.5*(r33-1j*r43+1j*r34+r44)*(-1)**(l1+l2)
						self.H0[col4,row4,ik,jk] = 0.5*np.conjugate(r33-1j*r43+1j*r34+r44)*(-1)**(l1+l2)
						self.H0[row5,col4,ik,jk] = 0.5*(r33-1j*r43-1j*r34-r44)*(-1)**(l1+l2)
						self.H0[col4,row5,ik,jk] = 0.5*np.conjugate(r33-1j*r43-1j*r34-r44)*(-1)**(l1+l2)
						self.H0[row6,col4,ik,jk] =-0.5*(r35-1j*r45+1j*r36+r46)*(-1)**(l1+l2)
						self.H0[col4,row6,ik,jk] =-0.5*np.conjugate(r35-1j*r45+1j*r36+r46)*(-1)**(l1+l2)
						self.H0[row7,col4,ik,jk] = 0.5*(r35-1j*r45-1j*r36-r46)*(-1)**(l1+l2)
						self.H0[col4,row7,ik,jk] = 0.5*np.conjugate(r35-1j*r45-1j*r36-r46)*(-1)**(l1+l2)
						########### d-2 COEFFICIENTS ##############################
						self.H0[row1,col5,ik,jk] = 1./np.sqrt(2.)*(r30+1j*r40)*(-1)**(l1+l2)
						self.H0[col5,row1,ik,jk] = 1./np.sqrt(2.)*np.conjugate(r30+1j*r40)*(-1)**(l1+l2)
						self.H0[row2,col5,ik,jk] =-0.5*(r31-r42+1j*(r32+r41))*(-1)**(l1+l2)
						self.H0[col5,row2,ik,jk] =-0.5*np.conjugate(r31-r42+1j*(r32+r41))*(-1)**(l1+l2)
						self.H0[row3,col5,ik,jk] = 0.5*(r31+r42+1j*(-r32+r41))*(-1)**(l1+l2)
						self.H0[col5,row3,ik,jk] = 0.5*np.conjugate(r31+r42+1j*(-r32+r41))*(-1)**(l1+l2)
						self.H0[row4,col5,ik,jk] = 0.5*(r33+1j*r43+1j*r34-r44)*(-1)**(l1+l2)
						self.H0[col5,row4,ik,jk] = 0.5*np.conjugate(r33+1j*r43+1j*r34-r44)*(-1)**(l1+l2)
						self.H0[row5,col5,ik,jk] = 0.5*(r33+1j*r43-1j*r34+r44)*(-1)**(l1+l2)
						self.H0[col5,row5,ik,jk] = 0.5*np.conjugate(r33+1j*r43-1j*r34+r44)*(-1)**(l1+l2)
						self.H0[row6,col5,ik,jk] =-0.5*(r35+1j*r45+1j*r36-r46)*(-1)**(l1+l2)
						self.H0[col5,row6,ik,jk] =-0.5*np.conjugate(r35+1j*r45+1j*r36-r46)*(-1)**(l1+l2)
						self.H0[row7,col5,ik,jk] = 0.5*(r35+1j*r45-1j*r36+r46)*(-1)**(l1+l2)
						self.H0[col5,row7,ik,jk] = 0.5*np.conjugate(r35+1j*r45-1j*r36+r46)*(-1)**(l1+l2)
					# iterate iik
					iik = iik + 1
		elif kg.D == 3:
			e = Unitcell.rcv
			[nk1, nk2, nk3] = kg.nkpts
			# run over k pts.
			iik = 0
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = kg.kgrid[iik]
						k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
						# nn data site1
						nndata_s1 = Unitcell.NNlist[Site1-1]
						r00 = 0.
						r10 = 0.
						r20 = 0.
						r30 = 0.
						r40 = 0.
						r01 = 0.
						r11 = 0.
						r21 = 0.
						r31 = 0.
						r41 = 0.
						r02 = 0.
						r12 = 0.
						r22 = 0.
						r32 = 0.
						r42 = 0.
						r03 = 0.
						r13 = 0.
						r23 = 0.
						r33 = 0.
						r43 = 0.
						r04 = 0.
						r14 = 0.
						r24 = 0.
						r34 = 0.
						r44 = 0.
						r05 = 0.
						r15 = 0.
						r25 = 0.
						r35 = 0.
						r45 = 0.
						r06 = 0.
						r16 = 0.
						r26 = 0.
						r36 = 0.
						r46 = 0.
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								r00 = (1./4*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(5.*lcoef[2]**2 - 3.)*tdf0 - 3./np.sqrt(8)*lcoef[2]*(5.*lcoef[2]**2 - 1.)*(lcoef[2]**2 - 1.)*tdf1 + 1./4*np.sqrt(45)*lcoef[2]*(lcoef[2]**2 -1.)**2 *tdf2)*cmath.exp(1j*kR) + r00
								r10 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(5.*lcoef[2]**2 - 3.)*tdf0 - np.sqrt(3./8)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r10
								r20 = (0.5*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./8)*lcoef[1]*(5.*lcoef[2]**2 -1.)*(2.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[1]*lcoef[2]**2 *(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r20
								r30 = (1./4*np.sqrt(3.)*lcoef[2]*(5.*lcoef[2]**2 -3.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(3./8)*lcoef[2]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1./4*np.sqrt(15.)*lcoef[2]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r30
								r40 = (0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -3.)*tdf0 - np.sqrt(3./2)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 +1.)*tdf2)*cmath.exp(1j*kR) + r40
								r01 = (np.sqrt(3./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[0]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[0]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r01
								r11 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[0]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r11
								r21 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r21
								r31 = (3./np.sqrt(32.)*lcoef[0]*(5.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*lcoef[0]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[1]**2 -4.*lcoef[2]**2)*tdf1 + np.sqrt(5./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[2]**2 +1.) - 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r31
								r41 = (3./np.sqrt(8.)*lcoef[0]**2 *lcoef[1]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*((6.*lcoef[0]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[1]*(lcoef[0]**2 *(3.*lcoef[2]**2 +1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r41
								r02 = (np.sqrt(3./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*np.sqrt(3.)*lcoef[1]*lcoef[2]**2 *(15.*lcoef[2]**2 -11.)*tdf1 + np.sqrt(15./32)*lcoef[1]*(lcoef[2]**2 -1.)*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r02
								r12 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(5.*lcoef[2]**2 -2.)*tdf1 + np.sqrt(45./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r12
								r22 = (3./np.sqrt(8.)*lcoef[1]**2 *lcoef[2]*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[2]*(lcoef[1]**2 *(30.*lcoef[2]**2 -11.) - 4.*lcoef[2]**2 + lcoef[0]**2)*tdf1 + np.sqrt(5./8)*lcoef[2]*(lcoef[2]**2 -1.)*(3.*lcoef[1]**2 -2.)*tdf2)*cmath.exp(1j*kR) + r22
								r32 = (3./np.sqrt(32.)*lcoef[1]*(lcoef[0]**2 - lcoef[1]**2)*(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[1]*(15.*lcoef[2]**2 *(lcoef[0]**2 - lcoef[1]**2) + 4.*lcoef[2]**2 - 2.*lcoef[0]**2)*tdf1 + np.sqrt(5./32)*lcoef[1]*((lcoef[0]**2 - lcoef[1]**2)*(3.*lcoef[2]**2 +1.) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r32
								r42 = (3./np.sqrt(8.)*lcoef[0]*lcoef[1]**2 *(5.*lcoef[2]**2 -1.)*tdf0 - 1./4*lcoef[0]*((6.*lcoef[1]**2 -1.)*(5.*lcoef[2]**2 -1.) + 4.*lcoef[1]**2)*tdf1 + np.sqrt(5./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[2]**2 + 1.) - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r42
								r03 = (1./4*np.sqrt(15.)*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf0 - np.sqrt(15./8)*lcoef[2]*(3.*lcoef[2]**2 - 1.)*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + np.sqrt(3.)/4*lcoef[2]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r03
								r13 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[0]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) - 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[0]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[1]**2 - 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r13
								r23 = (0.5*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(5./8)*lcoef[1]*((6.*lcoef[2]**2 -1.)*(lcoef[0]**2 - lcoef[1]**2) + 2.*lcoef[2]**2)*tdf1 + 0.5*lcoef[1]*(3.*lcoef[2]**2 *(lcoef[0]**2 -lcoef[1]**2) - 4.*lcoef[0]**2 + 2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r23
								r33 = (1./4*np.sqrt(45.)*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)**2 *tdf0 - np.sqrt(5./8)*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 2.*lcoef[2]**2 - 2.)*tdf1 + 1./4*lcoef[2]*(3.*(lcoef[0]**2 -lcoef[1]**2)**2 + 8.*lcoef[2]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r33
								r43 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r43
								r04 = (0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf0 - np.sqrt(15./2)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf1 + 0.5*np.sqrt(3.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r04
								r14 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]*lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[1]*(6.*lcoef[0]**2 *lcoef[2]**2 + lcoef[1]**2 -1.)*tdf1 + lcoef[1]*(3.*lcoef[0]**2 *lcoef[2]**2 + 2.*lcoef[1]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r14
								r24 = (np.sqrt(45.)*lcoef[0]*lcoef[1]**2 *lcoef[2]**2 *tdf0 - np.sqrt(5./2)*lcoef[0]*(6.*lcoef[1]**2 *lcoef[2]**2 + lcoef[0]**2 -1.)*tdf1 + lcoef[0]*(3.*lcoef[1]**2 *lcoef[2]**2 +2.*lcoef[0]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r24
								r34 = (0.5*np.sqrt(45.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf0 - np.sqrt(45./2)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 - lcoef[1]**2)*tdf1 + 1.5*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r34
								r44 = (np.sqrt(45.)*lcoef[0]**2 *lcoef[1]**2 *lcoef[2]*tdf0 - np.sqrt(5./2)*lcoef[2]*(6.*lcoef[0]**2 *lcoef[1]**2 +lcoef[2]**2 -1.)*tdf1 + lcoef[2]*(3.*lcoef[0]**2 *lcoef[1]**2 +2.*lcoef[2]**2 -1.)*tdf2)*cmath.exp(1j*kR) + r44
								r05 = (np.sqrt(5./32)*lcoef[0]*(3.*lcoef[2]**2 -1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[0]*lcoef[2]**2 *(lcoef[0]**2 -3.*lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[0]*(lcoef[2]**2 +1.)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r05
								r15 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[0]**2 *(lcoef[0]**2 - 3.*lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[0]**2 +2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r15
								r25 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(lcoef[0]**2 -3.*lcoef[1]**2 +4.)*tdf2)*cmath.exp(1j*kR) + r25
								r35 = (np.sqrt(15./32)*lcoef[0]*(lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*((lcoef[0]**2 - lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) - lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./32)*lcoef[0]*((lcoef[0]**2 -lcoef[1]**2)*(lcoef[0]**2 -3.*lcoef[1]**2) +4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r35
								r45 = (np.sqrt(15./8)*lcoef[0]**2 *lcoef[1]*(lcoef[0]**2 -3.*lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*(2.*lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -lcoef[2]**2 +1.)*tdf1 + np.sqrt(3./8)*lcoef[1]*(lcoef[0]**2 *(lcoef[0]**2 -3.*lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r45
								r06 = (np.sqrt(5./32)*lcoef[1]*(3.*lcoef[2]**2 -1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(45.)*lcoef[1]*lcoef[2]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf1 + 3./np.sqrt(32.)*lcoef[1]*(lcoef[2]**2 +1.)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r06
								r16 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 0.5*np.sqrt(15.)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*lcoef[1]*lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2 -4.)*tdf2)*cmath.exp(1j*kR) + r16
								r26 = (np.sqrt(15./8)*lcoef[1]**2 *lcoef[2]*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[2]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -lcoef[0]**2 +lcoef[1]**2)*tdf1 + np.sqrt(3./8)*lcoef[2]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[0]**2 + 2.*lcoef[1]**2)*tdf2)*cmath.exp(1j*kR) + r26
								r36 = (np.sqrt(15./32)*lcoef[1]*(lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./32)*lcoef[1]*((lcoef[0]**2 -lcoef[1]**2)*(3.*lcoef[0]**2 -lcoef[1]**2) + 4.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r36
								r46 = (np.sqrt(15./8)*lcoef[0]*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2)*tdf0 - 1./4*np.sqrt(15.)*lcoef[0]*(2.*lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) +lcoef[2]**2 -1.)*tdf1 + np.sqrt(3./8)*lcoef[0]*(lcoef[1]**2 *(3.*lcoef[0]**2 -lcoef[1]**2) -2.*lcoef[2]**2)*tdf2)*cmath.exp(1j*kR) + r46
						for ms in [-0.5, 0.5]:
							row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, 0, ms)
							row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							row4 = MatrixEntry(siteslist.Atomslist, Site1, l1, 2, ms)
							row5 = MatrixEntry(siteslist.Atomslist, Site1, l1,-2, ms)
							row6 = MatrixEntry(siteslist.Atomslist, Site1, l1, 3, ms)
							row7 = MatrixEntry(siteslist.Atomslist, Site1, l1,-3, ms)
							col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, 0, ms)
							col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							col4 = MatrixEntry(siteslist.Atomslist, Site2, l2, 2, ms)
							col5 = MatrixEntry(siteslist.Atomslist, Site2, l2,-2, ms)
							########### d0 COEFFICIENTS #############################
							self.H0[row1,col1,ik,jk,kk] = r00*(-1)**(l1+l2)
							self.H0[col1,row1,ik,jk,kk] = np.conjugate(r00)*(-1)**(l1+l2)
							self.H0[row2,col1,ik,jk,kk] =-1./np.sqrt(2.)*(r01+1j*r02)*(-1)**(l1+l2)
							self.H0[col1,row2,ik,jk,kk] =-1./np.sqrt(2.)*np.conjugate(r01+1j*r02)*(-1)**(l1+l2)
							self.H0[row3,col1,ik,jk,kk] = 1./np.sqrt(2.)*(r01-1j*r02)*(-1)**(l1+l2)
							self.H0[col1,row3,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r01-1j*r02)*(-1)**(l1+l2)
							self.H0[row4,col1,ik,jk,kk] = 1./np.sqrt(2.)*(r03+1j*r04)*(-1)**(l1+l2)
							self.H0[col1,row4,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r03+1j*r04)*(-1)**(l1+l2)
							self.H0[row5,col1,ik,jk,kk] = 1./np.sqrt(2.)*(r03-1j*r04)*(-1)**(l1+l2)
							self.H0[col1,row5,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r03-1j*r04)*(-1)**(l1+l2)
							self.H0[row6,col1,ik,jk,kk] =-1./np.sqrt(2.)*(r05+1j*r06)*(-1)**(l1+l2)
							self.H0[col1,row6,ik,jk,kk] =-1./np.sqrt(2.)*np.conjugate(r05+1j*r06)*(-1)**(l1+l2)
							self.H0[row7,col1,ik,jk,kk] = 1./np.sqrt(2.)*(r05-1j*r06)*(-1)**(l1+l2)
							self.H0[col1,row7,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r05-1j*r06)*(-1)**(l1+l2)
							########### d1 COEFFICIENTS ##############################
							self.H0[row1,col2,ik,jk,kk] =-1./np.sqrt(2.)*(r10-1j*r20)*(-1)**(l1+l2)
							self.H0[col2,row1,ik,jk,kk] =-1./np.sqrt(2.)*np.conjugate(r10-1j*r20)*(-1)**(l1+l2)
							self.H0[row2,col2,ik,jk,kk] = 0.5*(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
							self.H0[col2,row2,ik,jk,kk] = 0.5*np.conjugate(r11+r22+1j*(r12-r21))*(-1)**(l1+l2)
							self.H0[row3,col2,ik,jk,kk] =-0.5*(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
							self.H0[col2,row3,ik,jk,kk] =-0.5*np.conjugate(r11-r22-1j*(r12+r21))*(-1)**(l1+l2)
							self.H0[row4,col2,ik,jk,kk] =-0.5*(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
							self.H0[col2,row4,ik,jk,kk] =-0.5*np.conjugate(r13-1j*r23+1j*r14+r24)*(-1)**(l1+l2)
							self.H0[row5,col2,ik,jk,kk] =-0.5*(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
							self.H0[col2,row5,ik,jk,kk] =-0.5*np.conjugate(r13-1j*r23-1j*r14-r24)*(-1)**(l1+l2)
							self.H0[row6,col2,ik,jk,kk] = 0.5*(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
							self.H0[col2,row6,ik,jk,kk] = 0.5*np.conjugate(r15-1j*r25+1j*r16+r26)*(-1)**(l1+l2)
							self.H0[row7,col2,ik,jk,kk] =-0.5*(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
							self.H0[col2,row7,ik,jk,kk] =-0.5*np.conjugate(r15-1j*r25-1j*r16-r26)*(-1)**(l1+l2)
							########### d-1 COEFFICIENTS ##############################
							self.H0[row1,col3,ik,jk,kk] = 1./np.sqrt(2.)*(r10+1j*r20)*(-1)**(l1+l2)
							self.H0[col3,row1,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r10+1j*r20)*(-1)**(l1+l2)
							self.H0[row2,col3,ik,jk,kk] =-0.5*(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
							self.H0[col3,row2,ik,jk,kk] =-0.5*np.conjugate(r11-r22+1j*(r12+r21))*(-1)**(l1+l2)
							self.H0[row3,col3,ik,jk,kk] = 0.5*(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
							self.H0[col3,row3,ik,jk,kk] = 0.5*np.conjugate(r11+r22+1j*(-r12+r21))*(-1)**(l1+l2)
							self.H0[row4,col3,ik,jk,kk] = 0.5*(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
							self.H0[col3,row4,ik,jk,kk] = 0.5*np.conjugate(r13+1j*r23+1j*r14-r24)*(-1)**(l1+l2)
							self.H0[row5,col3,ik,jk,kk] = 0.5*(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
							self.H0[col3,row5,ik,jk,kk] = 0.5*np.conjugate(r13+1j*r23-1j*r14+r24)*(-1)**(l1+l2)
							self.H0[row6,col3,ik,jk,kk] =-0.5*(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
							self.H0[col3,row6,ik,jk,kk] =-0.5*np.conjugate(r15+1j*r25+1j*r16-r26)*(-1)**(l1+l2)
							self.H0[row7,col3,ik,jk,kk] = 0.5*(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
							self.H0[col3,row7,ik,jk,kk] = 0.5*np.conjugate(r15+1j*r25-1j*r16+r26)*(-1)**(l1+l2)
							########### d2 COEFFICIENTS ###############################
							self.H0[row1,col4,ik,jk,kk] = 1./np.sqrt(2.)*(r30-1j*r40)*(-1)**(l1+l2)
							self.H0[col4,row1,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r30-1j*r40)*(-1)**(l1+l2)
							self.H0[row2,col4,ik,jk,kk] =-0.5*(r31+r42+1j*(r32-r41))*(-1)**(l1+l2)
							self.H0[col4,row2,ik,jk,kk] =-0.5*np.conjugate(r31+r42+1j*(r32-r41))*(-1)**(l1+l2)
							self.H0[row3,col4,ik,jk,kk] = 0.5*(r31-r32-1j*(r32+r41))*(-1)**(l1+l2)
							self.H0[col4,row3,ik,jk,kk] = 0.5*np.conjugate(r31-r32-1j*(r32+r41))*(-1)**(l1+l2)
							self.H0[row4,col4,ik,jk,kk] = 0.5*(r33-1j*r43+1j*r34+r44)*(-1)**(l1+l2)
							self.H0[col4,row4,ik,jk,kk] = 0.5*np.conjugate(r33-1j*r43+1j*r34+r44)*(-1)**(l1+l2)
							self.H0[row5,col4,ik,jk,kk] = 0.5*(r33-1j*r43-1j*r34-r44)*(-1)**(l1+l2)
							self.H0[col4,row5,ik,jk,kk] = 0.5*np.conjugate(r33-1j*r43-1j*r34-r44)*(-1)**(l1+l2)
							self.H0[row6,col4,ik,jk,kk] =-0.5*(r35-1j*r45+1j*r36+r46)*(-1)**(l1+l2)
							self.H0[col4,row6,ik,jk,kk] =-0.5*np.conjugate(r35-1j*r45+1j*r36+r46)*(-1)**(l1+l2)
							self.H0[row7,col4,ik,jk,kk] = 0.5*(r35-1j*r45-1j*r36-r46)*(-1)**(l1+l2)
							self.H0[col4,row7,ik,jk,kk] = 0.5*np.conjugate(r35-1j*r45-1j*r36-r46)*(-1)**(l1+l2)
							########### d-2 COEFFICIENTS ##############################
							self.H0[row1,col5,ik,jk,kk] = 1./np.sqrt(2.)*(r30+1j*r40)*(-1)**(l1+l2)
							self.H0[col5,row1,ik,jk,kk] = 1./np.sqrt(2.)*np.conjugate(r30+1j*r40)*(-1)**(l1+l2)
							self.H0[row2,col5,ik,jk,kk] =-0.5*(r31-r42+1j*(r32+r41))*(-1)**(l1+l2)
							self.H0[col5,row2,ik,jk,kk] =-0.5*np.conjugate(r31-r42+1j*(r32+r41))*(-1)**(l1+l2)
							self.H0[row3,col5,ik,jk,kk] = 0.5*(r31+r42+1j*(-r32+r41))*(-1)**(l1+l2)
							self.H0[col5,row3,ik,jk,kk] = 0.5*np.conjugate(r31+r42+1j*(-r32+r41))*(-1)**(l1+l2)
							self.H0[row4,col5,ik,jk,kk] = 0.5*(r33+1j*r43+1j*r34-r44)*(-1)**(l1+l2)
							self.H0[col5,row4,ik,jk,kk] = 0.5*np.conjugate(r33+1j*r43+1j*r34-r44)*(-1)**(l1+l2)
							self.H0[row5,col5,ik,jk,kk] = 0.5*(r33+1j*r43-1j*r34+r44)*(-1)**(l1+l2)
							self.H0[col5,row5,ik,jk,kk] = 0.5*np.conjugate(r33+1j*r43-1j*r34+r44)*(-1)**(l1+l2)
							self.H0[row6,col5,ik,jk,kk] =-0.5*(r35+1j*r45+1j*r36-r46)*(-1)**(l1+l2)
							self.H0[col5,row6,ik,jk,kk] =-0.5*np.conjugate(r35+1j*r45+1j*r36-r46)*(-1)**(l1+l2)
							self.H0[row7,col5,ik,jk,kk] = 0.5*(r35+1j*r45-1j*r36+r46)*(-1)**(l1+l2)
							self.H0[col5,row7,ik,jk,kk] = 0.5*np.conjugate(r35+1j*r45-1j*r36+r46)*(-1)**(l1+l2)
						# iterate iik
						iik = iik + 1
	# set slater-koster matrix elements
	def set_tij(self, siteslist, kg, Unitcell, MatrixEntry):
		# iterate over atom's index
		for i1 in range(siteslist.Nsites):
			site1 = i1 + 1
			for i2 in range(i1, siteslist.Nsites):
				site2 = i2 + 1
				elements_pair = (siteslist.Atomslist[i1].element,siteslist.Atomslist[i2].element)
				if elements_pair in self.hopping_params.keys():
					# set ss SK integrals
					if 'ss' in self.hopping_params[elements_pair]:
						tss = self.hopping_params[elements_pair]['ss']
						self.set_ss_hopping_mtxel(site1, site2, tss, siteslist, kg, Unitcell, MatrixEntry)
					# sp SK integrals
					if 'sp' in self.hopping_params[elements_pair]:
						tsp = self.hopping_params[elements_pair]['sp']
						self.set_sp_hopping_mtxel(site1, site2, tsp, siteslist, kg, Unitcell, MatrixEntry)
					# ps SK integrals
					if 'ps' in self.hopping_params[elements_pair]:
						tsp = self.hopping_params[elements_pair]['ps']
						self.set_ps_hopping_mtxel(site1, site2, tsp, siteslist, kg, Unitcell, MatrixEntry)
					# pp SK integrals
					if 'pp' in self.hopping_params[elements_pair]:
						tpp = self.hopping_params[elements_pair]['pp']
						self.set_pp_hopping_mtxel(site1, site2, tpp, siteslist, kg, Unitcell, MatrixEntry)
					# set sd SK integrals
					if 'sd' in self.hopping_params[elements_pair]:
						tsd = self.hopping_params[elements_pair]['sd']
						self.set_sd_hopping_mtxel(site1, site2, tsd, siteslist, kg, Unitcell, MatrixEntry)
					# set ds SK integrals
					if 'ds' in self.hopping_params[elements_pair]:
						tsd = self.hopping_params[elements_pair]['ds']
						self.set_ds_hopping_mtxel(site1, site2, tsd, siteslist, kg, Unitcell, MatrixEntry)
					# set pd SK integrals
					if 'pd' in self.hopping_params[elements_pair]:
						tpd = self.hopping_params[elements_pair]['pd']
						self.set_pd_hopping_mtxel(site1, site2, tpd, siteslist, kg, Unitcell, MatrixEntry)
					# set dp SK integrals
					if 'dp' in self.hopping_params[elements_pair]:
						tpd = self.hopping_params[elements_pair]['dp']
						self.set_dp_hopping_mtxel(site1, site2, tpd, siteslist, kg, Unitcell, MatrixEntry)
					# set dd SK integrals
					if 'dd' in self.hopping_params[elements_pair]:
						tdd = self.hopping_params[elements_pair]['dd']
						self.set_dd_hopping_mtxel(site1, site2, tdd, siteslist, kg, Unitcell, MatrixEntry)
					# set sf SK integrals
					if 'sf' in self.hopping_params[elements_pair]:
						tsf = self.hopping_params[elements_pair]['sf']
						self.set_sf_hopping_mtxel(site1, site2, tsf, siteslist, kg, Unitcell, MatrixEntry)
					# set fs SK integrals
					if 'fs' in self.hopping_params[elements_pair]:
						tsf = self.hopping_params[elements_pair]['fs']
						self.set_fs_hopping_mtxel(site1, site2, tsf, siteslist, kg, Unitcell, MatrixEntry)
					# set pf SK integrals
					if 'pf' in self.hopping_params[elements_pair]:
						tpf = self.hopping_params[elements_pair]['pf']
						self.set_pf_hopping_mtxel(site1, site2, tpf, siteslist, kg, Unitcell, MatrixEntry)
					# set fp SK integrals
					if 'fp' in self.hopping_params[elements_pair]:
						tpf = self.hopping_params[elements_pair]['fp']
						self.set_fp_hopping_mtxel(site1, site2, tpf, siteslist, kg, Unitcell, MatrixEntry)
					# set df SK integrals
					if 'df' in self.hopping_params[elements_pair]:
						tdf = self.hopping_params[elements_pair]['df']
						self.set_df_hopping_mtxel(site1, site2, tdf, siteslist, kg, Unitcell, MatrixEntry)
					# set fd SK integrals
					if 'fd' in self.hopping_params[elements_pair]:
						tdf = self.hopping_params[elements_pair]['fd']
						self.set_fd_hopping_mtxel(site1, site2, tdf, siteslist, kg, Unitcell, MatrixEntry)
