#
#   This module defines the gradient of the crystal field
#   Hamiltonian of the system
#   H0(t_ij^l) = -\sum_ij,l1l2 t_ij^l1l2 c^+_ij^l1 c_ij^l2
#
import numpy as np
from pyspinorbitevol.utility_functions import set_lambda_coef
#
class CrystalFieldHamiltGradient:
	def __init__(self, hopping_params, siteslist, kg, Unitcell, MatrixEntry):
		# set the hopping terms
		self.hopping_params = hopping_params
		# set gradH0
		self.set_gradH0(siteslist, kg)
		# compute matrix elements
		self.set_grad_tij(siteslist, kg, Unitcell, MatrixEntry)
	# set crystal field gradient
	def set_gradH0(self, siteslist, kg):
		# set up matrix
		nst = siteslist.Nst
		if kg.D == 0:
			self.gradH0 = np.zeros((nst, nst, 3), dtype=np.complex128)
		elif kg.D == 1:
			[nk1, nk2, nk3] = kg.nkpts
			if nk1 != 0:
				self.gradH0 = np.zeros((nst, nst, nk1, 3), dtype=np.complex128)
			elif nk2 != 0:
				self.gradH0 = np.zeros((nst, nst, nk2, 3), dtype=np.complex128)
			elif nk3 != 0:
				self.gradH0 = np.zeros((nst, nst, nk3, 3), dtype=np.complex128)
			else:
				print("wrong n. k-pts for D=1")
				sys.exit(1)
		elif kg.D == 2:
			[nk1, nk2, nk3] = kg.nkpts
			if nk1 == 0:
				self.gradH0 = np.zeros((nst, nst, nk2, nk3, 3), dtype=np.complex128)
			elif nk2 == 0:
				self.gradH0 = np.zeros((nst, nst, nk1, nk3, 3), dtype=np.complex128)
			elif nk3 == 0:
				self.gradH0 = np.zeros((nst, nst, nk1, nk2, 3), dtype=np.complex128)
		elif kg.D == 3:
			[nk1, nk2, nk3] = kg.nkpts
			np.zeros((nst, nst, nk1, nk2, nk3, 3), dtype=np.complex128)
	# (s,s) SK integrals
	def set_ss_hopping_mtxel(self, Site1, Site2, tss, siteslist, kg, Unitcell, MatrixEntry):
		[tss0, gtss] = tss
		# (s,s) orbital pairs
		l1 = 0
		l2 = 0
		m = 0
		# check dimension
		if kg.D == 0:
			## nn data site1
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r = np.zeros(3, dtype=np.complex128)
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					lc = np.array(lcoef)
					r[:] = gtss * lc[:]
			for ms in [-0.5, 0.5]:
				row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
				col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
				self.gradH0[row,col,:] = r[:]
				self.gradH0[col,row,:] = np.conjugate(r[:])
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r = np.zeros(3, dtype=np.complex128)
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						lc = np.array(lcoef)
						r[:] = gtss * lc[:] * cmath.exp(1j*kR) + r[:]
				for ms in [-0.5, 0.5]:
					row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
					col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
					self.gradH0[row,col,ik,:] = r[:]
					self.gradH0[col,row,ik,:] = np.conjugate(r[:])
					if row == col:
						self.gradH0[row,col,ik,:] = self.gradH0[row,col,ik,:].real
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
					r = np.zeros(3, dtype=np.complex128)
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							lc = np.array(lcoef)
							r[:] = gtss * lc[:] * cmath.exp(1j*kR) + r[:]
					for ms in [-0.5, 0.5]:
						row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
						col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
						self.gradH0[row,col,ik,jk,:] = r[:]
						self.gradH0[col,row,ik,jk,:] = np.conjugate(r[:])
						if row == col:
							self.gradH0[row,col,ik,jk,:] = self.gradH0[row,col,ik,jk,:].real
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
						r = np.zeros(3, dtype=np.complex128)
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								lc = np.array(lcoef)
								r[:] = gtss * lc[:] * cmath.exp(1j*kR) + r[:]
						for ms in [-0.5, 0.5]:
							row = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
							col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
							self.gradH0[row,col,ik,jk,kk,:] = r[:]
							self.gradH0[col,row,ik,jk,kk,:] = np.conjugate(r[:])
							if row == col:
								self.gradH0[row,col,ik,jk,kk,:] = self.gradH0[row,col,ik,jk,kk,:].real
						# iterate iik
						iik = iik + 1
	# (s,p) SK integrals
	def set_sp_hopping_mtxel(self, Site1, Site2, tsp, siteslist, kg, Unitcell, MatrixEntry):
		[tsp0, gtsp] = tsp
		# (s,p) orbital pairs
		l1 = 0
		l2 = 1
		m = 0
		# check dimension
		if kg.D == 0:
			## nn data site1
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r1 = np.zeros(3, dtype=np.complex128)
			r2 = np.zeros(3, dtype=np.complex128)
			r3 = np.zeros(3, dtype=np.complex128)
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					lc = np.array(lcoef)
					R = siteslist.Atomslist[Site1-1].R0 - data['site'].coords
					d = norm_realv(R)
					glc= set_grad_lambda_coef(lc, d)
					r1[:] = glc[:,2]*tsp0 + lc[2]*gtsp*lc[:]
					r2[:] = -1./np.sqrt(2)*(glc[:,0]+1j*glc[:,1])*tsp0 - 1./np.sqrt(2)*(lc[0]+1j*lc[1])*gtsp*lc[:]
					r3[:] = 1./np.sqrt(2)*(glc[:,0]-1j*glc[:,1])*tsp0 + + 1./np.sqrt(2)*(lc[0]-1j*lc[1])*gtsp*lc[:]
			for ms in [-0.5, 0.5]:
				row  = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
				col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
				col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
				col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
				self.gradH0[row,col1,:] = r1[:]
				self.gradH0[col1,row,:] = np.conjugate(r1[:])
				self.gradH0[row,col2,:] = r2[:]
				self.gradH0[col2,row,:] = np.conjugate(r2[:])
				self.gradH0[row,col3,:] = r3[:]
				self.gradH0[col3,row,:] = np.conjugate(r3[:])
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r1 = np.zeros(3, dtype=np.complex128)
				r2 = np.zeros(3, dtype=np.complex128)
				r3 = np.zeros(3, dtype=np.complex128)
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						lc = np.array(lcoef)
						R = siteslist.Atomslist[Site1-1].R0 - data['site'].coords
						d = norm_realv(R)
						glc= set_grad_lambda_coef(lc, d)
						r1[:] = (glc[:,2]*tsp0 + lc[2]*gtsp*lc[:])*cmath.exp(1j*kR) + r1[:]
						r2[:] = (-1./np.sqrt(2)*(glc[:,0]+1j*glc[:,1])*tsp0 - 1./np.sqrt(2)*(lc[0]+1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r2[:]
						r3[:] = (1./np.sqrt(2)*(glc[:,0]-1j*glc[:,1])*tsp0 + + 1./np.sqrt(2)*(lc[0]-1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r3[:]
				for ms in [-0.5, 0.5]:
					row  = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
					col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
					col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
					col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
					self.gradH0[row,col1,ik,:] = r1[:]
					self.gradH0[col1,row,ik,:] = np.conjugate(r1[:])
					self.gradH0[row,col2,ik,:] = r2[:]
					self.gradH0[col2,row,ik,:] = np.conjugate(r2[:])
					self.gradH0[row,col3,ik,:] = r3[:]
					self.gradH0[col3,row,ik,:] = np.conjugate(r3[:])
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
					r1 = np.zeros(3, dtype=np.complex128)
					r2 = np.zeros(3, dtype=np.complex128)
					r3 = np.zeros(3, dtype=np.complex128)
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							lc = np.array(lcoef)
							R = siteslist.Atomslist[Site1-1].R0 - data['site'].coords
							d = norm_realv(R)
							glc= set_grad_lambda_coef(lc, d)
							r1[:] = (glc[:,2]*tsp0 + lc[2]*gtsp*lc[:])*cmath.exp(1j*kR) + r1[:]
							r2[:] = (-1./np.sqrt(2)*(glc[:,0]+1j*glc[:,1])*tsp0 - 1./np.sqrt(2)*(lc[0]+1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r2[:]
							r3[:] = (1./np.sqrt(2)*(glc[:,0]-1j*glc[:,1])*tsp0 + + 1./np.sqrt(2)*(lc[0]-1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r3[:]
					for ms in [-0.5, 0.5]:
						row  = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
						col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
						col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
						col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
						self.gradH0[row,col1,ik,jk,:] = r1[:]
						self.gradH0[col1,row,ik,jk,:] = np.conjugate(r1[:])
						self.gradH0[row,col2,ik,jk,:] = r2[:]
						self.gradH0[col2,row,ik,jk,:] = np.conjugate(r2[:])
						self.gradH0[row,col3,ik,jk,:] = r3[:]
						self.gradH0[col3,row,ik,jk,:] = np.conjugate(r3[:])
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
						r1 = np.zeros(3, dtype=np.complex128)
						r2 = np.zeros(3, dtype=np.complex128)
						r3 = np.zeros(3, dtype=np.complex128)
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								lc = np.array(lcoef)
								R = siteslist.Atomslist[Site1-1].R0 - data['site'].coords
								d = norm_realv(R)
								glc= set_grad_lambda_coef(lc, d)
								r1[:] = (glc[:,2]*tsp0 + lc[2]*gtsp*lc[:])*cmath.exp(1j*kR) + r1[:]
								r2[:] = (-1./np.sqrt(2)*(glc[:,0]+1j*glc[:,1])*tsp0 - 1./np.sqrt(2)*(lc[0]+1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r2[:]
								r3[:] = (1./np.sqrt(2)*(glc[:,0]-1j*glc[:,1])*tsp0 + + 1./np.sqrt(2)*(lc[0]-1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r3[:]
						for ms in [-0.5, 0.5]:
							row  = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
							col1 = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
							col2 = MatrixEntry(siteslist.Atomslist, Site2, l2, 1, ms)
							col3 = MatrixEntry(siteslist.Atomslist, Site2, l2,-1, ms)
							self.gradH0[row,col1,ik,jk,kk,:] = r1[:]
							self.gradH0[col1,row,ik,jk,kk,:] = np.conjugate(r1[:])
							self.gradH0[row,col2,ik,jk,kk,:] = r2[:]
							self.gradH0[col2,row,ik,jk,kk,:] = np.conjugate(r2[:])
							self.gradH0[row,col3,ik,jk,kk,:] = r3[:]
							self.gradH0[col3,row,ik,jk,kk,:] = np.conjugate(r3[:])
						# iterate iik
						iik = iik + 1
	# (p,s) SK integrals
	def set_ps_hopping_mtxel(self, Site1, Site2, tsp, siteslist, kg, Unitcell, MatrixEntry):
		[tsp0, gtsp] = tsp
		# (p,s) orbital pairs
		l1 = 1
		l2 = 0
		m = 0
		# check dimension
		if kg.D == 0:
			## nn data site1
			nndata_s1 = Unitcell.NNlist[Site1-1]
			r1 = np.zeros(3, dtype=np.complex128)
			r2 = np.zeros(3, dtype=np.complex128)
			r3 = np.zeros(3, dtype=np.complex128)
			for data in nndata_s1:
				if data['site'].index == (Site2 - 1):
					lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
					lc = np.array(lcoef)
					R  = siteslist.Atomslist[Site1-1].R0 - data['site'].coords
					d  = norm_realv(R)
					glc= set_grad_lambda_coef(lc, d)
					r1[:] = glc[:,2]*tsp0 + lc[2]*gtsp*lc[:]
					r2[:] = -1./np.sqrt(2)*(glc[:,0]+1j*glc[:,1])*tsp0 - 1./np.sqrt(2)*(lc[0]+1j*lc[1])*gtsp*lc[:]
					r3[:] = 1./np.sqrt(2)*(glc[:,0]-1j*glc[:,1])*tsp0 + + 1./np.sqrt(2)*(lc[0]-1j*lc[1])*gtsp*lc[:]
			for ms in [-0.5, 0.5]:
				row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
				row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
				row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
				col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
				self.gradH0[row1,col,:] = (-1)**(l1+l2)*r1[:]
				self.gradH0[col,row1,:] = (-1)**(l1+l2)*np.conjugate(r1[:])
				self.gradH0[row2,col,:] = (-1)**(l1+l2)*r2[:]
				self.gradH0[col,row2,:] = (-1)**(l1+l2)*np.conjugate(r2[:])
				self.gradH0[row3,col,:] = (-1)**(l1+l2)*r3[:]
				self.gradH0[col,row3,:] = (-1)**(l1+l2)*np.conjugate(r3[:])
		elif kg.D == 1:
			# run over k pts
			nk = kg.nkpts[np.where(kg.nkpts > 0)[0][0]]
			e = Unitcell.rcv
			for ik in range(nk):
				kpt = kg.kgrid[ik]
				k = kpt[0]*e[0]+kpt[1]*e[1]+kpt[2]*e[2]
				# nn data site1
				nndata_s1 = Unitcell.NNlist[Site1-1]
				r1 = np.zeros(3, dtype=np.complex128)
				r2 = np.zeros(3, dtype=np.complex128)
				r3 = np.zeros(3, dtype=np.complex128)
				for data in nndata_s1:
					if data['site'].index == (Site2 - 1):
						Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
						kR = np.inner(k,Rn)
						lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
						lc = np.array(lcoef)
						R = siteslist.Atomslist[Site1-1].R0 - data['site'].coords
						d = norm_realv(R)
						glc= set_grad_lambda_coef(lc, d)
						r1[:] = (glc[:,2]*tsp0 + lc[2]*gtsp*lc[:])*cmath.exp(1j*kR) + r1[:]
						r2[:] = (-1./np.sqrt(2)*(glc[:,0]+1j*glc[:,1])*tsp0 - 1./np.sqrt(2)*(lc[0]+1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r2[:]
						r3[:] = (1./np.sqrt(2)*(glc[:,0]-1j*glc[:,1])*tsp0 + + 1./np.sqrt(2)*(lc[0]-1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r3[:]
				for ms in [-0.5, 0.5]:
					row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
					row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
					row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
					col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
					self.gradH0[row1,col,ik,:] = (-1)**(l1+l2)*r1[:]
					self.gradH0[col,row1,ik,:] = (-1)**(l1+l2)*np.conjugate(r1[:])
					self.gradH0[row2,col,ik,:] = (-1)**(l1+l2)*r2[:]
					self.gradH0[col,row2,ik,:] = (-1)**(l1+l2)*np.conjugate(r2[:])
					self.gradH0[row3,col,ik,:] = (-1)**(l1+l2)*r3[:]
					self.gradH0[col,row3,ik,:] = (-1)**(l1+l2)*np.conjugate(r3[:])
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
					r1 = np.zeros(3, dtype=np.complex128)
					r2 = np.zeros(3, dtype=np.complex128)
					r3 = np.zeros(3, dtype=np.complex128)
					for data in nndata_s1:
						if data['site'].index == (Site2 - 1):
							Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
							kR = np.inner(k,Rn)
							lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
							lc = np.array(lcoef)
							R = siteslist.Atomslist[Site1-1].R0 - data['site'].coords
							d = norm_realv(R)
							glc= set_grad_lambda_coef(lc, d)
							r1[:] = (glc[:,2]*tsp0 + lc[2]*gtsp*lc[:])*cmath.exp(1j*kR) + r1[:]
							r2[:] = (-1./np.sqrt(2)*(glc[:,0]+1j*glc[:,1])*tsp0 - 1./np.sqrt(2)*(lc[0]+1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r2[:]
							r3[:] = (1./np.sqrt(2)*(glc[:,0]-1j*glc[:,1])*tsp0 + + 1./np.sqrt(2)*(lc[0]-1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r3[:]
					for ms in [-0.5, 0.5]:
						row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
						row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
						row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
						col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
						self.gradH0[row1,col,ik,jk,:] = (-1)**(l1+l2)*r1[:]
						self.gradH0[col,row1,ik,jk,:] = (-1)**(l1+l2)*np.conjugate(r1[:])
						self.gradH0[row2,col,ik,jk,:] = (-1)**(l1+l2)*r2[:]
						self.gradH0[col,row2,ik,jk,:] = (-1)**(l1+l2)*np.conjugate(r2[:])
						self.gradH0[row3,col,ik,jk,:] = (-1)**(l1+l2)*r3[:]
						self.gradH0[col,row3,ik,jk,:] = (-1)**(l1+l2)*np.conjugate(r3[:])
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
						r1 = np.zeros(3, dtype=np.complex128)
						r2 = np.zeros(3, dtype=np.complex128)
						r3 = np.zeros(3, dtype=np.complex128)
						for data in nndata_s1:
							if data['site'].index == (Site2 - 1):
								Rn = data['site'].coords - siteslist.Atomslist[Site2-1].R0
								kR = np.inner(k,Rn)
								lcoef = set_lambda_coef(siteslist.Atomslist[Site1-1].R0, data['site'].coords)
								lc = np.array(lcoef)
								R = siteslist.Atomslist[Site1-1].R0 - data['site'].coords
								d = norm_realv(R)
								glc= set_grad_lambda_coef(lc, d)
								r1[:] = (glc[:,2]*tsp0 + lc[2]*gtsp*lc[:])*cmath.exp(1j*kR) + r1[:]
								r2[:] = (-1./np.sqrt(2)*(glc[:,0]+1j*glc[:,1])*tsp0 - 1./np.sqrt(2)*(lc[0]+1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r2[:]
								r3[:] = (1./np.sqrt(2)*(glc[:,0]-1j*glc[:,1])*tsp0 + + 1./np.sqrt(2)*(lc[0]-1j*lc[1])*gtsp*lc[:])*cmath.exp(1j*kR) + r3[:]
						for ms in [-0.5, 0.5]:
							row1 = MatrixEntry(siteslist.Atomslist, Site1, l1, m, ms)
							row2 = MatrixEntry(siteslist.Atomslist, Site1, l1, 1, ms)
							row3 = MatrixEntry(siteslist.Atomslist, Site1, l1,-1, ms)
							col = MatrixEntry(siteslist.Atomslist, Site2, l2, m, ms)
							self.gradH0[row1,col,ik,jk,kk,:] = (-1)**(l1+l2)*r1[:]
							self.gradH0[col,row1,ik,jk,kk,:] = (-1)**(l1+l2)*np.conjugate(r1[:])
							self.gradH0[row2,col,ik,jk,kk,:] = (-1)**(l1+l2)*r2[:]
							self.gradH0[col,row2,ik,jk,kk,:] = (-1)**(l1+l2)*np.conjugate(r2[:])
							self.gradH0[row3,col,ik,jk,kk,:] = (-1)**(l1+l2)*r3[:]
							self.gradH0[col,row3,ik,jk,kk,:] = (-1)**(l1+l2)*np.conjugate(r3[:])
						# iterate iik
						iik = iik + 1
	# set grad_tij
	def set_grad_tij(self, siteslist, kg, Unitcell, MatrixEntry):
		# iterate atom's index
		for i1 in range(siteslist.Nsites):
			site1 = i1+1
			for i2 in range(i1, siteslist.Nsites):
				site2 = i2+1
				elements_pair = (siteslist.Atomslist[i1].element, siteslist.Atomslist[i2].element)
				if elements_pair in self.hopping_params.keys():
					# set ss SK integrals
					if 'ss' in self.hopping_params[elements_pair]:
						tss = self.hopping_params[elements_pair]['ss']
						self.set_ss_hopping_mtxel(site1, site2, tss, siteslist, kg, Unitcell, MatrixEntry)
					# set sp SK integrals
					if 'sp' in self.hopping_params[elements_pair]:
						tsp = self.hopping_params[elements_pair]['sp']
						self.set_sp_hopping_mtxel(site1, site2, tsp, siteslist, kg, Unitcell, MatrixEntry)
					# set ps SK integrals
					if 'sp' in self.hopping_params[elements_pair]:
						tsp = self.hopping_params[elements_pair]['ps']
						self.set_ps_hopping_mtxel(site1, site2, tsp, siteslist, kg, Unitcell, MatrixEntry)
