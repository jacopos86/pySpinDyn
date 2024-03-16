#
#  This module implements
#  the K point class
#
import numpy as np
from pyspinorbitevol.utility_functions import norm_realv
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.logging_module import log
#
class k_pts_class:
	
	def __init__(self):
		# k points along each direction
		self.nkpts = None
		self.kgrid = None

	def set_kgrid(self, unit_cell):
		self.nkpts = np.array(p.nkpts)
		self.kgrid = []
		if p.D == 0:
			pass
		elif p.D == 1:
			kpt = [0.,0.,0.]
			for i in range(3):
				nk = self.nkpts[i]
				if nk == 0:
					pass
				else:
					bv = unit_cell.rec_vecs[i]
					L = norm_realv(bv)
					for ik in range(nk):
						kpt = [0.,0.,0.]
						kpt[i] = ik*L/(nk-1) - 0.5*L
						self.kgrid.append(kpt)
		elif p.D == 2:
			[nk1, nk2, nk3] = self.nkpts
			kpt = [0.,0.,0.]
			if nk1 == 0:
				bv2 = unit_cell.rec_vecs[1]
				bv3 = unit_cell.rec_vecs[2]
				L2 = norm_realv(bv2)
				L3 = norm_realv(bv3)
				#
				for ik in range(nk2):
					for jk in range(nk3):
						kpt = [0.,0.,0.]
						kpt[0] = 0.
						kpt[1] = ik*L2/(nk2-1) - 0.5*L2
						kpt[2] = jk*L3/(nk3-1) - 0.5*L3
						self.kgrid.append(kpt)
			elif nk2 == 0:
				bv1 = unit_cell.rec_vecs[0]
				bv3 = unit_cell.rec_vecs[2]
				L1 = norm_realv(bv1)
				L3 = norm_realv(bv3)
				#
				for ik in range(nk1):
					for jk in range(nk3):
						kpt = [0.,0.,0.]
						kpt[0] = ik*L1/(nk1-1) - 0.5*L1
						kpt[1] = 0.
						kpt[2] = jk*L3/(nk3-1) - 0.5*L3
						self.kgrid.append(kpt)
			elif nk3 == 0:
				bv1 = unit_cell.rec_vecs[0]
				bv2 = unit_cell.rec_vecs[1]
				L1 = norm_realv(bv1)
				L2 = norm_realv(bv2)
				#
				for ik in range(nk1):
					for jk in range(nk2):
						kpt = [0.,0.,0.]
						kpt[0] = ik*L1/(nk1-1) - 0.5*L1
						kpt[1] = jk*L2/(nk2-1) - 0.5*L2
						kpt[2] = 0.
						self.kgrid.append(kpt)
			else:
				log.error("wrong k pt. grid...")
				raise Exception("wrong k pt. grid...")
		elif p.D == 3:
			[nk1, nk2, nk3] = self.nkpts
			kpt = [0.,0.,0.]
			bv1 = unit_cell.rec_vecs[0]
			bv2 = unit_cell.rec_vecs[1]
			bv3 = unit_cell.rec_vecs[2]
			L1 = norm_realv(bv1)
			L2 = norm_realv(bv2)
			L3 = norm_realv(bv3)
			#
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						kpt = [0.,0.,0.]
						kpt[0] = ik*L1/(nk1-1) - 0.5*L1
						kpt[1] = jk*L2/(nk2-1) - 0.5*L2
						kpt[2] = kk*L3/(nk3-1) - 0.5*L3
						self.kgrid.append(kpt)
		else:
			log.error("wrong periodic dimension...")
			raise Exception ("wrong periodic dimension...")
	#
	def set_kpts_weights(self):
		# assume no symmetry operations
		if p.D == 1:
			nk = self.nkpts[np.where(self.nkpts > 0)[0][0]]
			self.wk = np.zeros(nk)
			self.wk[:] = 1./nk
		elif p.D == 2:
			if self.nkpts[0] == 0:
				nk1 = self.nkpts[1]
				nk2 = self.nkpts[2]
			elif self.nkpts[1] == 0:
				nk1 = self.nkpts[0]
				nk2 = self.nkpts[2]
			elif self.nkpts[2] == 0:
				nk1 = self.nkpts[0]
				nk2 = self.nkpts[1]
			self.wk = np.zeros((nk1, nk2))
			self.wk[:,:] = 1./(nk1*nk2)
		elif p.D == 3:
			[nk1,nk2,nk3] = self.nkpts
			self.wk = np.zeros((nk1, nk2, nk3))
			self.wk[:,:,:] = 1./(nk1*nk2*nk3)
# k pts instantiation
kg = k_pts_class()