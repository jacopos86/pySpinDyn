#
#   This module sets all the
#   methods for the ground state calculation
#
import numpy as np
from scipy import linalg
import sys
from pyspinorbitevol.utility_functions import compute_nelec, logistic_function, F, lorentzian
#
class GroundState:
	def __init__(self, CrystalField, siteslist, kg, Unitcell, MatrixEntry):
		# diagonalize Hamiltonian
		self.hamilt_diagonal(CrystalField.H0, siteslist, kg)
	# diagonalization method
	def hamilt_diagonal(self, H, siteslist, kg):
		# this method diagonalizes H
		# input1 : H -> system's Hamiltonian
		# input2 : siteslist -> atoms list
		# input3 : kg -> k grid
		#
		# space dimension
		if kg.D == 0:
			if not (H==H.T.conj()).all():
				print("Error: Hamiltonian non hermitean...")
				sys.exit(1)
			w, v = linalg.eigh(H)
			si = np.argsort(w)
			self.eig = [None]*siteslist.Nst
			self.eigv = [None]*siteslist.Nst
			#
			r = np.zeros(siteslist.Nst, dtype=np.complex128)
			for i in range(len(si)):
				j = si[i]
				r[:] = 0.
				r[:] = v[:,j]
				self.eig[i] = w[j]
				self.eigv[i] = r.copy()
			# check diagonalization
			for i in range(siteslist.Nst):
				if not np.allclose(H @ self.eigv[i], self.eig[i] * self.eigv[i]):
					print("Error: wrong eigenvector")
					sys.exit(1)
			self.eig = np.array(self.eig)
		elif kg.D == 1:
			# run over kpts
			nk = kg.nkpts[np.where(kg.nkpts>0)[0][0]]
			for ik in range(nk):
				Hk = np.zeros((siteslist.Nst,siteslist.Nst), dtype=np.complex128)
				Hk[:,:] = H[:,:,ik]
				if not (Hk==Hk.T.conj()).all():
					print("Error: Hamiltonian non hermitean...")
					sys.exit(1)
				# eigenvectors
				w, v = linalg.eigh(Hk)
				if ik == 0:
					self.eig = np.zeros((siteslist.Nst,nk))
					self.eigv= np.zeros((siteslist.Nst,siteslist.Nst,nk), dtype=np.complex128)
					si = np.argsort(w)
				#
				r = np.zeros(siteslist.Nst, dtype=np.complex128)
				for i in range(len(si)):
					j = si[i]
					r[:] = 0.
					r[:] = v[:,j]
					self.eig[i,ik] = w[j]
					self.eigv[i,:,ik] = r.copy()
				# check diagonalization
				for i in range(siteslist.Nst):
					if not np.allclose(Hk @ self.eigv[i,:,ik], self.eig[i,ik] * self.eigv[i,:,ik]):
						print("Error: wrong eigenvector")
						sys.exit(1)
		elif kg.D == 2:
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
					Hk = np.zeros((siteslist.Nst,siteslist.Nst), dtype=np.complex128)
					Hk[:,:] = H[:,:,ik,jk]
					if not (Hk==Hk.T.conj()).all():
						print("Error: Hamiltonian non hermitean...")
						sys.exit(1)
					# eigenvectors
					w, v = linalg.eigh(Hk)
					if iik == 0:
						self.eig = np.zeros((siteslist.Nst,nk1,nk2))
						self.eigv= np.zeros((siteslist.Nst,siteslist.Nst,nk1,nk2), dtype=np.complex128)
						si = np.argsort(w)
					#
					r = np.zeros(siteslist.Nst, dtype=np.complex128)
					for i in range(len(si)):
						j = si[i]
						r[:] = 0.
						r[:] = v[:,j]
						self.eig[i,ik,jk] = w[j]
						self.eigv[i,:,ik,jk] = r.copy()
					# check diagonalization
					for i in range(siteslist.Nst):
						if not np.allclose(Hk @ self.eigv[i,:,ik,jk], self.eig[i,ik,jk] * self.eigv[i,:,ik,jk]):
							print("Error: wrong eigenvector")
							sys.exit(1)
					iik = iik + 1
		elif kg.D == 3:
			iik = 0
			[nk1, nk2, nk3] = kg.nkpts
			# run over kpts
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						Hk = np.zeros((siteslist.Nst,siteslist.Nst), dtype=np.complex128)
						Hk[:,:] = H[:,:,ik,jk,kk]
						if not (Hk==Hk.T.conj()).all():
							print("Error: Hamiltonian non hermitean...")
							sys.exit(1)
						# eigenvectors
						w, v = linalg.eigh(Hk)
						if iik == 0:
							self.eig = np.zeros((siteslist.Nst,nk1,nk2,nk3))
							self.eigv= np.zeros((siteslist.Nst,siteslist.Nst,nk1,nk2,nk3), dtype=np.complex128)
							si = np.argsort(w)
						#
						r = np.zeros(siteslist.Nst, dtype=np.complex128)
						for i in range(len(si)):
							j = si[i]
							r[:] = 0.
							r[:] = v[:,j]
							self.eig[i,ik,jk,kk] = w[j]
							self.eigv[i,:,ik,jk,kk] = r.copy()
						# check diagonalization
						for i in range(siteslist.Nst):
							if not np.allclose(Hk @ self.eigv[i,:,ik,jk,kk], self.eig[i,ik,jk,kk] * self.eigv[i,:,ik,jk,kk]):
								print("Error: wrong eigenvector")
								sys.exit(1)
						iik = iik + 1
	# set states occupations
	def set_wfc_occupations(self, siteslist, kg, kT, niters=1000, tol=1.E-4):
		# kT (eV units)
		# compute wfc occupations
		self.occup = np.zeros(self.eig.shape)
		# first compute total number 
		# of electrons in the system
		nel = 0
		nel = compute_nelec(siteslist, kg)
		# sort eigenvalue list
		sorted_eig = np.sort(self.eig, axis=None)
		# initialize E1, E2
		E1 = sorted_eig[0]
		for i in range(1, len(sorted_eig)):
			E2 = sorted_eig[i]
			if F(E1,nel,self.eig,kT,kg) * F(E2,nel,self.eig,kT,kg) > 0.:
				E1 = E2
			else:
				break
		# compute fermi energy
		self.ef = 0.
		for i in range(niters):
			z = 0.5 * (E1 + E2)
			r = F(z,nel,self.eig,kT,kg)
			if abs(r) < tol:
				break
			else:
				if F(E1,nel,self.eig,kT,kg) * r > 0.:
					E1 = z
				else:
					E2 = z
		self.ef = z
		# compute occup.
		nb = siteslist.Nst
		if kg.D == 0:
			for ib in range(nb):
				e = self.eig[ib]
				x = (self.ef - e) / kT
				self.occup[ib] = logistic_function(x)
		elif kg.D == 1:
			nk = self.eig.shape[1]
			for ib in range(nb):
				for ik in range(nk):
					e = self.eig[ib,ik]
					x = (self.ef - e) / kT
					self.occup[ib,ik] = logistic_function(x)
		elif kg.D == 2:
			nk1 = self.eig.shape[1]
			nk2 = self.eig.shape[2]
			for ib in range(nb):
				for ik in range(nk1):
					for jk in range(nk2):
						e = self.eig[ib,ik,jk]
						x = (self.ef - e) / kT
						self.occup[ib,ik,jk] = logistic_function(x)
		elif kg.D == 3:
			[nk1, nk2, nk3] = kg.nkpts
			for ib in range(nb):
				for ik in range(nk1):
					for jk in range(nk2):
						for kk in range(nk3):
							e = self.eig[ib,ik,jk,kk]
							x = (self.ef - e) / kT
							self.occup[ib,ik,jk,kk] = logistic_function(x)
	# set density matrix
	def set_density_matrix(self, siteslist, kg, tol=1.E-4):
		# compute density matrix operator
		nst = siteslist.Nst
		if kg.D == 0:
			self.rho_e = np.zeros((nst,nst), dtype=np.complex128)
		elif kg.D == 1:
			nk = self.eig.shape[1]
			self.rho_e = np.zeros((nst,nst,nk), dtype=np.complex128)
		elif kg.D == 2:
			nk1 = self.eig.shape[1]
			nk2 = self.eig.shape[2]
			self.rho_e = np.zeros((nst,nst,nk1,nk2), dtype=np.complex128)
		elif kg.D == 3:
			[nk1, nk2, nk3] = kg.nkpts
			self.rho_e = np.zeros((nst,nst,nk1,nk2,nk3), dtype=np.complex128)
		# compute matrix elements
		# rho_e = sum_i f_i |psi_i><psi_i|
		nel = 0
		if kg.D == 0:
			for i in range(nst):
				v = self.eigv[i]
				self.rho_e = self.rho_e + self.occup[i] * np.outer(v, v.conjugate())
			nel = self.rho_e.trace() + nel
		elif kg.D == 1:
			for ik in range(nk):
				for i in range(nst):
					v = self.eigv[i,:,ik]
					self.rho_e[:,:,ik] = self.rho_e[:,:,ik] + self.occup[i,ik] * np.outer(v, v.conjugate())
				nel = kg.wk[ik] * self.rho_e[:,:,ik].trace() + nel
		elif kg.D == 2:
			for ik in range(nk1):
				for jk in range(nk2):
					for i in range(nst):
						v = self.eigv[i,:,ik,jk]
						self.rho_e[:,:,ik,jk] = self.rho_e[:,:,ik,jk] + self.occup[i,ik,jk] * np.outer(v, v.conjugate())
					nel = kg.wk[ik,jk] * self.rho_e[:,:,ik,jk].trace() + nel
		elif kg.D == 3:
			for ik in range(nk1):
				for jk in range(nk2):
					for kk in range(nk3):
						for i in range(nst):
							v = self.eigv[i,:,ik,jk,kk]
							self.rho_e[:,:,ik,jk,kk] = self.rho_e[:,:,ik,jk,kk] + self.occup[i,ik,jk,kk] * np.outer(v, v.conjugate())
						nel = kg.wk[ik,jk,kk] * self.rho_e[:,:,ik,jk,kk].trace() + nel
		if abs(nel.real - compute_nelec(siteslist, kg)) > tol:
			print("wrong nel in density matrix...")
			sys.exit(1)
	# compute electronic DOS
	def compute_elec_DOS(self, kg, siteslist, out_file, Erng, dE, gamma):
		# E range
		[Emn, Emx] = Erng
		E = np.arange(Emn, Emx+dE, dE)
		nE = len(E)
		# open file
		f = open(out_file, 'w')
		# DOS(E) = \sum_k \sum_n delta(E - e_nk)
		dos_e = np.zeros(nE)
		# check dimension periodicity
		nst = siteslist.Nst
		if kg.D == 0:
			for ib in range(nst):
				en = self.eig[ib]
				dos_e[:] = dos_e[:] + lorentzian(en-E[:], gamma)
		elif kg.D == 1:
			nk = self.eig.shape[1]
			for ib in range(nst):
				for ik in range(nk):
					enk = self.eig[ib,ik]
					dos_e[:] = dos_e[:] + kg.wk[ik] * lorentzian(enk-E[:], gamma)
		elif kg.D == 2:
			nk1 = self.eig.shape[1]
			nk2 = self.eig.shape[2]
			for ib in range(nst):
				for ik in range(nk1):
					for jk in range(nk2):
						enk = self.eig[ib,ik,jk]
						dos_e[:] = dos_e[:] + kg.wk[ik,jk] * lorentzian(enk-E[:], gamma)
		elif kg.D == 3:
			[nk1,nk2,nk3] = kg.nkpts
			for ib in range(nst):
				for ik in range(nk1):
					for jk in range(nk2):
						for kk in range(nk3):
							enk = self.eig[ib,ik,jk,kk]
							dos_e[:] = dos_e[:] + kg.wk[ik,jk,kk] * lorentzian(enk-E[:], gamma)
		return dos_e
	# plot band structure method
	def plot_band_structure(self, kg, out_file):
		f = open(out_file, 'w')
		# check periodic dimension
		if kg.D == 1:
			nst = self.eig.shape[0]
			nk = self.eig.shape[1]
			# run over bands
			for ib in range(nst):
				for ik in range(nk):
					f.write( "%.7f\n" % self.eig[ib,ik] )
				f.write( "\n" )
		f.close()
