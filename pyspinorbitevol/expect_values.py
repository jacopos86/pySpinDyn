#
#      This module computes the 
#      expectation value of different
#      observables
#
import numpy as np
from pyspinorbitevol.phys_constants import hbar
#
def spin_expect_val(spin_operators, rho_e, siteslist, kg, MatrixEntry):
	# this function computes
	# <S> expect. value
	Sx_expect = np.zeros(siteslist.Nsites)
	Sy_expect = np.zeros(siteslist.Nsites)
	Sz_expect = np.zeros(siteslist.Nsites)
	# compute rho_e Si
	if kg.D == 0:
		rho_Sx = np.matmul(rho_e, spin_operators.S[:,:,0])
		rho_Sy = np.matmul(rho_e, spin_operators.S[:,:,1])
		rho_Sz = np.matmul(rho_e, spin_operators.S[:,:,2])
		# run over atomic sites
		for i in range(siteslist.Nsites):
			site = i+1
			for l in siteslist.Atomslist[i].OrbitalList:
				for ml in range(-l, l+1):
					for ms in [-0.5, 0.5]:
						row = MatrixEntry(siteslist.Atomslist, site, l, ml, ms)
						col = row
						# <Sx>
						Sx_expect[i] = Sx_expect[i] + rho_Sx[row,col].real
						# <Sy>
						Sy_expect[i] = Sy_expect[i] + rho_Sy[row,col].real
						# <Sz>
						Sz_expect[i] = Sz_expect[i] + rho_Sz[row,col].real
	elif kg.D == 1:
		nk = kg.nkpts[np.where(kg.nkpts>0)[0][0]]
		for ik in range(nk):
			rho_Sx = np.matmul(rho_e[:,:,ik], spin_operators.S[:,:,ik,0])
			rho_Sy = np.matmul(rho_e[:,:,ik], spin_operators.S[:,:,ik,1])
			rho_Sz = np.matmul(rho_e[:,:,ik], spin_operators.S[:,:,ik,2])
			# run over atom sites
			for i in range(siteslist.Nsites):
				site = i+1
				for l in siteslist.Atomslist[i].OrbitalList:
					for ml in range(-l, l+1):
						for ms in [-0.5, 0.5]:
							row = MatrixEntry(siteslist.Atomslist, site, l, ml, ms)
							col = row
							# <Sx>
							Sx_expect[i] = Sx_expect[i] + kg.wk[ik] * rho_Sx[row,col].real
							# <Sy>
							Sy_expect[i] = Sy_expect[i] + kg.wk[ik] * rho_Sy[row,col].real
							# <Sz>
							Sz_expect[i] = Sz_expect[i] + kg.wk[ik] * rho_Sz[row,col].real
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
		for ik in range(nk1):
			for jk in range(nk2):
				rho_Sx = np.matmul(rho_e[:,:,ik,jk], spin_operators.S[:,:,ik,jk,0])
				rho_Sy = np.matmul(rho_e[:,:,ik,jk], spin_operators.S[:,:,ik,jk,1])
				rho_Sz = np.matmul(rho_e[:,:,ik,jk], spin_operators.S[:,:,ik,jk,2])
				# run over atom sites
				for i in range(siteslist.Nsites):
					site = i+1
					for l in siteslist.Atomslist[i].OrbitalList:
						for ml in range(-l, l+1):
							for ms in [-0.5, 0.5]:
								row = MatrixEntry(siteslist.Atomslist, site, l, ml, ms)
								col = row
								# <Sx>
								Sx_expect[i] = Sx_expect[i] + kg.wk[ik,jk] * rho_Sx[row,col].real
								# <Sy>
								Sy_expect[i] = Sy_expect[i] + kg.wk[ik,jk] * rho_Sy[row,col].real
								# <Sz>
								Sz_expect[i] = Sz_expect[i] + kg.wk[ik,jk] * rho_Sz[row,col].real
	elif kg.D == 3:
		[nk1,nk2,nk3] = kg.nkpts
		for ik in range(nk1):
			for jk in range(nk2):
				for kk in range(nk3):
					rho_Sx = np.matmul(rho_e[:,:,ik,jk,kk], spin_operators.S[:,:,ik,jk,kk,0])
					rho_Sy = np.matmul(rho_e[:,:,ik,jk,kk], spin_operators.S[:,:,ik,jk,kk,1])
					rho_Sz = np.matmul(rho_e[:,:,ik,jk,kk], spin_operators.S[:,:,ik,jk,kk,2])
					# run over atom sites
					for i in range(siteslist.Nsites):
						site = i+1
						for l in siteslist.Atomslist[i].OrbitalList:
							for ml in range(-l, l+1):
								for ms in [-0.5, 0.5]:
									row = MatrixEntry(siteslist.Atomslist, site, l, ml, ms)
									col = row
									# <Sx>
									Sx_expect[i] = Sx_expect[i] + kg.wk[ik,jk,kk] * rho_Sx[row,col].real
									# <Sy>
									Sy_expect[i] = Sy_expect[i] + kg.wk[ik,jk,kk] * rho_Sy[row,col].real
									# <Sz>
									Sz_expect[i] = Sz_expect[i] + kg.wk[ik,jk,kk] * rho_Sz[row,col].real
	# return in (eV fs) units
	Sx_expect[:] = Sx_expect[:] * hbar
	Sy_expect[:] = Sy_expect[:] * hbar
	Sz_expect[:] = Sz_expect[:] * hbar
	return [Sx_expect, Sy_expect, Sz_expect]
#
def orbital_expect_val(orbital_operators, rho_e, siteslist, kg, MatrixEntry):
	# this routine computes <L0>
	# orbital momentum expect. value
	L0x_expect = np.zeros(siteslist.Nsites)
	L0y_expect = np.zeros(siteslist.Nsites)
	L0z_expect = np.zeros(siteslist.Nsites)
	# check dimensionality
	if kg.D == 0:
		# compute rho_e L0
		rho_Lx = np.matmul(rho_e, orbital_operators.L0[:,:,0])
		rho_Ly = np.matmul(rho_e, orbital_operators.L0[:,:,1])
		rho_Lz = np.matmul(rho_e, orbital_operators.L0[:,:,2])
		# run over atom sites
		for i in range(siteslist.Nsites):
			site = i+1
			for l in siteslist.Atomslist[i].OrbitalList:
				for ml in range(-l, l+1):
					for ms in [-0.5, 0.5]:
						row = MatrixEntry(siteslist.Atomslist, site, l, ml, ms)
						col = row
						# <L0x>
						L0x_expect[i] = L0x_expect[i] + rho_Lx[row,col].real
						# <L0x>
						L0y_expect[i] = L0y_expect[i] + rho_Ly[row,col].real
						# <L0x>
						L0z_expect[i] = L0z_expect[i] + rho_Lz[row,col].real
	elif kg.D == 1:
		nk = kg.nkpts[np.where(kg.nkpts>0)[0][0]]
		for ik in range(nk):
			# compute rho_e L0
			rho_Lx = np.matmul(rho_e[:,:,ik], orbital_operators.L0[:,:,ik,0])
			rho_Ly = np.matmul(rho_e[:,:,ik], orbital_operators.L0[:,:,ik,1])
			rho_Lz = np.matmul(rho_e[:,:,ik], orbital_operators.L0[:,:,ik,2])
			# run over atom sites
			for i in range(siteslist.Nsites):
				site = i+1
				for l in siteslist.Atomslist[i].OrbitalList:
					for ml in range(-l, l+1):
						for ms in [-0.5, 0.5]:
							row = MatrixEntry(siteslist.Atomslist, site, l, ml, ms)
							col = row
							# <L0x>
							L0x_expect[i] = L0x_expect[i] + kg.wk[ik] * rho_Lx[row,col].real
							# <L0y>
							L0y_expect[i] = L0y_expect[i] + kg.wk[ik] * rho_Ly[row,col].real
							# <L0z>
							L0z_expect[i] = L0z_expect[i] + kg.wk[ik] * rho_Lz[row,col].real
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
		for ik in range(nk1):
			for jk in range(nk2):
				# compute rho_e L0
				rho_Lx = np.matmul(rho_e[:,:,ik,jk], orbital_operators.L0[:,:,ik,jk,0])
				rho_Ly = np.matmul(rho_e[:,:,ik,jk], orbital_operators.L0[:,:,ik,jk,1])
				rho_Lz = np.matmul(rho_e[:,:,ik,jk], orbital_operators.L0[:,:,ik,jk,2])
				# run over atom sites
				for i in range(siteslist.Nsites):
					site = i+1
					for l in siteslist.Atomslist[i].OrbitalList:
						for ml in range(-l, l+1):
							for ms in [-0.5, 0.5]:
								row = MatrixEntry(siteslist.Atomslist, site, l, ml, ms)
								col = row
								# <L0x>
								L0x_expect[i] = L0x_expect[i] + kg.wk[ik,jk] * rho_Lx[row,col].real
								# <L0y>
								L0y_expect[i] = L0y_expect[i] + kg.wk[ik,jk] * rho_Ly[row,col].real
								# <L0z>
								L0z_expect[i] = L0z_expect[i] + kg.wk[ik,jk] * rho_Lz[row,col].real
	elif kg.D == 3:
		[nk1,nk2,nk3] = kg.nkpts
		for ik in range(nk1):
			for jk in range(nk2):
				for kk in range(nk3):
					# compute rho_e L0
					rho_Lx = np.matmul(rho_e[:,:,ik,jk,kk], orbital_operators.L0[:,:,ik,jk,kk,0])
					rho_Ly = np.matmul(rho_e[:,:,ik,jk,kk], orbital_operators.L0[:,:,ik,jk,kk,1])
					rho_Lz = np.matmul(rho_e[:,:,ik,jk,kk], orbital_operators.L0[:,:,ik,jk,kk,2])
					# run over atoms
					for i in range(siteslist.Nsites):
						site = i+1
						for l in siteslist.Atomslist[i].OrbitalList:
							for ml in range(-l, l+1):
								for ms in [-0.5, 0.5]:
									row = MatrixEntry(siteslist.Atomslist, site, l, ml, ms)
									col = row
									# <L0x>
									L0x_expect[i] = L0x_expect[i] + kg.wk[ik,jk,kk] * rho_Lx[row,col].real
									# <L0y>
									L0y_expect[i] = L0y_expect[i] + kg.wk[ik,jk,kk] * rho_Ly[row,col].real
									# <L0z>
									L0z_expect[i] = L0z_expect[i] + kg.wk[ik,jk,kk] * rho_Lz[row,col].real
	# return in (eV fs) units
	L0x_expect[:] = L0x_expect[:] * hbar
	L0y_expect[:] = L0y_expect[:] * hbar
	L0z_expect[:] = L0z_expect[:] * hbar
	return [L0x_expect, L0y_expect, L0z_expect]
#
def COM_expect_val(orbital_operators, rho_e, siteslist, kg, MatrixEntry):
	# this routine computes <L> expect val.
	Lx_expect = np.zeros(siteslist.Nsites)
	Ly_expect = np.zeros(siteslist.Nsites)
	Lz_expect = np.zeros(siteslist.Nsites)
	# check dimensions
	if kg.D == 0:
		# compute rho_e L
		rho_Lx = np.matmul(rho_e, orbital_operators.COM[:,:,0])
		rho_Ly = np.matmul(rho_e, orbital_operators.COM[:,:,1])
		rho_Lz = np.matmul(rho_e, orbital_operators.COM[:,:,2])
		# run over atoms
		for i in range(siteslist.Nsites):
			site = i+1
			for l in siteslist.Atomslist[i].OrbitalList:
				for ml in range(-l, l+1):
					for ms in [-0.5, 0.5]:
						row = MatrixEntry(siteslist.Atomslist, site, l, ml, ms)
						col = row
						# <L>
						Lx_expect[i] = Lx_expect[i] + rho_Lx[row,col].real
						Ly_expect[i] = Ly_expect[i] + rho_Ly[row,col].real
						Lz_expect[i] = Lz_expect[i] + rho_Lz[row,col].real
