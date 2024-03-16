#
#   This routine adds the onsite
#   atomic energies to the
#   crystal field Hamiltonian
#
import numpy as np
#
def set_onsite_energies(CrystalFieldH, siteslist, kg, MatrixEntry, atomic_energies):
	# input : H0 -> crystal field Hamiltonian
	# inputs: siteslist, kg, MatrixEntry
	# input : atomic_energies dictionary
	# output: H0 + onsite_energies
	for i in range(siteslist.Nsites):
		site = i+1
		element = siteslist.Atomslist[i].element
		for l in siteslist.Atomslist[i].OrbitalList:
			for ml in range(-l, l+1):
				for ms in [-0.5, 0.5]:
					row = MatrixEntry(siteslist.Atomslist, site, l, ml, ms)
					col = row
					e = atomic_energies[element][l]
					# dimension
					if kg.D == 0:
						CrystalFieldH.H0[row,col] = CrystalFieldH.H0[row,col] + e
					elif kg.D == 1:
						CrystalFieldH.H0[row,col,:] = CrystalFieldH.H0[row,col,:] + e
					elif kg.D == 2:
						CrystalFieldH.H0[row,col,:,:] = CrystalFieldH.H0[row,col,:,:] + e
					elif kg.D == 3:
						CrystalFieldH.H0[row,col,:,:,:] = CrystalFieldH.H0[row,col,:,:,:] + e
					else:
						print("Error: wrong k grid size")
						sys.exit(1)
