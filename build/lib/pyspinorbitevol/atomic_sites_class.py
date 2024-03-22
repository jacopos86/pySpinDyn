import numpy as np
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.phys_constants import bohr_to_ang, mp
from abc import ABC
#
#   This module implements
#   the atomic site class
#
class AtomicSite:
	def __init__(self, Element, Mass, Z, R0, V0):
		self.element = Element
		# mass in eV fs^2/Ang^2
		self.mass = Mass
		# atomic number
		self.Z = Z
		# R0 in Ang
		self.R0 = np.array(R0)
		# V0 in Ang/fs
		self.V0 = np.array(V0)
		# delta R
		self.dR0 = np.array([0., 0., 0.])
		# delta V
		self.dV0 = np.array([0., 0., 0.])
	def update_position(self, R0):
		self.R0 = np.array(R0)
	def update_velocity(self, V0):
		self.V0 = np.array(V0)
	def set_orbital_momentum(self):
		self.L0 = self.mass * np.cross(self.R0, self.V0)
		# eV fs units
	def set_relative_position(self, Rcm):
		self.dR0 = self.R0 - Rcm
	def set_relative_velocity(self, Vcm):
		self.dV0 = self.V0 - Vcm
#
#   AtomicSiteList class
#
class AtomicSiteList(ABC):
	def __init__(self):
		self.Atomslist = []
		self.Rcm = np.zeros(3)
		self.Vcm = np.zeros(3)
		self.M = 0.
		self.latt_orbital_mom = np.zeros(3)
	def add_site_to_list(self, Element, Mass, Z, R, V):
		# set atomic site
		site = AtomicSite(Element, Mass, Z, R, V)
		site.set_orbital_momentum()
		# append site to list
		self.Atomslist.append(site)
	def set_total_mass(self):
		self.M = 0
		for site in self.Atomslist:
			self.M = self.M + site.mass
	def set_center_of_mass_position(self):
		Rcm = np.zeros(3)
		for site in self.Atomslist:
			Rcm[:] += site.mass * site.R0[:]
		Rcm[:] = Rcm[:] / self.M
		self.Rcm = Rcm
	def set_center_of_mass_velocity(self):
		vcm = np.zeros(3)
		for site in self.Atomslist:
			vcm[:] = vcm[:] + site.mass * site.V0[:]
		vcm[:] = vcm[:] / self.M
		self.Vcm = vcm
	def set_lattice_orbital_momentum(self):
		self.latt_orbital_mom[:] = 0.
		for site in self.Atomslist:
			self.latt_orbital_mom = self.latt_orbital_mom + site.L0

#
# Psi4 subclass
class psi4_AtomicSiteList(AtomicSiteList):
	def __init__(self):
		super().__init__()
	# set number of atoms
	def set_natoms(self, molecule):
		self.natoms = molecule.geometry.natom()
		log.info("number of atoms: " + str(self.natoms))
	def initialize_atoms_list(self, molecule):
		self.set_natoms(molecule)
		# make atom list
		for ia in range(self.natoms):
			Element = molecule.geometry.fsymbol(ia)
			if len(Element) == 2:
				c = Element[-1]
				Element = Element[0] + c.lower()
			Mass = molecule.geometry.mass(ia) * mp
			# eV fs^2 / ang^2
			Z = molecule.geometry.fZ(ia)
			# coordinates
			R = np.array([molecule.geometry.x(ia), molecule.geometry.y(ia), molecule.geometry.z(ia)])
			R[:] = R[:] * bohr_to_ang
			V = np.zeros(3)
			# add site to list
			self.add_site_to_list(Element, Mass, Z, R, V)
		# center of mass
		self.set_total_mass()
		self.set_center_of_mass_position()
		self.set_center_of_mass_velocity()
		self.set_lattice_orbital_momentum()
		for ia in range(self.natoms):
			self.Atomslist[ia].set_relative_position(self.Rcm)
			self.Atomslist[ia].set_relative_velocity(self.Vcm)
	def print_geometry(self, molecule):
		molecule.geometry.print_out_in_angstrom()

#
# QE subclass
class QE_AtomicSiteList(AtomicSiteList):
	def __init__(self):
		super().__init__()
	# set number of atoms
	def set_natoms(self, cell_struct):
		cell_struct.set_number_of_atoms()
		self.natoms = cell_struct.get_number_of_atoms()
		cell_struct.print_number_of_atoms()
	def initialize_atoms_list(self, cell_struct):
		self.set_natoms(cell_struct)
		# symbols
		symb_lst = cell_struct.get_chemical_symbols()
		# make atoms list
		for ia in range(self.natoms):
			Element = symb_lst[ia]