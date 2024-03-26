#
#  This module defines the system's
#  unit cell class
#  with number of periodic dimensions : D
#  lattice vectors
#  reciprocal vectors
#
import numpy as np
import os
from pymatgen.core.lattice import Lattice
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.psi4_molecule_class import psi4_molecule_class
#
# PSI4 derived class
class psi4_cell_class:
	def __init__(self):
		self.cell = None
	# cell atomic structure
	def build_cell(self):
		# look for optimized geometry
		isExist = os.path.exists(p.optimized_coordinate_file)
		if isExist:
			self.cell = psi4_molecule_class(p.optimized_coordinate_file)
		else:
			self.cell = psi4_molecule_class(p.coordinate_file)
		self.cell.set_num_electrons()
		self.cell.print_info_data()
# QE cell class
#
# use phonopy here to build the super cell from unit cell information
class QE_cell_class:
	def __init__(self):
		self.volume = None
		self.cell = None
		self.lattice = None
		self.rec_lattice = None
		self.primitive_vect = None
		self.scaled_coords = None
		self.at_species = None
	def set_volume(self):
		# primitive vectors
		a1 = self.primitive_vect[0,:]
		a2 = self.primitive_vect[1,:]
		a3 = self.primitive_vect[2,:]
		a23 = np.cross(a2, a3)
		self.volume = np.dot(a1, a23)
		log.info("\t unit cell volume = " + str(self.volume) + " Ang^3")
		# Ang^3
	def set_primitive_vectors(self):
		# primitive vectors -> ang
		self.primitive_vect = np.array(p.lattice_vect)*p.lattice_ang
		self.lattice = Lattice(self.primitive_vect)
	def set_rec_lattice(self):
		self.rec_lattice = self.lattice.reciprocal_lattice
	def build_cell(self):
		# set up the unit cell
		unit_cell = PhonopyAtoms(cell=self.primitive_vect,
                    scaled_positions=p.atoms_coords, symbols=p.atoms_symb)
		phonon = Phonopy(unitcell=unit_cell, supercell_matrix=p.supercell_size)
		self.cell = phonon.get_supercell()
	def print_number_of_atoms(self):
		log.info("number of atoms : " + str(self.nat))
	def print_cell_charge(self):
		log.info("cell charge : " + str(self.charge))
	def print_nuclear_charge(self):
		log.info("cell nuclear charge : " + str(self.nuclear_charge))
	def print_num_electrons(self):
		log.info("number of electrons : " + str(self.nelec))
	def set_number_of_atoms(self):
		self.nat = len(self.cell.get_chemical_symbols())
	def get_number_of_atoms(self):
		return self.nat
	def set_cell_charge(self):
		self.charge = p.charge
	def set_electrons_number(self):
		self.nelec = self.nuclear_charge - self.charge
	def set_nuclear_charge(self):
		zv = 0.
		symb_lst = self.cell.get_chemical_symbols()
		for ia in range(self.nat):
			pseudo_file = p.pseudo_dir + '/' + p.pseudo[symb_lst[ia]]
			zv += p.read_Zv_from_pseudo_file(pseudo_file)
		self.nuclear_charge = zv