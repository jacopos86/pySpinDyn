#
#  This module defines the system's
#  unit cell class
#  with number of periodic dimensions : D
#  lattice vectors
#  reciprocal vectors
#
import numpy as np
from abc import ABC
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pyspinorbitevol.utility_functions import norm_realv
from pyspinorbitevol.phys_constants import eps
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.parser import parser
import warnings
# abstract implementation
class UnitCell_class(ABC):
	def __init__(self):
		self.volume = None
		self.cell_struct = None
		self.lattice = None
		self.rec_lattice = None
		self.primitive_vect = None
	def set_volume(self, a1, a2, a3):
		a23 = np.cross(a2, a3)
		self.volume = np.dot(a1, a23)
		log.info("\t unit cell volume = " + str(self.volume) + " Ang^3")
		# Ang^3
	def set_primitive_vectors(self):
		# primitive vectors -> ang
		self.primitive_vect = np.array(p.lattice_vect)*p.latticep_ang
		self.lattice = Lattice(self.primitive_vect)
	def set_rec_lattice(self):
		self.rec_lattice = self.lattice.reciprocal_lattice
	def set_cell_structure(self, atomic_structure):
		# set up the full atomic structure
		Atomslist = atomic_structure.sites_list.Atomslist
		species = []
		for Site in Atomslist:
			elem = Site.element
			species.append(elem)
		coords = []
		for Site in Atomslist:
			R = Site.R0
			coords.append(R)
		# set up the structure
		self.cell_struct = Structure(lattice=self.lattice, species=species, coords=coords,
			charge=p.charge, validate_proximity=True, coords_are_cartesian=True)

#
# PSI4 derived class
class psi4_UnitCell_class(UnitCell_class):
	def __init__(self):
		super().__init__()
		# primitive vectors must be in angstrom
		self.NNlist = None
	def set_nn_atoms(self, atomic_structure):
		Atomslist = atomic_structure.sites_list.Atomslist
		self.NNlist = []
		#
		# periodic dimension 0
		assert p.D == 0
		# suppress warnings
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			NNstruct = CrystalNN()
			for i in range(len(Atomslist)):
				nndata = NNstruct.get_nn_info(self.cell_struct, i)
				# store ONLY first nn in list (weight = 1)
				# check if the atom is in the structure
				nndata2 = []
				for data in nndata:
					R0 = Atomslist[data['site'].index].R0
					d = data['site'].coords - R0
					if norm_realv(d) < eps:
						#print(i, data['site'].coords, data['site'].index)
						nndata2.append(data)
				# store ONLY first nn in list (weight = 1)
				self.NNlist.append(nndata2)

# QE cell class
#
# use phonopy here to build the super cell from unit cell information
class QE_UnitCell_class(UnitCell_class):
	def __init__(self):
		super().__init__()
		self.super_cell = None
	def build_supercell(self, atomic_structure):
		# set up the full atomic structure
		Atomslist = atomic_structure.sites_list.Atomslist
		# atomic species
		species = []
		for Site in Atomslist:
			elem = Site.element
			species.append(elem)
		# atom coordinates
		coords = []
		for Site in Atomslist:
			R = Site.R0
			coords.append(R)
		# set up the unit cell
		unit_cell = PhonopyAtoms(cell=self.primitive_vect,
                    scaled_positions=coords, symbols=species)
		phonon = Phonopy(unitcell=unit_cell, supercell_matrix=p.supercell_size)
		self.super_cell = phonon.get_supercell()

# 
#  set uc object
arguments = parser.parse_args()
code = arguments.calc_typ
if code == "PSI4":
	uc = psi4_UnitCell_class()
elif code == "QE":
    uc = QE_UnitCell_class()
else:
    uc = None