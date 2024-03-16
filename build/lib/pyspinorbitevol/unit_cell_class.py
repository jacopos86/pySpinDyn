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
		# primitive vectors must be in angstrom
		self.prim_vecs = None
		self.volume = None
		self.struct = None
	def set_primitive_vectors(self, primitive_vectors):
		# primitive vectors
		self.lattice = Lattice(primitive_vectors)
		self.a1 = np.array(primitive_vectors[0])
		self.a2 = np.array(primitive_vectors[1])
		self.a3 = np.array(primitive_vectors[2])
		self.prim_vecs = np.array([self.a1, self.a2, self.a3])
	def set_volume(self):
		a23 = np.cross(self.a2, self.a3)
		self.volume = np.dot(self.a1, a23)
		log.info("unit cell volume= " + str(self.volume) + " Ang^3")
		# Ang^3
	def set_rec_vectors(self):
		self.rec_lattice = self.lattice.reciprocal_lattice
		# b1 = 2pi a2 x a3 / V (ang^-1)
		self.b1 = 2.*np.pi*np.cross(self.a2, self.a3) / self.volume
		# b2 = 2pi a3 x a1 / V
		self.b2 = 2.*np.pi*np.cross(self.a3, self.a1) / self.volume
		# b3 = 2pi a1 x a2 / V
		self.b3 = 2.*np.pi*np.cross(self.a1, self.a2) / self.volume
		self.rec_vecs = np.array([self.b1, self.b2, self.b3])
		# unit vectors
		self.set_rec_versors()
	def set_rec_versors(self):
		self.rcv = [None]*3
		self.rcv[0] = self.b1 / norm_realv(self.b1)
		self.rcv[1] = self.b2 / norm_realv(self.b2)
		self.rcv[2] = self.b3 / norm_realv(self.b3)
  
#
# PSI4 derived class
class psi4_UnitCell_class(UnitCell_class):
	def __init__(self):
		super().__init__()
	def set_structure(self, atomic_structure):
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
		self.struct = Structure(lattice=self.lattice, species=species, coords=coords,
			charge=0, validate_proximity=True, coords_are_cartesian=True)
	def set_nn_atoms(self, atomic_structure):
		Atomslist = atomic_structure.sites_list.Atomslist
		self.NNlist = []
		#
		# periodic dimension 0
		assert p.D == 0
		if p.D == 0:
			# suppress warnings
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				NNstruct = CrystalNN()
				for i in range(len(Atomslist)):
					nndata = NNstruct.get_nn_info(self.struct, i)
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
	def set_structure(self, atomic_structure):
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
		self.struct = PhonopyAtoms(cell=p.lattice_vect*p.lattice_param,
                             scaled_positions=coords, symbols=species)
	def build_supercell(self):
		phonon = Phonopy(unitcell=self.struct, supercell_matrix=p.supercell_size)
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