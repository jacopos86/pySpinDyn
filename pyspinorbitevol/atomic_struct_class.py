from abc import ABC
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.atomic_sites_class import QE_AtomicSiteList, psi4_AtomicSiteList
from pyspinorbitevol.basis_set_module import sph_harm_basis_class, psi4_basis_class
from pyspinorbitevol.cell_class import QE_cell_class
#
# atomic structure
class AtomsStructureClass(ABC):
    def __init__(self):
        self.sites_list = None
        self.cell_struct = None

#
# psi4 atomic structure
class Psi4_AtomsStructureClass(AtomsStructureClass):
    def __init__(self):
        super().__init__()
        if p.sph_basis:
            self.bs = sph_harm_basis_class()
        else:
            self.bs = psi4_basis_class()
        self.sites_list = psi4_AtomicSiteList()
    # orbital basis
    def set_orbital_basis(self, wfn):
        self.bs.set_up_basis_set(self.cell_struct, self.sites_list, wfn)
    # compute cell structure
    def compute_cell_structure(self):
        self.cell_struct.build_cell()
    # sites list
    def set_sites_list(self):
        # initialize atoms list
        self.sites_list.initialize_atoms_list(self.cell_struct)
        self.sites_list.print_geometry(self.cell_struct)
#
# QE atomic structure
class QE_AtomsStructureClass(AtomsStructureClass):
    def __init__(self):
        super().__init__()
        self.sites_list = QE_AtomicSiteList()
    # compute cell structure
    def compute_cell_structure(self):
        self.cell_struct = QE_cell_class()
        self.cell_struct.set_primitive_vectors()
        self.cell_struct.set_volume()
        self.cell_struct.set_rec_lattice()
        self.cell_struct.build_cell()
    # sites list
    def set_sites_list(self):
        # initialize atoms list
        self.sites_list.initialize_atoms_list(self.cell_struct)