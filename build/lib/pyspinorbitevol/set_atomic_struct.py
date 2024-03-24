from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.atomic_struct_class import Psi4_AtomsStructureClass, QE_AtomsStructureClass
from pyspinorbitevol.parser import parser
from pyspinorbitevol.cell_class import QE_cell_class
import psi4

#
#     System class
#
class Psi4_SystemClass():
    def __init__(self):
        self.atomic_struct = None
        self.wfn = None
    def check_number_irrep(self, wfn):
        psi4.compare_values(wfn.nirrep(), 1, 6, 'n. irrep.')
    # set up atoms system
    def init_atomic_structure(self):
        self.atomic_struct = Psi4_AtomsStructureClass()
    def set_up_wfn(self):
        self.wfn = psi4.core.Wavefunction.build(self.atomic_struct.uc_struct.geometry, psi4.core.get_global_option('BASIS'))
        self.check_number_irrep(self.wfn)
    #  compute the atomic struct.
    def setup_atomic_structure(self):
        # set up atomic structure
        self.init_atomic_structure()
        self.atomic_struct.compute_cell_structure()
        self.atomic_struct.set_sites_list()
        self.set_up_wfn()
        # orbital basis
        self.atomic_struct.set_orbital_basis(self.wfn)
        # this is a struct made of unit_cell_atoms, basis set and sites_list
        # no info on periodicity
        if p.D > 0:
            # make auxiliary structure -> extended
            # it includes all nearest neighbors outside unit cell
            self.atomic_struct.set_supercell_struct()
            self.atomic_struct.set_sites_list()
            self.atomic_struct.set_orbital_basis()

#
# QE system class
class QE_SystemClass():
    def __init__(self):
        self.atomic_struct = None
        self.k_grid = None
    # set up atoms system
    def init_atomic_structure(self):
        self.atomic_struct = QE_AtomsStructureClass()
    # self cell structure
    def set_cell(self):
        self.cell = QE_cell_class()
        self.cell.set_primitive_vectors()
        # cell volume
    #  set k grid
    def set_kgrid(self):
        pass
    #  compute the atomic struct.
    def setup_atomic_structure(self):
        # first atomic structure
        self.init_atomic_structure()
        # build cell structure
        self.set_cell()
        self.atomic_struct.set_sites_list()
        # set k grid

#
## global system object
arguments = parser.parse_args()
code = arguments.calc_typ
if code == "PSI4":
    System = Psi4_SystemClass()
elif code == "QE":
    System = QE_SystemClass()