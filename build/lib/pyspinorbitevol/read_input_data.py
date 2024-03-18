from pyspinorbitevol.logging_module import log
from pyspinorbitevol.parser import parser
import yaml
import numpy as np
from abc import ABC
#
class input_data_class(ABC):
    # initialization
    def __init__(self):
        # working directory
        self.working_dir = ''
        # energy convergence treshold
        self.e_converg = 1.e-7
        # phonons -> atom dynamics
        self.phonons = True
        # active space -> size
        self.active_space = 0
        # n. active electrons in calc.
        self.active_electrons = 0
        # periodic dimension
        self.D = 0
    # read input data
    def read_shared_data(self, data):
        # other input parameters
        # working directory
        if 'working_directory' in data:
            self.working_dir = data['working_directory']
        #  active space -> dimension
        if 'active_space' in data:
            self.active_space = data['active_space']
        # n. active electrons
        if 'active_electrons' in data:
            self.active_electrons = data['active_electrons']
        # system's charge
        if 'charge' in data:
            self.charge = data['charge']
        # phonons
        if 'phonons' in data:
            self.ph = data['phonons']
#
#  PSI4 parameters class
class psi4_input_data_class(input_data_class):
    def __init__(self):
        super().__init__()
        #
        # psi4 data
        self.scf_psi4 = 'direct'
        # reference calculation
        self.psi4_reference = 'uks'
        # max. iterations
        self.psi4_maxiter = 1000
        # orbital init.
        self.orbital_initialization = "SAD"
        # sph. basis set
        self.sph_basis = False
        # d convergence
        self.psi4_d_converg = 1.e-6
        # basis set file
        self.basis_file_name = ''
        # multiplicity
        self.multiplicity = 1
    def read_data(self, input_file):
        try :
            f = open(input_file)
        except :
            msg = "could not find: " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        # read shared variables
        self.read_shared_data(data)
        assert self.D == 0
        # psi4 section
        if p.sph_basis and 'llist' in data:
            self.llist = data['llist']
        if not p.sph_basis:
            # atomic basis set
            if 'basis_set' in data:
                self.psi4_basis = data['basis_set']
            # type scf calculation
            if 'scf_type' in data:
                self.scf_psi4 = data['scf_type']
            # reference calc.
            if 'reference' in data:
                self.psi4_reference = data['reference']
            # d convergence
            if 'd_convergence' in data:
                self.psi4_d_converg = data['d_convergence']
            # orbital initialization
            if 'orbital_initialization' in data:
                self.orbital_initialization = data['orbital_initialization']
            # max. iterations
            if 'maxiter' in data:
                self.psi4_maxiter = data['maxiter']
            # basis file name
            if 'basis_file_name' in data:
                self.basis_file_name = data['basis_file_name']
            if 'optimized_coordinate_file' in data:
                self.optimized_coordinate_file = data['optimized_coordinate_file']
            # coordinate files
            if 'coordinate_file' in data:
                self.coordinate_file = data['coordinate_file']
            # multiplicity
            if 'multiplicity' in data:
                self.multiplicity = data['multiplicity']

#
# QE input class
class QE_input_data_class(input_data_class):
    def __init__(self):
        super().__init__()
        # n. kpt -> along each dimension
        # D = 0 -> [1,1,1]
        # D = 1 -> [1,1,nkpt]
        # D = 2 -> [nkpt,nkpt,1]
        # D = 3 -> [nkpt,nkpt,nkpt]
        self.nkpt = 0
        self.kgr  = []
        # cell size
        self.supercell_size = []
        # lattice vectors
        self.lattice_vect = []
        self.lattice_ang = None
        self.atoms_coords = []
        # prefix
        self.prefix = ''
        # non collinearity
        self.noncollinear = True
        # pseudo
        self.pseudo_dir = ''
        self.pseudo = []
        # cut-off
        self.ecutwfc = 60.0
        
    def read_data(self, input_file):
        try :
            f = open(input_file)
        except :
            msg = "could not find: " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        # read shared variables
        self.read_shared_data(data)
        #
        # read data
        # PWSCF part
        if 'prefix' in data:
            self.prefix = data['prefix']
        # super cell size
        if 'supercell_size' in data:
            self.supercell_size = data['supercell_size']
        # periodic dimension
        if 'periodic_dimension' in data:
            self.D = data['periodic_dimension']
        # atoms in unit cell
        if 'unitcell_atoms' in data:
            self.atoms_coords = np.array(data['unitcell_atoms'])
        # lattice vectors
        if 'primitive_vectors' in data:
            self.lattice_vect = np.array(data['primitive_vectors'])
        if 'lattice_parameter_ang' in data:
            self.lattice_ang = data['lattice_parameter_ang']
        # atom symbols
        if 'atom_symbols' in data:
            self.atoms_symb = data['atom_symbols']
        # pseudo potentials info
        if 'pseudo_dir' in data:
            self.pseudo_dir = data['pseudo_dir']
        if 'pseudo' in data:
            self.pseudo = data['pseudo']
        # cut-off energy
        if 'ecutwfc' in data:
            self.ecutwfc = data['ecutwfc']
        # non collinearity
        if 'noncollinear' in data:
            self.noncollinear = data['noncollinear']
        # k points
        if 'nkpts' in data:
            self.nkpt = data['nkpts']
            if self.D == 0:
                self.kgr = [1, 1, 1]
            elif self.D == 1:
                self.kgr = [1, 1, self.nkpt]
            elif self.D == 2:
                self.kgr = [self.nkpt, self.nkpt, 1]
            elif self.D == 3:
                self.kgr = [self.nkpt, self.nkpt, self.nkpt]
            else:
                log.error("Wrong periodic D : " + str(self.D))
            
# input parameters object
arguments = parser.parse_args()
code = arguments.calc_typ
if code == "PSI4":
    p = psi4_input_data_class()
elif code == "QE":
    p = QE_input_data_class()
else:
    p = None
p.sep = "*"*94

# read config.yml
try:
    f = open("./config.yml")
except:
    raise Exception("config.yml cannot be opened")
config = yaml.load(f, Loader=yaml.Loader)
f.close()
if 'SPH_BASIS' in config:
    p.sph_basis = config['SPH_BASIS']