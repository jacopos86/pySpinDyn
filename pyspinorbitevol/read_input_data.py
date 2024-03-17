from pyspinorbitevol.logging_module import log
import yaml
import numpy as np
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
        # periodic dimension
        if 'periodic_dimension' in data:
            self.D = data['periodic_dimension']
        if 'lattice_parameter_ang' in data:
            self.latt_ang = data['lattice_parameter_ang']
        if 'unit_cell' in data:
            if data['unit_cell'] == "FCC":
                self.primitive_vectors[0] = 0.5*self.latt_ang*np.array([1., 1., 0.])
                self.primitive_vectors[1] = 0.5*self.latt_ang*np.array([0., 1., 1.])
                self.primitive_vectors[2] = 0.5*self.latt_ang*np.array([1., 0., 1.])
            elif data['unit_cell'] == "None":
                self.primitive_vectors[0] = self.latt_ang*np.array([1., 0., 0.])
                self.primitive_vectors[1] = self.latt_ang*np.array([0., 1., 0.])
                self.primitive_vectors[2] = self.latt_ang*np.array([0., 0., 1.])
            elif data['unit_cell'] == "SC":
                self.primitive_vectors[0] = self.latt_ang*np.array([1.,0.,0.])
                self.primitive_vectors[1] = self.latt_ang*np.array([0.,1.,0.])
                self.primitive_vectors[2] = self.latt_ang*np.array([0.,0.,1.])
            else:
                raise Exception("unit cell not recognised")
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
        if 'nkpts' in data:
            self.nkpt = data['nkpt']
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
    p = None
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