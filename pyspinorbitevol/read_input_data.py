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
        try :
            f = open(input_file)
        except :
            msg = "could not find: " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
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
        if 'coordinate_file' in data:
            self.coordinate_file = data['coordinate_file']
        if 'optimized_coordinate_file' in data:
            self.optimized_coordinate_file = data['optimized_coordinate_file']
        if 'periodic_dimension' in data:
            self.D = data['periodic_dimension']
        if 'nkpts' in data:
            self.nkpts = data['nkpts']
        if p.sph_basis and 'llist' in data:
            self.llist = data['llist']
        if 'phonons' in data:
            self.ph = data['phonons']
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
        self.sph_basis = None
        # d convergence
        self.psi4_d_converg = 1.e-6
    def read_data(self, data):
        # psi4 section
        if not p.sph_basis:
            # atomic basis set
            if 'basis_set' in data:
                self.psi4_basis = data['basis_set']
            # type scf calculation
            if 'scf_type' in data:
                self.scf_psi4 = data['scf_type']
            if 'reference' in data['psi4_parameters']:
                self.psi4_reference = data['psi4_parameters']['reference']
            if 'e_convergence' in data['psi4_parameters']:
                self.psi4_e_converg = data['psi4_parameters']['e_convergence']
            if 'd_convergence' in data['psi4_parameters']:
                self.psi4_d_converg = data['psi4_parameters']['d_convergence']
            if 'psi4_orbital_initialization' in data['psi4_parameters']:
                self.orbital_initialization = data['psi4_parameters']['psi4_orbital_initialization']
            if 'maxiter' in data['psi4_parameters']:
                self.psi4_maxiter = data['psi4_parameters']['maxiter']
            if 'basis_file_name' in data['psi4_parameters']:
                self.basis_file_name = data['psi4_parameters']['basis_file_name']
            if 'multiplicity' in data['psi4_parameters']:
                self.multiplicity = data['psi4_parameters']['multiplicity']
            if 'charge' in data['psi4_parameters']:
                self.charge = data['psi4_parameters']['charge']

p = input_data_class()
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