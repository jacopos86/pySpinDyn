from pyspinorbitevol.logging_module import log
import yaml
import numpy as np
#
class input_data_class():
    # initialization
    def __init__(self):
        # psi4 data
        self.scf_psi4 = 'direct'
        self.psi4_reference = 'uks'
        self.psi4_e_converg = 1.e-7
        self.psi4_d_converg = 1.e-6
        self.psi4_maxiter = 1000
        self.orbital_initialization = "SAD"
        self.sph_basis = None
        self.primitive_vectors = [None]*3
    # read input data
    def read_data(self, input_file):
        try :
            f = open(input_file)
        except :
            msg = "could not find: " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        # psi4 section
        if 'psi4_parameters' in data and not p.sph_basis:
            if 'basis_set' in data['psi4_parameters']:
                self.psi4_basis = data['psi4_parameters']['basis_set']
            if 'scf_type' in data['psi4_parameters']:
                self.scf_psi4 = data['psi4_parameters']['scf_type']
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
        # other input parameters
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
        if 'active_space' in data:
            self.nact = data['active_space']
        if 'active_els' in data:
            self.act_els = data['active_els']
#
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