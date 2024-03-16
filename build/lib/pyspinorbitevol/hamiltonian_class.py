from pyspinorbitevol.set_atomic_struct import System
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.molecular_orbitals_module import MO_obj
import numpy as np
#
# define active space hamiltonian
class hamiltonian_class:
    def __init__(self):
        self.H1p = None
        self.H2p = None
        # Vnn
        self.Vnn = None
        # TEI -> MO basis
        self.Iijkl = None
        # OEI -> MO basis
        self.hij = None
        self.T_ao = None
        self.V_ao = None
        self.H_ao = None
        self.gH_ao= None
    # set nuclear interaction energy
    def set_nuclear_repulsion_energy(self):
        self.Vnn = System.atomic_struct.uc_struct.set_nuclear_repulsion_energy()
        log.info("Nuclear repulsion energy: " + str(self.Vnn))
    # set AO kinetic operator
    def set_ao_kinetic_operator(self, mints):
        self.T_ao = np.asarray(mints.ao_kinetic())
    # set AO one particle potential
    def set_ao_one_part_potential(self, mints):
        self.V_ao = np.asarray(mints.ao_potential())
    # set AO one particle hamiltonian
    def set_ao_one_part_hamiltonian(self):
        self.H_ao = self.T_ao + self.V_ao
    # set AO one particle hamiltonian gradient
    def set_ao_one_part_hamiltonian_gradient(self, mints):
        # compute gH0
        nbf = System.atomic_struct.bs.nbf
        nat = System.atomic_struct.sites_list.natoms
        self.gH_ao = np.zeros((3,nat,nbf,nbf))
        # run over atom index
        for ia in range(nat):
            gT = mints.ao_oei_deriv1("KINETIC", ia)
            for idx in range(3):
                A = gT[idx].to_array()
                # dimensionality
                assert nbf == A.shape[0] and nbf == A.shape[1]
                self.gH_ao[idx,ia,:,:] = A[:,:]
            gV = mints.ao_oei_deriv1("POTENTIAL", ia)
            for idx in range(3):
                A = gV[idx].to_array()
                # dimensionality
                assert nbf == A.shape[0] and nbf == A.shape[1]
                self.gH_ao[idx,ia,:,:] += A[:,:]
    # AE single particle matr. elements
    def set_ae_1p_matr_elements(self):
        log.info("compute single particle matr. elements")
        self.hij = [None]*2
        if p.psi4_reference == 'rhf':
            HC = np.matmul(self.H_ao, MO_obj.C)
            self.hij[0] = np.matmul(MO_obj.C.T, HC)
            self.hij[1] = self.hij[0]
        elif p.psi4_reference == 'rohf' or p.psi4_reference == 'uhf':
            HC = np.matmul(self.H_ao, MO_obj.Ca)
            self.hij[0] = np.matmul(MO_obj.Ca.T, HC)
            HC = np.matmul(self.H_ao, MO_obj.Cb)
            self.hij[1] = np.matmul(MO_obj.Cb.T, HC)
H0 = hamiltonian_class()