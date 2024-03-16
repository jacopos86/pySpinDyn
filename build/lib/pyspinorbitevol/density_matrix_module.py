import numpy as np
import psi4
from pyspinorbitevol.set_atomic_struct import System
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.molecular_orbitals_module import MO_obj
from pyspinorbitevol.overlap_matr_module import S_obj
#
# density matrix module
class density_matrix_class():
    def __init__(self):
        # AO basis
        self.Dao = None
        self.Dact= None
        self.Dae = None
        # occupation numbers
        self.nocc = np.zeros(2, dtype=int)
        self.f = None
        self.fa= None
        self.fb= None
    def set_occup(self):
        nel = System.atomic_struct.uc_struct.nel
        necp_el = System.atomic_struct.bs.necp
        nel -= necp_el
        if p.psi4_reference == 'rhf':
            if nel % 2 != 0:
                log.error("rohf or uhf should be used for open shell systems")
            self.nocc[:] = int(nel / 2)
        elif p.psi4_reference == 'rohf' or p.psi4_reference == 'uhf':
            ms = (System.atomic_struct.uc_struct.geometry.multiplicity() - 1) / 2
            self.nocc[0] = (nel + 2*ms) / 2
            self.nocc[1] = nel - self.nocc[0]
    def set_orbital_occup(self):
        if p.psi4_reference == 'rhf':
            self.f = np.zeros(2*MO_obj.nmo)
            for i in range(self.nocc[0]):
                self.f[2*i] = 1.
            for i in range(self.nocc[1]):
                self.f[2*i+1] = 1.
        elif p.psi4_reference == 'rohf' or p.psi4_reference == 'uhf':
            self.fa = np.zeros(2*MO_obj.nmo)
            self.fb = np.zeros(2*MO_obj.nmo)
            for i in range(self.nocc[0]):
                self.fa[2*i] = 1.
            for i in range(self.nocc[1]):
                self.fb[2*i+1] = 1.
    # DM from MO
    def compute_dm_from_mo(self):
        if p.psi4_reference == 'rhf':
            C = MO_obj.C
            D = np.einsum('mi,ni->mn', C[:,:self.nocc[0]], C[:,:self.nocc[0]])
            self.Dao = [D, D]
        elif p.psi4_reference == 'rohf' or p.psi4_reference == 'uhf':
            Ca = MO_obj.Ca
            Dup = np.einsum('mi,ni->mn', Ca[:,:self.nocc[0]], Ca[:,:self.nocc[0]])
            Cb = MO_obj.Cb
            Ddw = np.einsum('mi,ni->mn', Cb[:,:self.nocc[1]], Cb[:,:self.nocc[1]])
            self.Dao = [Dup, Ddw]
    # check total n. electrons
    def compare_total_electron_number(self):
        nel0 = System.atomic_struct.uc_struct.nel
        necp_el = System.atomic_struct.bs.necp
        nel0 -= necp_el
        log.info("n. electrons: " + str(nel0))
        S = S_obj.S
        # Tr(rho S) = nel
        if p.psi4_reference == 'rhf':
            SD = np.matmul(S, self.D)
            nel = 2 * SD.trace()
        elif p.psi4_reference == 'rohf' or p.psi4_reference == 'uhf':
            Srho = np.matmul(S, self.Dao[0])
            nel_up = Srho.trace()
            Srho = np.matmul(S, self.Dao[1])
            nel_dw = Srho.trace()
            nel = nel_up + nel_dw
            log.info("Tr{S\rho_up}= " + str(nel_up))
            log.info("Tr{S\rho_dw}= " + str(nel_dw))
        psi4.compare_values(nel, nel0, 4, 'number of electrons')
    # compare active space n. electrons
    def compare_active_electron_number(self):
        log.info("n. active electrons: " + str(p.act_els))
        if  p.psi4_reference == 'rhf':
            nel = 2 * np.trace(self.Dact[0])
        elif p.psi4_reference == 'rohf' or p.psi4_reference == 'uhf':
            nel = np.trace(self.Dact[0]) + np.trace(self.Dact[1])
        psi4.compare_values(nel, p.act_els, 4, 'number of active electrons')
    # active space DM
    def set_active_space_dm(self):
        i0 = 2*MO_obj.ncore_st
        if p.psi4_reference == 'rhf':
            D = np.zeros((2*p.nact, 2*p.nact))
            for i in range(i0, i0+2*p.nact):
                D[i-i0,i-i0] = self.f[i]
            self.Dact = [D, D]
        elif p.psi4_reference == 'rohf' or p.psi4_reference == 'uhf':
            Dup = np.zeros((2*p.nact, 2*p.nact))
            Ddw = np.zeros((2*p.nact, 2*p.nact))
            for i in range(i0, i0+2*p.nact):
                Dup[i-i0,i-i0] = self.fa[i]
                Ddw[i-i0,i-i0] = self.fb[i]
            self.Dact = [Dup, Ddw]
    # ae DM
    def set_ae_space_dm(self):
        if p.psi4_reference == 'rhf':
            D = np.zeros((2*MO_obj.nmo, 2*MO_obj.nmo))
            for i in range(2*MO_obj.nmo):
                D[i,i] = self.f[i]
            self.Dae = [D, D]
        elif p.psi4_reference == 'rohf' or p.psi4_reference == 'uhf':
            Dup = np.zeros((2*MO_obj.nmo, 2*MO_obj.nmo))
            Ddw = np.zeros((2*MO_obj.nmo, 2*MO_obj.nmo))
            for i in range(2*MO_obj.nmo):
                Dup[i,i] = self.fa[i]
                Ddw[i,i] = self.fb[i]
            self.Dae = [Dup, Ddw]
DM_obj = density_matrix_class()