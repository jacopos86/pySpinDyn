from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.set_atomic_struct import System
import numpy as np
from mendeleev import element

class molecular_orbital_class:

    def __init__(self):
        self.C = None
        self.Ca= None
        self.Cb= None
        self.nmo = 0

    def print_info(self):
        log.info("n. molecular orbitals: " + str(self.nmo))

    def set_mo_from_wfn(self, wfn):
        if p.psi4_reference == 'rhf':
            self.C = np.asarray(wfn.Ca())
            self.nmo = self.C.shape[0]
        elif p.psi4_reference == 'rohf' or p.psi4_reference == 'uhf':
            self.Ca = np.asarray(wfn.Ca())
            self.Cb = np.asarray(wfn.Cb())
            self.nmo= self.Ca.shape[0]

    def set_orbital_space(self):
        log.info("active space: " + str(p.nact))
        log.info("active electrons: " + str(p.act_els))
        # total number core states
        necp_el = System.atomic_struct.bs.necp
        nelec = 0
        # total n. electrons
        nat = System.atomic_struct.sites_list.natoms
        for ia in range(nat):
            symb = System.atomic_struct.sites_list.Atomslist[ia].element
            at = element(symb)
            nelec += at.atomic_number
        nelec -= p.charge
        self.ncore = nelec - necp_el - p.act_els
        log.info("n. core electrons: " + str(self.ncore))
        self.ncore_st = int(self.ncore / 2)

    # check orthogonality MOs
    def check_orthogonality(self, S):
        # C^T S C = I
        if p.psi4_reference == 'rhf':
            CTS = self.C.T.dot(S)
            ONE = CTS.dot(self.C)
            for i in range(ONE.shape[0]):
                for j in range(ONE.shape[1]):
                    if i == j:
                        if abs(ONE[i,j] - 1.) > p.psi4_d_converg:
                            print(ONE[i,j])
                            log.error("C^T S C = I -> TEST FAILED")
                    else:
                        if abs(ONE[i,j]) > p.psi4_d_converg:
                            print(ONE[i,j])
                            log.error("C^T S C = I -> TEST FAILED")
        elif p.psi4_reference == 'rohf' or p.psi4_reference == 'uhf':
            CaTS = self.Ca.T.dot(S)
            ONEa = CaTS.dot(self.Ca)
            CbTS = self.Cb.T.dot(S)
            ONEb = CbTS.dot(self.Cb)
            for i in range(ONEa.shape[0]):
                for j in range(ONEa.shape[1]):
                    if i == j:
                        if abs(ONEa[i,j] - 1.) > 1.E-3 or abs(ONEb[i,j] - 1.) > 1.E-3:
                            print(ONEa[i,j], ONEb[i,j])
                            log.error("C^T S C = I -> TEST FAILED")
                    else:
                        if abs(ONEa[i,j]) > 1.E-3 or abs(ONEb[i,j]) > 1.E-3:
                            print(ONEa[i,j], ONEb[i,j])
                            log.error("C^T S C = I -> TEST FAILED")
        log.warning("C^T S C = I -> TEST SUCCEDED")

MO_obj = molecular_orbital_class()