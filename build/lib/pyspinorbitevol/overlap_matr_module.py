import numpy as np
import psi4
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.kpoints_class import kg
from pyspinorbitevol.set_atomic_struct import System

class ao_overlap_class:

    def __init__(self):
        self.S = None
        self.TOLER = None

    def initialize_overlap_matr(self):
        nbf = System.atomic_struct.bs.nbf
        # check periodicity
        if p.ph or p.D == 0:
            self.S = np.zeros((nbf, nbf))
        elif not p.ph and p.D == 1:
            i0 = np.where(np.array(kg.nkpts) > 0)[0][0]
            nk = kg.nkpts[i0]
            self.S = np.zeros((nbf, nbf, nk))
        elif not p.ph and p.D == 2:
            [i0, i1] = np.where(np.array(kg.nkpts) > 0)[0]
            nk1 = kg.nkpts[i0]
            nk2 = kg.nkpts[i1]
            self.S = np.zeros((nbf, nbf, nk1, nk2))
        elif not p.ph and p.D == 3:
            [nk1, nk2, nk3] = kg.nkpts
            self.S = np.zeros((nbf, nbf, nk1, nk2, nk3))
        # gradient overlap
        nat = System.atomic_struct.sites_list.natoms
        self.gradS = np.zeros((3,nat,nbf,nbf))

    def set_overlap_matr(self, mints):
        # extract psi4 -> S
        nbf = System.atomic_struct.bs.nbf
        S = np.asarray(mints.ao_overlap())
        # check dimensionality
        assert nbf == S.shape[0]
        for i in range(nbf):
            for j in range(nbf):
                self.S[i,j] = S[i,j]

    def set_overlap_matr_grad(self, mints):
        # extract psi4 -> gS
        nbf = System.atomic_struct.bs.nbf
        nat = System.atomic_struct.sites_list.natoms
        for ia in range(nat):
            gS = mints.ao_oei_deriv1("OVERLAP", ia)
            for idx in range(3):
                A = gS[idx].to_array()
                # dimensionality
                assert nbf == A.shape[0] and nbf == A.shape[1]
                self.gradS[idx,ia,:,:] = A[:,:]

    def check_overlap_matr_properties(self):
        nbf = System.atomic_struct.bs.nbf
        # check coeffs
        if p.ph or p.D == 0:
            for i in range(nbf):
                if np.abs(self.S[i,i]-1.) > self.TOLER:
                    log.error("S[i,i] element != 1")
                for j in range(nbf):
                    if i != j:
                        if self.S[i,j] > 1.+self.TOLER:
                            log.error("S[i,j] > 1 : " + str(i) + ", " + str(j) + ' ' + str(self.S[i,j]))
            psi4.compare_values(1, 1, 6, 'overlap matrix')

    def set_overlap_matr_sph_basis(self):
        pass

S_obj = ao_overlap_class()
S_obj.TOLER = 1.E-6