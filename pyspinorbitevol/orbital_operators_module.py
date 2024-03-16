import numpy as np
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.kpoints_class import kg
from pyspinorbitevol.orbital_mtxel_functions import L0x_mtxel, L0y_mtxel, L0z_mtxel
from pyspinorbitevol.set_atomic_struct import System

class orbital_operators_class:

    def __init__(self):
        self.L0 = None
        self.Lel = None
        self.Lph = None

    def set_orbital_operators(self):
        nbf = System.atomic_struct.bs.nbf
        # initialize L0 -> atom localized OM
        # initialize Lel -> electron full orbital operator
        # initialize Lph -> Lel + phonon orbital momentum
        if p.ph or p.D == 0:
            self.L0 = np.zeros((3, nbf, nbf), dtype=np.complex128)
            self.Lel= np.zeros((3, nbf, nbf), dtype=np.complex128)
            self.Lph= np.zeros((3, nbf, nbf), dtype=np.complex128)
        elif not p.ph and p.D == 1:
            i0 = np.where(np.array(kg.nkpts) > 0)[0][0]
            nk = kg.nkpts[i0]
            self.L0 = np.zeros((3, nbf, nbf, nk), dtype=np.complex128)
            self.Lel= np.zeros((3, nbf, nbf, nk), dtype=np.complex128)
            self.Lph= np.zeros((3, nbf, nbf, nk), dtype=np.complex128)
        elif not p.ph and p.D == 2:
            [i0, i1] = np.where(np.array(kg.nkpts) > 0)[0]
            nk1 = kg.nkpts[i0]
            nk2 = kg.nkpts[i1]
            self.L0 = np.zeros((3, nbf, nbf, nk1, nk2), dtype=np.complex128)
            self.Lel= np.zeros((3, nbf, nbf, nk1, nk2), dtype=np.complex128)
            self.Lph= np.zeros((3, nbf, nbf, nk1, nk2), dtype=np.complex128)
        elif not p.ph and p.D == 3:
            [nk1, nk2, nk3] = kg.nkpts
            self.L0 = np.zeros((3, nbf, nbf, nk1, nk2, nk3), dtype=np.complex128)
            self.Lel= np.zeros((3, nbf, nbf, nk1, nk2, nk3), dtype=np.complex128)
            self.Lph= np.zeros((3, nbf, nbf, nk1, nk2, nk3), dtype=np.complex128)

    def set_L0(self, mints=None):
        if p.sph_basis:
            self.set_L0_sph_basis()
        else:
            self.set_L0_ao_set(mints)

    def set_L0_sph_basis(self):
        # set up structure
        sites_list = System.atomic_struct.sites_list
        bs = System.atomic_struct.bs
        # run over atom sites
        for ist in range(sites_list.natoms):
            site = sites_list.Atomslist[ist]
            shl = bs.orbital_set[site.element]
            for sh1 in shl:
                l1 = sh1['l']
                for ml1 in range(-l1, l1+1):
                    for sh2 in shl:
                        l2 = sh2['l']
                        for ml2 in range(-l2, l2+1):
                            for ms in [-0.5, 0.5]:
                                row = bs.matrix_entry(ist, l1, ml1, ms)
                                col = bs.matrix_entry(ist, l2, ml2, ms)
                                if p.ph or p.D == 0:
                                    self.L0[0,row,col] = L0x_mtxel(l1, ml1, l2, ml2)
                                    self.L0[1,row,col] = L0y_mtxel(l1, ml1, l2, ml2)
                                    self.L0[2,row,col] = L0z_mtxel(l1, ml1, l2, ml2)
                                elif not p.ph and p.D == 1:
                                    self.L0[0,row,col,:] = L0x_mtxel(l1, ml1, l2, ml2)
                                    self.L0[1,row,col,:] = L0y_mtxel(l1, ml1, l2, ml2)
                                    self.L0[2,row,col,:] = L0z_mtxel(l1, ml1, l2, ml2)
                                elif not p.ph and p.D == 2:
                                    self.L0[0,row,col,:,:] = L0x_mtxel(l1, ml1, l2, ml2)
                                    self.L0[1,row,col,:,:] = L0y_mtxel(l1, ml1, l2, ml2)
                                    self.L0[2,row,col,:,:] = L0z_mtxel(l1, ml1, l2, ml2)
                                elif not p.ph and p.D == 3:
                                    self.L0[0,row,col,:,:,:] = L0x_mtxel(l1, ml1, l2, ml2)
                                    self.L0[1,row,col,:,:,:] = L0y_mtxel(l1, ml1, l2, ml2)
                                    self.L0[2,row,col,:,:,:] = L0z_mtxel(l1, ml1, l2, ml2)
                                else:
                                    log.error("Error: wrong k grid size")
                                    raise Exception("Error: wrong k grid size")

    def set_L0_ao_set(self, mints):
        L0_psi4 = np.asarray(mints.so_angular_momentum())
        # check if phonons present
        sites_list = System.atomic_struct.sites_list
        bs = System.atomic_struct.bs
        # check p.ph
        if p.ph or p.D == 0:
            for row in range(int(bs.nbf/2)):
                for col in range(int(bs.nbf/2)):
                    self.L0[:,2*row,2*col]     = L0_psi4[:,row,col]
                    self.L0[:,2*row+1,2*col+1] = L0_psi4[:,row,col]
                    
L_obj = orbital_operators_class()