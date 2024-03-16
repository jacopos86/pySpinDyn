import psi4
import logging
import sys
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.overlap_matr_module import S_obj
from pyspinorbitevol.orbital_operators_module import L_obj
from pyspinorbitevol.set_atomic_struct import System
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.molecular_orbitals_module import MO_obj
from pyspinorbitevol.psi4_molecule_class import psi4_molecule_class
from pyspinorbitevol.phys_constants import bohr_to_ang
from pyspinorbitevol.density_matrix_module import DM_obj
from pyspinorbitevol.hamiltonian_class import H0
import numpy as np

def set_up_calc_parameters():
    basis_name = p.basis_file_name[:-4]
    psi4.set_options(
        {   
            'basis': basis_name,
            'scf_type': p.scf_psi4, 
            'reference': p.psi4_reference,
            'e_convergence': p.psi4_e_converg,
            'd_convergence': p.psi4_d_converg,
            'maxiter' : p.psi4_maxiter,
            'guess' : 'SAD',
            'soscf' : True,
            'soscf_max_iter' : 40,
        }
    )
    log.info("frozen core: " + str(psi4.core.get_global_option('FREEZE_CORE')))

def geometry_optimization():
    init_struct = psi4_molecule_class(p.coordinate_file)
    basis_name = p.basis_file_name[:-4]
    name = "scf/" + basis_name
    E_SCF_psi, wfn_psi = psi4.optimize(name, molecule=init_struct.geometry, return_wfn=True)
    # save data on file
    natoms= init_struct.geometry.natom()
    # open file
    f = open(p.optimized_coordinate_file, 'w')
    f.write("%d\n" % natoms)
    f.write("Angstrom\n")
    for i in range(natoms):
        symb = init_struct.geometry.label(i).lower()
        symb = symb.capitalize()
        # bohr
        Ri = np.array([init_struct.geometry.x(i),init_struct.geometry.y(i),init_struct.geometry.z(i)])
        Ri[:] = Ri[:] * bohr_to_ang
        # ang units
        f.write(symb + "       " + str(Ri[0]) + "       " + str(Ri[1]) + "       " + str(Ri[2]) + "\n")
    f.close()
    return E_SCF_psi, wfn_psi

def operators_initialization(wfn):
    log.info("start model initialization")
    mints = psi4.core.MintsHelper(System.wfn.basisset())
    # compute nuclear energy
    H0.set_nuclear_repulsion_energy()
    # one particle energy operators
    H0.set_ao_kinetic_operator(mints)
    H0.set_ao_one_part_potential(mints)
    H0.set_ao_one_part_hamiltonian()
    # set one particle energy gradients
    H0.set_ao_one_part_hamiltonian_gradient(mints)
    
    # set AO angular momentum
    #L_obj.set_orbital_operators()
    #L_obj.set_L0(mints)
    #print(L_obj.L0[2,4:10,4:10])
    #H = np.asarray(mints.so_kinetic()) + np.asarray(mints.so_potential())
    #R = np.matmul(H, L_obj.L0[2,:,:]) - np.matmul(L_obj.L0[2,:,:], H)
    # orbital initialization
    MO_obj.set_mo_from_wfn(wfn)
    MO_obj.print_info()
    MO_obj.set_orbital_space()
    # density matrix
    DM_obj.set_occup()
    DM_obj.set_orbital_occup()
    DM_obj.compute_dm_from_mo()
    DM_obj.set_active_space_dm()
    DM_obj.set_ae_space_dm()
    
    # overlap matrix
    S_obj.initialize_overlap_matr()
    S_obj.set_overlap_matr(mints)
    S_obj.set_overlap_matr_grad(mints)
    if log.level <= logging.INFO:
        S_obj.check_overlap_matr_properties()
    #dS1 = np.asarray(mints.overlap_grad(), order='F')
    # gradient overlap matrix

def set_MO_basis_operators():
    # 1p hamiltonian
    H0.set_ae_1p_matr_elements()

def psi4_geometry_driver():
    # initialize calculation parameters
    set_up_calc_parameters()
    # optimize geometry
    E_SCF, wfn = geometry_optimization()
    return E_SCF, wfn

def tests():
    # TESTS #############
    MO_obj.check_orthogonality(S_obj.S)
    DM_obj.compare_total_electron_number()
    DM_obj.compare_active_electron_number()

def energy_tests(E_SCF):
    # ENERGY TESTS ######
    log.info("SCF total energy : " + str(E_SCF) + " H")

def psi4_main_driver(E_SCF, wfn):
    # model initialization
    operators_initialization(wfn)
    # tests
    tests()
    # finalize calculation -> MO basis conversion
    set_MO_basis_operators()
    # energy tests
    energy_tests(E_SCF)