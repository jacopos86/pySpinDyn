from pyspinorbitevol.crystal_field_class import CrystalFieldHamilt
from pyspinorbitevol.unit_cell_class import uc
from pyspinorbitevol.kpoints_class import kg
from pyspinorbitevol.spin_operators_class import SpinMomentumOperators
from pyspinorbitevol.onsite_energies import set_onsite_energies
from pyspinorbitevol.crystal_field_gradient_class import CrystalFieldHamiltGradient
from pyspinorbitevol.ground_state_calc import GroundState
from pyspinorbitevol.expect_values import spin_expect_val, orbital_expect_val
from pyspinorbitevol.psi4_molecule_class import psi4_molecule_class
import matplotlib.pyplot as plt
from pyspinorbitevol.set_atomic_struct import System
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.psi4_driver import psi4_main_driver, psi4_geometry_driver
from pyspinorbitevol.sphbasis_driver import sphbasis_driver
from pyspinorbitevol.basis_set_module import write_basis_lib_file
import sys
#
inp = sys.argv[1]
p.read_data(inp)
# unit cell object
System.init_atomic_structure()
# collect unique elements list
elem_list = []
f = open(p.coordinate_file)
N = int(f.readline())
head = f.readline()
for i in range(N):
    line = f.readline().split()
    elem_list.append(line[0])
# write basis set file
if not p.sph_basis:
    write_basis_lib_file(p.basis_file_name, elem_list)
psi4_geometry_driver()
# set up atomic structures
System.setup_atomic_structures()
kg.set_kgrid(uc)
kg.set_kpts_weights()
#
if not p.sph_basis:
    psi4_main_driver()
else:
    sphbasis_driver(molecule)
#sites_list.add_site_to_list('Fe', 1.0, [0., 0., 0.], [0., 0., 0.], [0, 1, 2], 8)
#sites_list.add_site_to_list('Gd', 1.0, [1., 0., 0.], [0., 0., 0.], [0, 1, 2, 3])
#print(sites_list.latt_orbital_mom)
sys.exit()
#
uc.set_structure(sites_list.Atomslist, kg)
#sys.exit()
uc.set_nn_atoms(sites_list.Atomslist, kg)
t = {('Fe','Fe') : {'ss' : 0.1, 'sp' : 0.2, 'ps' : 0.2, 'pp' : [0.2,0.3], 'sd' : 0.05, 'ds' : 0.05, 'pd' : [0.1, 0.2], 'dp' : [0.1, 0.2], 'dd' : [0.05, 0.06, 0.06]}}
#
#t = {('Fe','Gd') : {'ss' : 0.1, 'sp' : 0.2, 'ps' : 0.2, 'pp' : [0.2,0.3], 'sd' : 0.05, 'ds' : 0.05, 'pd' : [0.1, 0.2], 'dp' : [0.1, 0.2], 'dd' : [0.05, 0.06, 0.06], 'sf' : 0.0, 'pf' : [0.05, 0.05], 'df' : [0.01, 0.02, 0.005]}}
#atomic_energies = {'Fe' : {0 : -1.0, 1 : 12.0, 2 : -2.0}, 'Gd' : {0 : -1.0, 1 : 12.0, 2 :-2.0, 3 : -5.0}}
atomic_energies = {'Fe' : {0 : -1.0, 1 : 6.0, 2 : -2.0}}
CrystalFieldH = CrystalFieldHamilt(t, sites_list, kg, uc, MatrixEntry)
#
orbital_operators = OrbitalMomentumOperators(sites_list, kg, MatrixEntry)
spin_operators = SpinMomentumOperators(sites_list, kg, MatrixEntry)
print(spin_operators.S.shape)
set_onsite_energies(CrystalFieldH, sites_list, kg, MatrixEntry, atomic_energies)
#
#gt = {('Fe','Gd') : {'ss' : [0.1, 0.01]}}
#CrystalFieldGrad = CrystalFieldHamiltGradient(gt, sites_list, kg, uc, MatrixEntry)
#
GS = GroundState(CrystalFieldH, sites_list, kg, uc, MatrixEntry)
GS.plot_band_structure(kg, "./bands-plot.dat")
GS.set_wfc_occupations(sites_list, kg, 0.03)
GS.set_density_matrix(sites_list, kg)
#R = np.matmul(CrystalFieldH.H0, orbital_operators.L0[:,:,2]) - np.matmul(orbital_operators.L0[:,:,2], CrystalFieldH.H0)
#for i in range(sites_list.Nst):
#	for j in range(sites_list.Nst):
#		if abs(R[i,j]) > 1.E-7:
#			print(i,j,R[i,j])
dos = GS.compute_elec_DOS(kg, sites_list, "./DOS", [0., 10.], 0.05, 0.01)
Sexp = spin_expect_val(spin_operators, GS.rho_e, sites_list, kg, MatrixEntry)
orbital_expect_val(orbital_operators, GS.rho_e, sites_list, kg, MatrixEntry)
