from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.psi4_driver import psi4_main_driver, psi4_geometry_driver
from pyspinorbitevol.sphbasis_driver import sphbasis_driver
from pyspinorbitevol.basis_set_module import prepare_basis_set
from pyspinorbitevol.unit_cell_class import uc
from pyspinorbitevol.kpoints_class import kg
from pyspinorbitevol.set_atomic_struct import System
from pyspinorbitevol.parser import parser
# read input config data
arguments = parser.parse_args()
p.read_data(arguments.input_file)
code = arguments.calc_typ
#
# branch (1) -> PSI4
# branch (2) -> QE
if code == "PSI4":
    log.info("\t " + p.sep)
    log.info("\n")
    log.info("\t START PSI4 CALCULATION")
    log.info("\n")
    log.info("\t " + p.sep)
    # PSI4 -> ONLY molecular systems
    log.info("SPH BASIS SET: " + str(p.sph_basis))
    # set up system object
    System.init_atomic_structure()
    # prepare/write basis set
    prepare_basis_set()
    # optimize system geometry
    E_SCF, wfn = psi4_geometry_driver()
    # set up atomic structure
    System.setup_atomic_structures()
    # set k grid
    kg.set_kgrid(uc)
    kg.set_kpts_weights()
    # main code driver -> set up ground state calculation
    if not p.sph_basis:
        psi4_main_driver(E_SCF, wfn)
    else:
        sphbasis_driver()
    #print(atomic_structure.sites_list.latt_orbital_mom)
elif code == "QE":
    log.info("\t " + p.sep)
    log.info("\n")
    log.info("\t START QE CALCULATION")
    log.info("\n")
    log.info("\t " + p.sep)
    # first build cell structure