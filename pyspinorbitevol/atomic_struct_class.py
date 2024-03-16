from pyspinorbitevol.psi4_molecule_class import psi4_molecule_class
from pyspinorbitevol.read_input_data import p
from pyspinorbitevol.atomic_sites_class import AtomicSiteList
from pyspinorbitevol.basis_set_module import sph_harm_basis_class, psi4_basis_class
from pyspinorbitevol.unit_cell_class import uc
import os

# atomic structure
class AtomsStructureClass():
    def __init__(self):
        if p.sph_basis:
            self.bs = sph_harm_basis_class()
        else:
            self.bs = psi4_basis_class()
        self.sites_list = AtomicSiteList()
        self.uc_struct = None
    # instance
    def generate_instance(self):
        if p.D > 0:
            return PeriodicAtomsStructClass()
        else:
            return AperiodicAtomsStructClass()
    # unit cell atoms struct
    def set_uc_atomic_system(self):
        # look for optimized geometry
        isExist = os.path.exists(p.optimized_coordinate_file)
        if isExist:
            self.uc_struct = psi4_molecule_class(p.optimized_coordinate_file)
        else:
            self.uc_struct = psi4_molecule_class(p.coordinate_file)
        self.uc_struct.set_num_electrons()
        self.uc_struct.print_info_data()
    # sites list
    def set_sites_list(self):
        # initialize atoms list
        self.sites_list.initialize_atoms_list(self.uc_struct)
        self.sites_list.print_geometry(self.uc_struct)
    # orbital basis
    def set_orbital_basis(self, wfn):
        self.bs.set_up_basis_set(self.uc_struct, self.sites_list, wfn)

class AperiodicAtomsStructClass(AtomsStructureClass):
    def __init__(self):
        super(AperiodicAtomsStructClass, self).__init__()

class PeriodicAtomsStructClass(AtomsStructureClass):
    def __init__(self):
        super(PeriodicAtomsStructClass, self).__init__()
        self.sc_struct = None
    def set_supercell_struct(self):
        # generate temporary xyz file
        Atomslist0 = self.sites_list.Atomslist
        extended_atoms_list = []
        for site in range(len(Atomslist0)):
            atom0 = Atomslist0[site]
            atom_dict = {'symbol': atom0.element, 'coordinate': atom0.R0}
            insert_dict = True
            for atom_1 in extended_atoms_list:
                if (atom_1['coordinate'] == atom_dict['coordinate']).all():
                    insert_dict = False
                    break
            if insert_dict:
                extended_atoms_list.append(atom_dict)
            nndata = uc.NNlist[site]
            for atom in nndata:
                atom_dict = {'symbol': Atomslist0[atom['site'].index].element, 'coordinate': atom['site'].coords}
                insert_dict = True
                for atom_1 in extended_atoms_list:
                    if (atom_1['coordinate'] == atom_dict['coordinate']).all():
                        insert_dict = False
                        break
                if insert_dict:
                    extended_atoms_list.append(atom_dict)
        # write temp xyz file
        lines = []
        line = str(len(extended_atoms_list)) + "\n"
        lines.append(line)
        line = "Angstrom\n"
        lines.append(line)
        for atom_1 in extended_atoms_list:
            line = ""
            line = atom_1['symbol'] 
            line += " " + str(atom_1['coordinate'][0]) 
            line += " " + str(atom_1['coordinate'][1]) 
            line += " " + str(atom_1['coordinate'][2]) 
            line += "\n"
            lines.append(line)
        with open("./tmp_struct.xyz", 'w') as fp:
            for line in lines:
                fp.write(line)
        self.sc_struct = psi4_molecule_class("./tmp_struct.xyz")
        self.sc_struct.set_num_electrons()
        self.sc_struct.print_num_elec()