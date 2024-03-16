from pyspinorbitevol.logging_module import log
from pyspinorbitevol.read_input_data import p
import numpy as np
import psi4
import basis_set_exchange
import os

class sph_harm_basis_class:
    def __init__(self):
        self.nbf = 0
        self.orbital_set = None
    # set orbital l for each element
    def set_orbital_states(self):
        self.orbital_set = {}
        for Element in p.llist:
            self.orbital_set[Element] = []
        for Element in p.llist:
            llist = p.llist[Element]
            for l in llist:
                self.orbital_set[Element].append({'l' : l})
    # set number basis functions
    def set_bset_size(self, molecule):
        for ia in range(molecule.geometry.natom()):
            Element = molecule.geometry.fsymbol(ia)
            llist = self.orbital_set[Element]
            for ll in llist:
                l = ll['l']
                nst = 2*l+1
                self.nbf += 2*nst
        log.info("nbf: " + str(self.nbf))
    # matrix entry method
    def matrix_entry(self, sites_list, isite, l, ml, ms):
        k = 0
        for i in range(isite):
            el = sites_list.Atomslist[i].element
            shl= self.orbital_set[el]
            for sh in shl:
                il = sh['l']
                k += 2*(2*il+1)
        for il in range(l):
            k += 2*(2*il+1)
        mllist = np.arange(-l, ml, 1)
        k += 2*len(mllist)
        if ms == 0.5:
            k = k
        elif ms == -0.5:
            k += 1
        else:
            raise Exception("Error in matrix entry -> ms != +/-0.5")
            log.error("Error in matrix entry -> ms != +/-0.5")
        return k
    # full set up method
    def set_up_basis_set(self, molecule):
        self.set_orbital_states()
        self.set_bset_size(molecule)

class psi4_basis_class:
    def __init__(self):
        self.nbf = 0
        self.basis_obj = None
        self.basis_to_atom = None
    # set up psi4 basis obj
    def set_basis_obj(self, molecule):
        self.basis_obj = psi4.core.BasisSet.build(molecule.geometry)
    # orbital states
    def set_orbital_states(self, sites_list):
        self.nshell = self.basis_obj.nshell()
        log.info("n. orbital shells: " + str(self.nshell))
        #for ish in range(self.nshell):
        #    print(ish, self.basis_obj.shell_to_basis_function(ish), self.basis_obj.shell_to_center(ish))
        #for ia in range(sites_list.natoms):
        #    print(ia, self.basis_obj.nshell_on_center(ia))
        self.nshell_site = []
        for ia in range(sites_list.natoms):
            self.nshell_site.append(self.basis_obj.nshell_on_center(ia))
        self.nbfunc_site = np.zeros(sites_list.natoms, dtype=int)
        self.nstate_shell = np.zeros(self.nshell, dtype=int)
        for ish in range(self.nshell-1):
            nbf = self.basis_obj.shell_to_basis_function(ish+1) - self.basis_obj.shell_to_basis_function(ish)
            self.nstate_shell[ish] = nbf
            self.nbfunc_site[self.basis_obj.shell_to_center(ish)] += nbf
        self.nstate_shell[self.nshell-1] = self.basis_obj.nbf() - sum(self.nbfunc_site)
        self.nbfunc_site[self.basis_obj.shell_to_center(self.nshell-1)] += self.basis_obj.nbf() - sum(self.nbfunc_site)
        assert sum(self.nbfunc_site) == self.nbf
    # set n. basis functions
    def set_bset_size(self, wfn):
        nsopi = list(wfn.nsopi())
        nbf = sum(nsopi)
        # n. basis functions
        self.nbf = int(self.basis_obj.nbf())
        log.info("nbf: " + str(self.nbf))
        assert nbf == self.nbf
        # n. AOs
        self.nao = int(self.basis_obj.nao())
        log.info("nao: " + str(self.nao))
        # n. frozen core
        self.ncore = int(self.basis_obj.n_frozen_core())
        log.info("ncore: " + str(self.ncore))
        # n. ecp core
        self.necp = int(self.basis_obj.n_ecp_core())
        log.info("n. ecp electrons: " + str(self.necp))

    # local matrix entry method
    def matrix_entry(self, isite, sh, bf, ms):
        k = 0
        for i in range(isite):
            k += 2*self.nbfunc_site[i]
            # include spin pol.
        for ish in range(sh):
            k += 2*self.nstate_shell[ish]
        ibf = 0
        while ibf < bf:
            k += 2
            ibf += 1
        if ms == 0.5:
            k = k
        elif ms == -0.5:
            k += 1
        else:
            raise Exception("Error in matrix entry -> ms != +/-0.5")
            log.error("Error in matrix entry -> ms != +/-0.5")
        return k
    def print_info(self):
        self.basis_obj.print_detail_out()
    # full set up method
    def set_up_basis_set(self, molecule, sites_list, wfn):
        self.set_basis_obj(molecule)
        self.print_info()
        self.set_bset_size(wfn)
        self.set_orbital_states(sites_list)

# 
# write BASIS LIB file
def write_basis_lib_file(basis_set_file, unique_elem_list):
    special_basis_types = {
        "SBKJC" : "SBKJC-VDZ"
    }
    isExist = os.path.exists(basis_set_file)
    if isExist:
        os.remove(basis_set_file)
    for symbol in unique_elem_list:
        basis_name = p.psi4_basis[symbol]
        if basis_name in special_basis_types:
            basis_name = special_basis_types[basis_name]
        basis_str = basis_set_exchange.get_basis(basis_name, elements=symbol, fmt="psi4", header=False)
        if basis_str is None:
            log.error("basis set not found for element: " + symbol)
        # write to file
        with open(basis_set_file, "a") as lib_f:
            lib_f.write(basis_str+"\n")
#
# interface write basis set
def prepare_basis_set():
    # first collect unique element list from file
    elem_list = []
    f = open(p.coordinate_file, 'r')
    N = int(f.readline())
    header = f.readline()
    for i in range(N):
        line = f.readline().split()
        elem_list.append(line[0])
    elem_list = list(set(elem_list))
    # write basis set
    write_basis_lib_file(p.basis_file_name, elem_list)