import os
import psi4
from openbabel import openbabel
from pyspinorbitevol.logging_module import log
from pyspinorbitevol.read_input_data import p

class psi4_molecule_class:

    def __init__(self, input_geometry_xyz):
        self.nel = 0
        self.nuclear_charge = 0
        input_geometry_list = self.xyz_to_zmatr(input_geometry_xyz)
        input_geometry_zmatr = ""
        for line in input_geometry_list:
            input_geometry_zmatr += line
            input_geometry_zmatr += "\n"
        input_geometry_zmatr += "symmetry c1"
        self.set_geometry(input_geometry_zmatr)
        self.set_charge_multiplicity()

    def xyz_to_zmatr(self, xyz_file):
        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats("xyz", "gzmat")
        mol = openbabel.OBMol()
        conv.ReadFile(mol, xyz_file)
        conv.WriteFile(mol, "tmp.gzmat")
        zmatr = self.generate_zmatr()
        return zmatr

    def generate_zmatr(self):
        f = open("tmp.gzmat", 'r')
        data = f.readlines()
        for i in range(len(data)):
            l = data[i].split()
            if len(l) > 1:
                if len(l) == 2 and l[0] == '0':
                    j = i+1
                    break
        if i == len(data)-1:
            raise Exception("problem reading xyz file")
        f.close()
        os.remove("tmp.gzmat")
        new_text = []
        variables = self.extract_variables(data)
        for i in range(j, len(data)):
            l = data[i].split()
            if len(l) > 0:
                if l[0] == "Variables:":
                    break
                else:
                    new_line = ""
                    for c in l:
                        if c in variables:
                            new_line += str(variables[c])
                        else:
                            new_line += c
                        new_line += " "
                    new_text.append(new_line)
        return new_text
    
    def extract_variables(self, data):
        var_dict = {}
        j = 0
        for line in data:
            l = line.split()
            if len(l) > 0:
                if l[0] == "Variables:":
                    j += 1
                    break
            j += 1
        for k in range(j, len(data)):
            l = data[k].strip().split('=')
            if len(l) == 2:
                var_dict[l[0]] = float(l[1])
        return var_dict

    def set_geometry(self, data):
        try:
            self.geometry = psi4.geometry(data)
        except:
            raise Exception("Error: cannot build geometry")

    def print_num_atoms(self):
        print("number of atoms= ", self.geometry.natom())

    def set_nuclear_charge(self):
        self.nuclear_charge = 0
        Z = 0
        for ia in range(self.geometry.natom()):
            Z += self.geometry.Z(ia)
        self.nuclear_charge = Z

    def set_charge_multiplicity(self):
        self.geometry.set_multiplicity(p.multiplicity)
        self.geometry.set_molecular_charge(p.charge)

    def set_num_electrons(self):
        self.set_nuclear_charge()
        self.nel = self.nuclear_charge - self.geometry.molecular_charge()
        self.nel = int(self.nel)

    def set_nuclear_repulsion_energy(self):
        return self.geometry.nuclear_repulsion_energy()
        
    def print_info_data(self):
        log.info("Total num. electrons= " + str(self.nel))
        log.info("Multiplicity: " + str(self.geometry.multiplicity()))
        log.info("System charge: " + str(self.geometry.molecular_charge()))