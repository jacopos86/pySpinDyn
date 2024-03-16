from pyspinorbitevol.orbital_operators_module import L_obj

def sph_model_initialization():
    # set overlap matrix -> diag
    # set orbital operators
    L_obj.set_orbital_operators()
    L_obj.set_L0()

def sphbasis_driver():
    # sph. model
    sph_model_initialization()