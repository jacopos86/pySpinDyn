import pyscf
from pyscf import gto, mp, scf, mcscf
from pyscf.tools.cubegen import density

mol = gto.M(
    atom = './NiO/nio.xyz',
    basis= 'def2tzvppd',
    spin = 2,
    verbose = 4,
    output = 'output.log')
mol.cart = True
mf = mol.ROHF()
mf.chkfile = 'step4a.chk'
mf.init_guess = 'chkfile'
mf.max_cycle = 50
mf.run(max_cycle=50)
print(f"@ROHF Energy Newton = {mf.e_tot}")
mf.analyze(verbose=5)
dm = mf.make_rdm1(mf.stability()[0], mf.mo_occ)
density(mol, 'a_edens.cube', dm[0])
density(mol, 'b_edens.cube', dm[1])