# orbital-evol-model-code
this code computes the coupled spin orbital evolution with fully conserved J = L + S orbital momentum in 0D, 1D, 2D, 3D periodic systems\
1- it implements spin orbit coupling\
2- Ehrenfest atomic motion or electrons + phonon quantum dynamics\
3- full orbital momentum conservation
# installation procedure
conda create -n pyspindyn python=3.8 "numpy>=1.16.5,<1.23.0" scipy\
conda activate pyspindyn\
pip install matplotlib\
pip install pymatgen\
pip install colorlog\
pip install mendeleev\
conda install psi4::psi4\
conda install conda-forge::openbabel\
pip install basis_set_exchange\
conda install -c conda-forge phonopy\
python setup.py install
