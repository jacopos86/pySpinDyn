install :
	python setup.py install
.PHONY :
	clean test
clean :
	rm -rf ./pyspinorbitevol/*~ ./pyspinorbitevol/__pycache__ ./build/lib/pyspinorbitevol/* ./__pycache__ ./timer.dat
