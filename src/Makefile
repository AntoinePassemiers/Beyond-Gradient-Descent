bgd:
	python3 setup.py install

build:
	python3 setup.py build_ext --inplace

clean:
	rm -f bgd/*.c
	rm -f bgd/**/*.c
	rm -f bgd/*.so
	rm -f bgd/**/*.so
	rm -f classes.eps
	rm -f packages.eps
	rm -fr build/

uml:
	pyreverse -o eps bgd/*.py

.PHONY: bgd build clean uml
