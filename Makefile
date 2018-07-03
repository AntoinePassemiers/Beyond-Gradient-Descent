bgd:
	make -C src/ bgd

doc: bgd
	make -C src/ build
	sphinx-apidoc -f -M -e -o doc/source/ src/bgd/ src/bgd/setup.py
	make -C doc/ html
	rm -f src/bgd/*.so

.PHONY: bgd doc
