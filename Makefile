bgd:
	make -C src/ bgd

doc: bgd
	make -C src/ build
	make -C doc/ html
	rm -f src/bgd/*.so

.PHONY: bgd, doc
