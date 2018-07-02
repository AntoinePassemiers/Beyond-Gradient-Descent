bgd:
	make -C src/ bgd

doc: bgd
	make -C src/ build
	make -C doc/ html

.PHONY: bgd, doc
