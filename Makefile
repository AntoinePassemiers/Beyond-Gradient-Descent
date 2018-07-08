REMOTE_ORIGIN=git@github.com:antoinepassemiers/Beyond-Gradient-Descent.git
DOCS_DIR     =../BGD-docs/

bgd:
	make -C src/ bgd

doc: 
	make -C src/ build
	SPHINX_APIDOC_OPTIONS='members,private-members,show-inheritance' sphinx-apidoc -f -M -e -o doc/source/ src/bgd/ src/bgd/setup.py
	make -C doc/ html
	rm -f src/bgd/*.so

pushdoc: doc
	cd ${DOCS_DIR}/html && \
	if [ ! -d ".git" ]; then git init; git remote add origin ${REMOTE_ORIGIN}; fi && \
	git add . && \
	git commit -m "Build the docs" && \
	git push -f origin HEAD:gh-pages

.PHONY: bgd doc pushdoc
