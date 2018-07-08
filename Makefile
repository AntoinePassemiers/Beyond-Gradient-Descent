REMOTE_ORIGIN=git@github.com:antoinepassemiers/Beyond-Gradient-Descent.git
DOCS_DIR     =../BGD-docs/
THIS_FILE    =$(lastword $(MAKEFILE_LIST))

bgd:
	make -C src/ bgd

${DOCS_DIR}:
	mkdir -p $@ && \
	cd $@ && \
	git clone -b gh-pages --single-branch ${REMOTE_ORIGIN} html 

doc: | ${DOCS_DIR}
	make -C src/ build
	SPHINX_APIDOC_OPTIONS='members,private-members,show-inheritance' sphinx-apidoc -f -M -e -o doc/source/ src/bgd/ src/bgd/setup.py
	make -C doc/ html
	rm -f src/bgd/*.so

pushdoc: | ${DOCS_DIR}
	cd ${DOCS_DIR}/html && \
	git clean -xdf && \
	git pull origin gh-pages && \
	${MAKE} -f ${THIS_FILE} doc && \
	git add . && \
	git commit -m "Build the docs" && \
	git push origin gh-pages

.PHONY: bgd doc pushdoc
