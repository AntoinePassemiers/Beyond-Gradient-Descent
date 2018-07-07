REMOTE_ORIGIN=git@github.com:antoinepassemiers/Beyond-Gradient-Descent.git
DOCS_DIR     =../BGD-docs/

bgd:
	make -C src/ bgd

${DOCS_DIR}:
	mkdir -p $@ && \
	cd $@ && \
	git clone ${REMOTE_ORIGIN} html && \
	cd html && \
	git branch gh-pages && \
	git symbolic-ref HEAD refs/heads/gh-pages && \
	rm .git/index && \
	git clean -fdx && \
	git commit -m "Prepare for docs"

doc: | ${DOCS_DIR}
	make -C src/ build
	SPHINX_APIDOC_OPTIONS='members,private-members,show-inheritance' sphinx-apidoc -f -M -e -o doc/source/ src/bgd/ src/bgd/setup.py
	make -C doc/ html
	rm -f src/bgd/*.so

pushdoc: doc
	cd ${DOCS_DIR}/html && \
	git add . && \
	git commit -m "Build the docs" && \
	git push origin gh-pages

.PHONY: bgd doc
