REMOTE_ORIGIN=git@github.com:antoinepassemiers/Beyond-Gradient-Descent.git
DOCS_DIR     =../BGD-docs/

bgd:
	make -C src/ bgd

test: bgd
	pytest test/test.py

doc: 
	make -C src/ build
	export SPHINX_APIDOC_OPTIONS=members,private-members,show-inheritance,ignore-module-all && \
	sphinx-apidoc -Mef -o doc/source/ src/bgd/ src/bgd/setup.py
	make -C doc/ html

pushdoc: doc
	cd ${DOCS_DIR}/html && \
	if [ ! -d ".git" ]; then git init; git remote add origin ${REMOTE_ORIGIN}; fi && \
	git add . && \
	git commit -m "Build the docs" && \
	git push -f origin HEAD:gh-pages

clean:
	make -C doc/ clean
	make -C src/ clean

.PHONY: bgd doc pushdoc clean
