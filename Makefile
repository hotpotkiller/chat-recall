PYTHON ?= python3
NODE ?= node
export NODE
.PHONY: bootstrap dev check check-js test-integration eval package
bootstrap:
	$(PYTHON) scripts/bootstrap.py
dev:
	$(PYTHON) -m chat_recall serve --demo
check:
	$(PYTHON) scripts/check_source.py
	$(PYTHON) -m unittest discover -s tests -v
check-js:
	$(PYTHON) scripts/check_js.py
test-integration:
	$(PYTHON) -m unittest discover -s tests -p 'test_server.py' -v
eval:
	$(PYTHON) scripts/evaluate.py
package:
	$(PYTHON) scripts/package.py
