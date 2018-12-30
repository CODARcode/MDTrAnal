LIB_DIR = lib
TEST_DIR = test
TEST_SCRIPT_NAME = run_tests.sh

.PHONY: test

default: build-test-script
	cd work/orient-v3 && make -f GNUmakefile

build-test-script:
	echo '#!/bin/bash' > $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'script_dir=$$(dirname "$$(readlink -f "$$0")")' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'export PYTHONPATH=$$script_dir/../$(LIB_DIR):$$PATH:$$PYTHONPATH' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'cd test' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'python3 -u -m unittest discover -p "test_*.py"' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	chmod +x $(TEST_DIR)/$(TEST_SCRIPT_NAME)
test:
	bash $(TEST_DIR)/$(TEST_SCRIPT_NAME)

