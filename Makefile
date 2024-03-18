PYTHON_MODULE_PATH=minimind

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "__pycache__" -type d -delete
	find . -name ".ipynb_checkpoints" -type d -delete

format:
	black ${PYTHON_MODULE_PATH}
	isort --verbose ${PYTHON_MODULE_PATH}
	autoflake --in-place  --remove-all-unused-imports --expand-star-imports --ignore-init-module-imports -r ${PYTHON_MODULE_PATH}
	docformatter --in-place --recursive ${PYTHON_MODULE_PATH}

pylinting:
	## https://vald-phoenix.github.io/pylint-errors/
	pylint --output-format=colorized ${PYTHON_MODULE_PATH}
