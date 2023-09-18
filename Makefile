build:
	python3 setup.py build
	python3 setup.py sdist bdist_wheel
	pip install dist/*.whl --force-reinstall

clean:
	rm -rf build
	rm -rf dist
	rm -rf NeuroMorph.egg-info
