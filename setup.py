from setuptools import setup, Extension

module = Extension('neuromorph',sources=['NeuroMorph.c', 'hashmap.c'])

setup(
    name="NeuroMorph",
    version='1.0',
    description='Graphical neural network framework',
    ext_modules=[module]
)


