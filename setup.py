from setuptools import setup, find_packages
import os


def parse_requirements(file):
    return sorted(set(
        line.partition('#')[0].strip()
        for line in open(os.path.join(os.path.dirname(__file__), file))
    ) - set(''))


setup(
	name='NN-ET0',
	python_requires='>=3.8.9',
	version='0.0.1',
	description='NN-ET0',
	author='SOWIT',
	packages=find_packages(where='src'),
	install_requires=parse_requirements('requirements.txt'),
	package_dir={'': 'src'}
	)