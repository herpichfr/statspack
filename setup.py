from setuptools import setup, find_packages
from setuptools_scm import get_version

setup(
    name='statspack',
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'colorlog',
    ],
    author='Fabio R Herpich',
    author_email='fabio.herpich@ast.cam.ac.uk',
    description='A statistical visualization package optimized to work with percentiles and histograms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/herpichfr/statspack',
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
