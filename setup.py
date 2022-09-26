#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'lagrtools',
    version          = '0.0.1',
    author           = 'Dmitrii Torbunov',
    author_email     = 'dtorbunov@bnl.gov',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = 'A collection of tools to handle Wire-Cell Graphs',
    packages         = setuptools.find_packages(
        include = [ 'lagrtools', 'lagrtools.*' ]
    ),
    install_requires = [ 'numpy' ],
)

