import os
from setuptools import setup

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
# read the docs could not compile numpy and c extensions
if on_rtd:
    setup_requires = []
    install_requires = []
else:
    setup_requires = [
        'nose',
        'coverage',
    ]
    install_requires = [
        'six',
        'numpy',
        'scipy',
        'matplotlib',
    ]

long_description = ("See `github <https://github.com/leepand/RLXtreme>`_ "
                    "for more information.")

setup(
    name='RLXtreme',
    version='0.2.5',
    description='Contextual bandit/RL in python',
    long_description=long_description,
    author='Leepand',
    author_email='pandeng.li@163.com',
    url='https://github.com/leepand/RLXtreme',
    setup_requires=setup_requires,
    install_requires=install_requires,
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='nose.collector',
    packages=[
        'rlxtreme',
        'rlxtreme.agent',
        'rlxtreme.storage',
        'rlxtreme.utils',
    ],
    package_dir={
        'rlxtreme': 'rlxtreme',
        'rlxtreme.bandit': 'rlxtreme/agent',
        'rlxtreme.storage': 'rlxtreme/storage',
        'rlxtreme.utils': 'rlxtreme/utils',
    },
)