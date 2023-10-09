import os
from setuptools import find_packages, setup

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
# read the docs could not compile numpy and c extensions
if on_rtd:
    setup_requires = []
    install_requires = []
else:
    install_requires = [
        "six",
        "numpy",
        "scipy",
        "matplotlib",
    ]

long_description = (
    "See `github <https://github.com/leepand/RLXtreme>`_ " "for more information."
)

setup(
    name="RLXtreme",
    version="0.0.1",
    description="Contextual bandit/RL in python",
    long_description=long_description,
    author="Leepand",
    license="Apache 2.0",
    author_email="pandeng.li@163.com",
    url="https://github.com/leepand/RLXtreme",
    install_requires=install_requires,

    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3",
    ],
    test_suite="nose.collector",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
