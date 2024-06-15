from setuptools import setup
from glob import glob
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension("cpp_extensions.cpp_module", sorted(glob("src/cpp_extensions/*.cpp")))
]

setup(
    name="mcdp2",
    version="0.0.2",
    author="Askar Gafurov",
    author_email="askar.gafurov@fmph.uniba.sk",
    license="MIT License",
    description="A tool to compute p-values for number of overlaps between two genome annotations",
    packages=["mcdp2", "mcdp2.common", "mcdp2.models"],
    package_dir={
        "mcdp2": "src/mcdp2",
        "mcdp2.common": "src/mcdp2/common",
        "mcdp2.models": "src/mcdp2/models"
    },
    entry_points={
        "console_scripts": [
            "mcdp2 = mcdp2.main:main"
        ]
    },
    install_requires=[
        'argh',
        'pybind11',
        'pyyaml',
        'pathos',
        'numpy',
        'numba',
        'scipy',
    ],
    ext_modules=ext_modules
)
