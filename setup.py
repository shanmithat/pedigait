"""
Setup script for Pediatric Gait Digital Twin Analysis package.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pedigait",
    version="1.0.0",
    description="Pediatric Gait Digital Twin Analysis Package",
    author="Pediatric Gait Analysis Team",
    author_email="",
    url="https://github.com/shanforpresident/pedigait",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pedigait=pedigait.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="gait analysis, biomechanics, pediatric, c3d, motion capture, digital twin",
    project_urls={
        "Bug Reports": "https://github.com/shanforpresident/pedigait/issues",
        "Source": "https://github.com/shanforpresident/pedigait",
    },
)
