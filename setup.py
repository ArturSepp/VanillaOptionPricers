from setuptools import setup, find_packages


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


def read_file(file):
    with open(file) as f:
        return f.read()


long_description = read_file("README.md")
requirements = read_requirements("requirements.txt")

setup(
    name='vanilla_option_pricers',
    version='1.1.2',
    author='Artur Sepp',
    author_email='artursepp@gmail.com',
    url='https://github.com/ArturSepp/VanillaOptionPricers',
    description='Implementation of fast pricers for Black-Scholes-Merton and Bachelier models',
    long_description_content_type="text/x-rst",  # If this causes a warning, upgrade your setuptools package
    long_description=long_description,
    license="MIT license",
    packages=find_packages(exclude=["examples/figures"]),  # Don't include test directory in binary distribution
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)