from setuptools import find_packages
from setuptools import setup

# with open("requirements.txt") as file:
#     REQUIRED_PACKAGES = file.read()

setup(
    name='CoLM2024',
    version='0.0.1',
    description=('Landmodel python.'),
    long_description='',
    url='https://xxxxxxx',
    author='Qingliang Li, Jinlong Zhu',
    author_email='',
    # install_requires=REQUIRED_PACKAGES,
    # packages=find_packages(include=['colm', 'colm.*']),
    packages=["colm"],
    # PyPI package information.
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Atmosphere',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=(
        'landmodel leaftemperature'))