""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('xmixers/version.py').read())
setup(
    name='xmixers',
    version=__version__,
    description='Xmixers: A collection of SOTA efficient token/channel mixers',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Doraemonzzz/xmixers',
    author='Doraemonzzz',
    author_email='doraemon_zzz@163.com',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch pretrained models efficientnet mobilenetv3 mnasnet resnet vision transformer vit',
    packages=find_packages(exclude=['convert', 'tests', 'results']),
    include_package_data=True,
    install_requires=['torch >= 1.7'],
    python_requires='>=3.7',
)