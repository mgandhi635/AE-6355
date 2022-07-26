try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'AE 6355 Project',
    'author': 'Manan Gandhi',
    'author_email': 'mgandhi@gatech.edu',
    'version': '1.0',
    'packages': ['edl_control'],
    'scripts': [],
    'name': 'edl_control'
}

setup(**config)