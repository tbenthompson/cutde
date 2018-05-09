from setuptools import setup

version = open('VERSION').read()

try:
    import pypandoc
    description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    description = open('README.md').read()

setup(
    packages = ['tde'],

    install_requires = [],
    zip_safe = False,

    name = 'tde',
    version = version,
    description = '',
    long_description = description,

    url = 'https://github.com/tbenthompson/tde',
    author = 'T. Ben Thompson',
    author_email = 't.ben.thompson@gmail.com',
    license = 'MIT',
    platforms = ['any']
)
