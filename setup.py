from setuptools import setup

version = open('VERSION').read()

try:
    import pypandoc
    description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    description = open('README.md').read()

setup(
    packages = ['cutde'],

    install_requires = [],
    zip_safe = False,
    include_package_data = True,

    name = 'cutde',
    version = version,
    description = '',
    long_description = description,

    url = 'https://github.com/tbenthompson/cutde',
    author = 'T. Ben Thompson',
    author_email = 't.ben.thompson@gmail.com',
    license = 'MIT',
    platforms = ['any']
)
