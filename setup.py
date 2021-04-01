from setuptools import setup

version = open("VERSION").read()

description = open("README.md").read()

setup(
    packages=["cutde"],
    install_requires=[],
    zip_safe=False,
    include_package_data=True,
    name="cutde",
    version=version,
    description="130 million TDEs per second, Python + CUDA TDEs from Nikkhoo and Walter 2015",  # noqa:E501
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/tbenthompson/cutde",
    author="T. Ben Thompson",
    author_email="t.ben.thompson@gmail.com",
    license="MIT",
    platforms=["any"],
)
