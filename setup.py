# from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

from cutde.gpu import get_template, template_with_mako

version = open("VERSION").read()

description = open("README.md").read()

tmpl_dir = "cutde"
tmpl_name = "pairs.cu"
tmpl_args = dict(float_type="double")

tmpl = get_template(tmpl_name, tmpl_dir)
code = template_with_mako(tmpl, tmpl_args)
# write to temp file

ext_modules = [
    # Pybind11Extension(
    #     "python_example",
    #     sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
    # ),
]

setup(
    packages=["cutde"],
    install_requires=[],
    ext_modules=ext_modules,
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
