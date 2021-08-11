import os

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

version = open("VERSION").read()

description = open("README.md").read()

for float_type in ["float", "double"]:
    tmpl_args = dict(float_type=float_type)

    import mako.lookup

    tmpl_fp = "cutde/cpp_backend.cpp"
    tmpl_name = os.path.basename(tmpl_fp)
    lookup = mako.lookup.TemplateLookup(directories=["cutde"])
    tmpl = lookup.get_template(tmpl_name)
    try:
        rendered_tmpl = tmpl.render(**tmpl_args, backend="cpp", preamble="")
    except:  # noqa: E722
        # bare except is okay because we re-raise immediately
        import mako.exceptions

        print(mako.exceptions.text_error_template().render())
        raise

    rendered_fp = os.path.join(
        os.path.dirname(tmpl_fp), f".rendered.{float_type}.{tmpl_name}"
    )
    with open(rendered_fp, "w") as f:
        f.write(rendered_tmpl)

ext_modules = [
    Pybind11Extension(
        f"cutde.cpp_backend_{float_type}",
        [f"cutde/.rendered.{float_type}.cpp_backend.cpp"],
        extra_compile_args=["-fopenmp", "-std=c++17"],
        extra_link_args=["-fopenmp"],
    )
    for float_type in ["float", "double"]
]

setup(
    packages=["cutde"],
    install_requires=["mako", "pybind11"],
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
