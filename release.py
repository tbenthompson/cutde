"""
Setting up to run this script:
- install github-release from
   https://github.com/github-release/github-release/releases/tag/v0.10.0
- create a github token and set the GITHUB_TOKEN environment variable
- activate your conda environment.
"""

import datetime
import os
import subprocess
import sys

from setuptools.extern import packaging

assert "GITHUB_TOKEN" in os.environ


def run(cmd):
    print(" ".join(cmd))
    os.system(" ".join(cmd))


d = datetime.datetime.now()
version_str = str(packaging.version.Version(d.strftime("%y.%m.%d")))
if len(sys.argv) > 1:
    version_str += "." + sys.argv[1]
open("VERSION", "w").write(version_str)


run(["git", "commit", "-a", "-m", '"' + version_str + '"'])
run(["rm", "-r", "dist"])
run(["python", "setup.py", "sdist"])

# Set the version and sha in the conda meta.yaml

package_path = os.path.join("dist", os.listdir("dist")[0])
raw_sha256 = subprocess.check_output(
    f"openssl sha256 {package_path} | sed 's/^.* //'", shell=True
)
sha256 = raw_sha256.decode("ascii").strip()
run(
    [
        f"sed -E -i '' 's/version = \"[0-9.]+/version = \"{version_str}/'"
        " conda.recipe/meta.yaml"
    ]
)
run([f"sed -E -i '' 's/sha256: [a-z0-9]+/sha256: {sha256}/' conda.recipe/meta.yaml"])

# Upload to pypi
run(["twine", "upload", "dist/*"])

# Commit and push to GitHub.
run(["git", "commit", "-a", "-m", '"' + version_str + '"'])
run(["git", "push"])

version_tag = f"v{version_str}"
run(["git", "tag", version_tag])
run(["git", "push", "origin", version_tag])
run(
    [
        "github-release",
        "release",
        "--user",
        "tbenthompson",
        "--repo",
        "cutde",
        "--tag",
        version_tag,
        "--name",
        version_tag,
        "--description",
        version_tag,
    ]
)
