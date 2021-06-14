import datetime
import os
import sys

assert "GITHUB_TOKEN" in os.environ

d = datetime.datetime.now()
version_str = d.strftime("%y.%m.%d")
if len(sys.argv) > 1:
    version_str += "." + sys.argv[1]
open("VERSION", "w").write(version_str)


def run(cmd):
    print(" ".join(cmd))
    os.system(" ".join(cmd))


run(["git", "commit", "-a", "-m", '"' + version_str + '"'])
run(["rm", "-r", "dist"])
run(["python", "setup.py", "sdist"])
run(["twine", "upload", "dist/*"])
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
