import subprocess
import os, sys
from typing import Any
import pkg_resources
from packaging import version as pv

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
req_file = os.path.join(BASE_PATH, "requirements.txt")

def pip_install(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args])

def pip_uninstall(*args):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *args])

def is_installed(
        package: str, version: str | None = None, strict: bool = True
):
    try:
        has_package = pkg_resources.get_distribution(package)
        if has_package is not None:
            installed_version = has_package.version
            if (installed_version != version and strict) or (pv.parse(installed_version) < pv.parse(version) and not strict):
                return False
            else:
                return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if not is_installed("dlib"):
    with open(req_file) as file:
        install_count = 0
        for package in file:
            package_version = None
            strict = True
            try:
                package = package.strip()
                if "==" in package:
                    package_version = package.split('==')[1]
                elif ">=" in package:
                    package_version = package.split('>=')[1]
                    strict = False
                if not is_installed(package, package_version, strict):
                    install_count += 1
                    pip_install(package)
            except Exception as e:
                print(e)
                print(f"\nERROR: Failed to install {package}")
                raise e
