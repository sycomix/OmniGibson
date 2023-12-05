import importlib
import os
import pkgutil
import shutil
from string import Template

import omnigibson
from omnigibson import examples
from omnigibson.utils.asset_utils import download_assets

download_assets()


def main():
    examples_list = [
        package.name[17:]
        for package in pkgutil.walk_packages(
            examples.__path__, f"{examples.__name__}."
        )
        if (
            not package.ispkg
            and package.name[17:] != "example_selector"
            and "web_ui"
            not in package.name[
                17:
            ]  # The WebUI examples require additional server setup
            and "vr_"
            not in package.name[
                17:
            ]  # The VR examples require additional dependencies
            and "ray_"
            not in package.name[
                17:
            ]  # The Ray/RLLib example does not run in a subprocess
        )
    ]
    temp_folder_of_test = os.path.join("/", "tmp", "tests_of_examples")
    shutil.rmtree(temp_folder_of_test, ignore_errors=True)
    os.makedirs(temp_folder_of_test, exist_ok=True)

    for example in examples_list:
        template_file_name = os.path.join(omnigibson.__path__[0], "..", "tests", "test_of_example_template.txt")
        with open(template_file_name, "r") as f:
            substitutes = dict()
            substitutes["module"] = example
            name = example.rsplit(".", 1)[-1]
            substitutes["name"] = name
            src = Template(f.read())
            dst = src.substitute(substitutes)
            with open(os.path.join(temp_folder_of_test, f"{name}_test.py"), "w") as test_file:
                n = test_file.write(dst)


if __name__ == "__main__":
    main()
