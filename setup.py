"""Installation script for the 'omni_custom_gym' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os
import sys

root_dir = os.path.dirname(os.path.realpath(__file__))

# Modify the version name based on Python version
python_version = ".".join(map(str, sys.version_info[:2]))  # Get major.minor Python version
package_name = 'omni_custom_gym'

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "torch==1.12.0",
    "gym==0.26.2",
    "gymnasium==0.28.1", 
    "stable_baselines3[extra]==2.0.0a10", 
    "box2d-py",
    "tensorboard",
    "tensorboard-plugin-wit",
    "termcolor",
    "rospkg",
    ]

# Installation operation
setup(
    name=package_name,
    author="AndPatr",
    version="1.0.0_py" + python_version,
    description="",
    keywords=["gym", "omniverse", "RL"],
    include_package_data=True,
    python_requires=">=3.7, <3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.6, 3.7, 3.8"],
    zip_safe=False,
)

# EOF
