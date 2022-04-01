from setuptools import find_packages, setup
from telco_churn import __version__

setup(
    name="telco_churn",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="",
    author=""
)
