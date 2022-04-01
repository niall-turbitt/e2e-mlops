from setuptools import find_packages, setup
from telco_churn import __version__

with open("requirements.txt") as f:
    reqs = f.read()

setup(
    name="telco_churn",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    install_requires=reqs,
    version=__version__,
    # entry_points={
    #     "console_scripts": [
    #         "feature_table_creation = telco_churn.jobs.feature_table_creator:main"
    #     ]
    # },
    description="",
    author=""
)
