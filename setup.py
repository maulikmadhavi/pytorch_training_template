"""Main setup script."""

from setuptools import setup, find_packages

NAME = "ptt"
VERSION = "1.0.0"
DESCRIPTION = "PTT: pytorch_training_template"
AUTHOR = "Maulik Madhavi"
AUTHOR_EMAIL = "maulikmadhavi@gmail.com"

setup(
    name=NAME, version=VERSION, description=DESCRIPTION,
    author=AUTHOR, author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR, maintainer_email=AUTHOR_EMAIL,
    requires=["numpy"],
    install_requires=["numpy"],
    classifiers=[
        "Environment :: Console",
        "Topic :: System :: Clustering",
        "Intended Audience :: Science/Research"
    ],
    license="Apache-2.0",
    platforms=["all"],
    long_description_content_type="text/markdown",
    packages=find_packages()
)