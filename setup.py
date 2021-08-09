from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="skku-shinhan-bank/koelectra_classfication",
    version="0.0.1",
    author="SKKU Shinhan Bank",
    author_email="ajtwlswjddnv1102@gmail.com",
    description="koelectra classfication",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skku-shinhan-bank/koelectra_classification",
    packages=find_packages(),
    install_requires=requirements,
)
