from setuptools import setup, find_packages

setup(
    name='gridworld',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium>=0.26.0',
        'numpy',
        'matplotlib',
    ],
)