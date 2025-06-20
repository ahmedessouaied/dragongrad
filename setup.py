from setuptools import setup, find_packages

setup(
    name='dragongrad',
    version='0.1.0',
    description='A tiny reverse-mode autodiff engine with visualization',
    author='Ahmed Essouaied',
    packages=find_packages(),
    install_requires=[
        'matplotlib'
    ],
    python_requires='>=3.6',
)
