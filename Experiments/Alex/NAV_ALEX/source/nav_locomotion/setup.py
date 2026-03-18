"""Setup for nav_locomotion Isaac Lab extension package."""

from setuptools import setup, find_packages

setup(
    name="nav_locomotion",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
    ],
    extras_require={
        "coach": ["anthropic"],
    },
)
