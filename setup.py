from setuptools import setup, find_packages

setup(
    name="synth",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "": ["py.typed"],
    },
    install_requires=[
        "pyfluidsynth>=1.3.4",
        "python-osc>=1.8.1",
        "mido>=1.3.0",
    ],
    python_requires=">=3.8",
)
