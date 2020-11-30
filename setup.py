import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="splearn",
    version="0.1a1",
    author="Jingles",
    author_email="jinglescode@gmail.com",
    description="splearn: package for signal processing and machine learning with Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jinglescode/python-signal-processing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
