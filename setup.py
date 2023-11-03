import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pml", # Replace with your own username
    version="0.0.1",
    author="Manuel Baum",
    author_email="baum@tu-berlin.de",
    description="Probabilistic Machine Learning in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manuelbaum/pml",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=["torch"]
)
