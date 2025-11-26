from setuptools import setup, find_packages

setup(
    name="rvlm",
    version="0.0.1",
    description="",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "google-genai",
        "iisignature",
    ],
)
