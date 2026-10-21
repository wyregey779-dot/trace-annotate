from setuptools import setup, find_packages

setup(
    name="trace-annotate",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "trace-annotate=trace_annotate.cli:main",
        ],
    },
)
