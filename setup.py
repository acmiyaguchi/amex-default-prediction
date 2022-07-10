import setuptools

setuptools.setup(
    name="amex_default_prediction",
    version="0.2.0",
    description="Utilities for amex default prediction challenge",
    author="Anthony Miyaguchi",
    author_email="acmiyaguchi@gmail.com",
    url="https://github.com/acmiyaguchi/amex-default-prediction",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "click",
        "tqdm",
        "pyarrow",
        "pyspark",
        "torch",
        "pytorch-lightning",
        "torch-summary",
        'importlib-metadata>=0.12;python_version<"3.8"',
    ],
)
