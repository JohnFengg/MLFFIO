from setuptools import setup,find_packages

setup(
    name="mlffio",
    version="0.1",
    author='Jiangyuan John Feng',
    description='',
    requires=[],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mlffio = mlffio.cli:main"
        ]
    }
)