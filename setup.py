from setuptools import setup, find_packages

setup(
    name="whisper-server",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "optimum",
        "accelerate",
        "fastapi",
        "uvicorn",
        "python-multipart",
    ],
)
