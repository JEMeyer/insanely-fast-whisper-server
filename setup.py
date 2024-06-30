from setuptools import setup, find_packages

setup(
    name="insanely-fast-whisper-server",
    version="1.0",
    packages=find_packages(),
    install_requires=["transformers", "optimum", "accelerate", "fastapi", "uvicorn"],
)
