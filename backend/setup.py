"""Setup configuration for scholar-agent backend."""
from setuptools import find_packages, setup

setup(
    name="scholar-agent",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn>=0.29.0",
        "celery[redis]>=5.4.0",
        "sqlalchemy>=2.0.29",
        "langgraph>=0.2.0",
        "langchain-core>=0.3.0",
    ],
)
