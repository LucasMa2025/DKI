"""
DKI - Dynamic KV Injection
Attention-Level User Memory Plugin for LLMs

Setup script for package installation

Version History:
- 2.0.0: Config-driven adapters, hybrid injection
- 2.5.0: Memory Trigger, Reference Resolver, Redis distributed cache
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dki",
    version="2.5.0",
    author="Lucas Ma",
    author_email="lucas_ma2025@126.com",
    description="DKI - Attention-Level User Memory Plugin for LLMs via K/V Injection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LucasMa2025/DKI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.18.0",
        ],
        "postgresql": [
            "asyncpg>=0.29.0",
        ],
        "mysql": [
            "aiomysql>=0.2.0",
        ],
        # Redis is now included in base requirements (important for production)
        # This extra is kept for backwards compatibility
        "redis": [
            "redis>=5.0.0",
        ],
        # Full production setup
        "production": [
            "redis>=5.0.0",
            "asyncpg>=0.29.0",
            "uvloop>=0.19.0",  # Faster event loop
        ],
    },
    entry_points={
        "console_scripts": [
            "dki-server=dki.web.app:run_server",
            "dki-experiment=dki.experiment.runner:main",
            "dki-generate-data=dki.experiment.data_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dki": [
            "config/*.yaml",
            "scripts/*.sql",
        ],
    },
)
