from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="saucedo-quantitative-trading-engine",
    version="1.0.0",
    author="Joel Saucedo",
    author_email="joel.saucedo@example.com",
    description="A comprehensive backtesting and strategy analysis framework for quantitative trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joelsaucedo/saucedo-quantitative-trading-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "notebook>=6.4.0",
        ],
        "optimization": [
            "numba>=0.56.0",
            "PyPortfolioOpt>=1.5.0",
            "cvxpy>=1.2.0",
        ],
        "market-data": [
            "yfinance>=0.1.87",
            "pandas-market-calendars>=4.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    entry_points={
        "console_scripts": [
            "strategy-analyzer=src.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/trading-strategy-analyzer/issues",
        "Source": "https://github.com/yourusername/trading-strategy-analyzer",
        "Documentation": "https://trading-strategy-analyzer.readthedocs.io/",
    },
)
