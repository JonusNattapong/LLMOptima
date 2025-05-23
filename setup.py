from setuptools import setup, find_packages

setup(
    name="llmoptima",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.26.0",
        "numpy>=1.20.0",
        "tqdm>=4.64.0"
    ],
    author="JonusNattapong",
    author_email="info@llmoptima.ai",
    description="Next-generation LLM Optimization Engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JonusNattapong/LLMOptima",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
