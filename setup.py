from setuptools import setup, find_packages

setup(
    name="ml_agents",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.19.0',
        'scikit-learn>=0.24.0',
        'joblib>=1.0.0'
    ],
    author="Neeraj Chavan",
    author_email="ncneerajchavan@gmail.com",
    description="A collection of ML agents for various data science tasks",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neerajchavan/ml-agents",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
