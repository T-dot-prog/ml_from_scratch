from setuptools import setup, find_packages

setup(
    name='machine_learning_from_scratch',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        # e.g., 'numpy>=1.18.0',
        # 'pandas>=1.0.0',
        'kagglehub',
    ],
    author='Tahasin islam',
    author_email='tahasinahoni2@gmail.com',
    description='A machine learning library built from scratch',
    url='https://github.com/T-dot-prog/ml_from_scratch',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
