from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='st-trace',
    version='0.1.0',
    description='Neural Graph Search with Spatio-Temporal Priors for Efficient Multi-Camera Tracking',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='ST-Trace Authors',
    url='https://github.com/Listcapture/ST-trace',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Computer Vision',
    ],
    python_requires='>=3.8',
)
