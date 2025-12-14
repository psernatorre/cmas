from setuptools import setup, find_packages

setup(
        name='cmaspy',
        version='0.1.0',
        author='cmaspy developers',
        author_email='your.email@example.com',
        description='Control of multi-agent systems',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        packages=find_packages(),
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: MacOS :: MacOS X',
            'Topic :: Scientific/Engineering',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        python_requires='>=3.12',
        install_requires=[
            "pandas",
            "numpy",
            "scipy",
            "tabulate", 
	        "cvxpy"
        ],
    )
