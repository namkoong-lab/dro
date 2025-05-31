from setuptools import setup,find_packages
import os 

here = os.path.abspath(os.path.dirname(__file__))
md_path = os.path.join(here, 'src/README.md')

setup(
    name='dro',
    version='0.3.3',    
    license='MIT License',
    description='A package of distributionally robust optimization (DRO) methods. Implemented via cvxpy and PyTorch',
    long_description=open(md_path, encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/namkoong-lab/dro',
    author='DRO developers.',
    author_email='liujiashuo77@gmail.com, tw2837@columbia.edu',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=['pandas',
                      'numpy>=1.20',                     
                      'scikit-learn',
                      'torch', 
                      'scipy',
                      'cvxpy', 
                      'torchvision', 
                      'ucimlrepo', 
                      'matplotlib',
                      'torchattacks',
                      'mosek>=10.2.17',
                      'xgboost',
                      'lightgbm',

                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        'Operating System :: POSIX :: Linux',
    ],
    python_requires=">=3.7",
)