from setuptools import setup

setup(
    name='geotransformer',
    version='1.0.0+review',
    install_requires=[
        'torch==1.7.1',
        'numpy==1.19.2',
        'scipy==1.5.2',
        'matplotlib==3.3.2',
        'ipython==7.29.0',
        'tqdm==4.50.2',
        'coloredlogs==15.0',
        'easydict==1.9',
        'nibabel==3.2.1',
        'open3d==0.11.2',
    ]
)
