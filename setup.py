from distutils.core import setup

setup(
    name='ACrab',
    version='0.1.0',
    author='Hanlin Zhu',
    author_email='hanlinzhu@berkeley.edu',
    packages=['ACrab'],
    url='https://github.com/zhuhl98/ACrab.git',
    license='MIT LICENSE',
    description='Code for the ACrab algorithm',
    long_description=open('README.md').read(),
    install_requires=[
        "gym==0.17.2",
        "torchaudio==2.1.0",
        "torch==2.1.0",
        "tensorboard==2.10.0",
        "psutil==5.9.1",
        "protobuf==3.19.4",
        "dowel==0.0.4",
        "d4rl@git+https://github.com/rail-berkeley/d4rl@6330b4e09e36a80f4b706a3885d59d97745c05a9"
    ]
)
