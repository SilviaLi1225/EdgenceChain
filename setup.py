from setuptools import setup,find_packages
setup(
    name='edgenceChain',
    version='0.0.1',
    url='https://github.com/EdgeIntelligenceChain/EdgenceChain',
    install_requires=['base58>=0.2.5','ecdsa>=0.13','docopt==0.6.2'],
    python_requires='>=3.6.7',
    packages=find_packages()
  )