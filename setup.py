from distutils.core import setup

setup(name='pytexes',
      version='0.1.0',
      description='TEXES reduction library',
      author='Klaus Pontoppidan',
      author_email='pontoppi@stsci.edu',
      url='http://www.stsci.edu/~pontoppi',
      packages=['pytexes','utils'],
      package_data={'pytexes': ['*.ini','*.fits']}
      )

    
