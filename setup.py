import subprocess
from setuptools import setup

with open('requirements.txt', encoding="utf-16") as f:
    requirements = f.read().splitlines()

python_magic = [line for line in requirements if 'python-magic' in line]
if(len(python_magic) != 1):
    for p in python_magic:
        if("bin" in p):
            python_magic.remove(p)
    if(len(python_magic) != 1):
        raise Exception(f'python-magic requirement not found in requirements.txt or found multiple times: \'{python_magic}\'')
python_magic = python_magic[0]
python_magic_version = python_magic.split('==')[1]
if(len(python_magic_version) != 0):
    subprocess.call(['pip', 'install', f'python-magic=={python_magic_version}'])
else:
    subprocess.call(['pip', 'install', 'python-magic'])

python_magic_bin = [line for line in requirements if 'python-magic-bin' in line]
if(len(python_magic_bin) != 1):
    raise Exception(f'python-magic-bin requirement not found in requirements.txt or found multiple times: \'{python_magic_bin}\'')
python_magic_bin = python_magic_bin[0]
python_magic_bin_version = python_magic_bin.split('==')[1]
if(len(python_magic_bin_version) != 0):
    subprocess.call(['pip', 'uninstall', f'python-magic-bin=={python_magic_bin_version}'])
    subprocess.call(['pip', 'install', f'python-magic-bin=={python_magic_bin_version}'])
else:
    subprocess.call(['pip', 'uninstall', 'python-magic-bin'])
    subprocess.call(['pip', 'install', 'python-magic-bin'])

requirements.remove(python_magic)
requirements.remove(python_magic_bin)

setup(
    name='DataToolkit',
    url='https://github.com/coolneighbors/DataToolkit.git',
    author='Aaron Meisner',
    author_email='aaron.meisner@noirlab.edu',
    packages=['DataToolkit'],
    install_requires=requirements,
    version='1.0',
    license='MIT',
    description='The Cool Neighbors data analysis GitHub repository.',
    long_description=open('README.md').read(),
)

# This fixes an import bug with the python-magic and python-magic-bin packages.
# https://github.com/zooniverse/panoptes-python-client/issues/264
python_magic_bin = [line for line in requirements if 'python-magic-bin' in line]
if(len(python_magic_bin) != 1):
    raise Exception('python-magic-bin requirement not found in requirements.txt or found multiple times.')
python_magic_bin = python_magic_bin[0]
python_magic_bin_version = python_magic_bin.split('==')[1]
subprocess.call(['pip', 'uninstall', f'python-magic-bin=={python_magic_bin_version}'])
subprocess.call(['pip', 'install', f'python-magic-bin=={python_magic_bin_version}'])

