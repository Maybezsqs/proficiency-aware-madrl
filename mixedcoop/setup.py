from setuptools import setup, find_packages

setup(name='mixdrl',
      version='0.0.1',
      description='Proficiency Aware Multi-Agent Actor-Critic for Mixed Robot Teaming',
      url='https://github.com/florayym/proficiency-aware-madrl',
      author='Qifei Yu',
      author_email='yuqifei.hfut@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
