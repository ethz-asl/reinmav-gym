from setuptools import setup

extras={
    'mujoco':['mujoco_py>=1.50', 'imageio'],
}

# dependency
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

setup(name='gym_reinmav',
      version='0.0.1',
      url='https://github.com/ethz-asl/reinmav-gym',
      install_requires=[
          'gym',
          'vpython',
          'pyquaternion',
          'matplotlib',
      ],
      extras_require=extras,
      )
