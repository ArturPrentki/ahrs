from setuptools import find_packages, setup

package_name = 'walk_pattern_generator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pos_test_trajectory = walk_pattern_generator.testing_trajectory_generator:main',
            'vel_test_trajectory = walk_pattern_generator.test_velocity_trajectory_generator:main',
            'inference = walk_pattern_generator.trajectory_inference:main',
            'raibert = walk_pattern_generator.raibert_trajectory_generator:main',
            'circle = walk_pattern_generator.circular_trajectory:main'
        ],
    },
)
