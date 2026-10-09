import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

wbc_src_dir = "/gardo_ws/shared/whole_body_controller"
this_src_dir = "/gardo_ws/rl_src/wbc_ext/"
isystem_flags = []
system_includes = [
    '/usr/include/eigen3',       
    '/opt/ros/jazzy/include',      
    '/usr/local/include/osqp',
    '/usr/local/include/OsqpEigen',
    '/usr/include/urdfdom', 
]

for inc_path in system_includes:
    isystem_flags.extend(['-isystem', inc_path])
setup(
    name='wbc_ext',
    ext_modules=[
        CppExtension(
            name='wbc_ext',
            sources=[
                os.path.join(this_src_dir, 'batched_wbc.cpp'),
                # os.path.join(this_src_dir, 'bindings.cpp'),
                os.path.join(wbc_src_dir, 'src/quad_prog.cpp'),
                os.path.join(wbc_src_dir, 'src/wbc_math.cpp'),
            ],
            define_macros=[
                ('BOOST_MPL_LIMIT_LIST_SIZE', '30'),
                ('BOOST_MPL_LIMIT_VECTOR_SIZE', '30'),
                ('COAL_HAS_OCTOMAP', None),
                ('COAL_HAVE_OCTOMAP', None),
                ('DEFAULT_RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
                ('FASTCDR_DYN_LINK', None),
                ('FMT_SHARED', None),
                ('OCTOMAP_MAJOR_VERSION', '1'),
                ('OCTOMAP_MINOR_VERSION', '9'),
                ('OCTOMAP_PATCH_VERSION', '7'),
                ('PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION', None),
                ('PINOCCHIO_WITH_COLLISION', None),
                ('PINOCCHIO_WITH_HPP_FCL', None),
                ('PINOCCHIO_WITH_URDFDOM', None),
                ('TINYXML2_IMPORT', None),
                ('_FILE_OFFSET_BITS', '64'),
                ('BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS', None),
                ('BOOST_FUSION_INVOKE_MAX_ARITY',12),
            ],
            include_dirs=[
                os.path.join(wbc_src_dir, 'include'),
                '/usr/include/eigen3',       
                '/opt/ros/jazzy/include',      # APT Pinocchio headers
                '/usr/local/include/osqp',     # OSQP headers
                '/usr/local/include/OsqpEigen',
                '/usr/include/urdfdom',        # URDF headers
            ],
            library_dirs=[
                '/opt/ros/jazzy/lib',
                '/usr/local/lib',
                '/opt/ros/jazzy/lib/x86_64-linux-gnu',
                '/usr/lib/x86_64-linux-gnu'
            ],
            libraries=[
                'osqp', 
                'OsqpEigen', 
                'pinocchio_default',
                'pinocchio_parsers',
                'pinocchio_collision',
                'urdfdom_model', # <--- Add URDF dependencies
                'urdfdom_world', 
                'urdfdom_sensor', 
                'console_bridge',
                'boost_system',  # <--- Pinocchio often needs this
                'boost_filesystem',
                'coal',
                'octomap',
                'octomath'
            ],
            extra_compile_args=[
                '-fopenmp',
                '-O3', 
                '-std=c++17',
                *isystem_flags,                       # Suppress warnings inside third-party headers
                '-Wno-unused-but-set-variable',      # Suppress Eigen Tensor Trace warnings
                '-Wno-unused-variable',              # Suppress general unused variable warnings
                '-Wno-deprecated-declarations',      # Suppress Boost/C++ deprecation warnings
                ],
            extra_link_args=['-fopenmp']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)