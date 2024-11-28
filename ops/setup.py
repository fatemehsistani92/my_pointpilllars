from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointpillars_ops',
    ext_modules=[
        CUDAExtension(
            name='voxel_op', 
            sources=[
                'voxelization/voxelization.cpp',
                'voxelization/voxelization_cpu.cpp',
                'voxelization/voxelization_cuda.cu',
            ],
            include_dirs=[
                '/home/hora/.local/lib/python3.8/site-packages/torch/include',
                '/home/hora/.local/lib/python3.8/site-packages/torch/include/torch/csrc/api/include',
                '/usr/local/cuda/include',
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': [
                    '-O3',
                    '-std=c++14',
                    '--expt-relaxed-constexpr',
                    '--compiler-options', '-fPIC',
                ],
            },
        ),
        CUDAExtension(
            name='iou3d_op', 
            sources=[
                'iou3d/iou3d.cpp',
                'iou3d/iou3d_kernel.cu',
            ],
            include_dirs=[
                '/home/hora/.local/lib/python3.8/site-packages/torch/include',
                '/home/hora/.local/lib/python3.8/site-packages/torch/include/torch/csrc/api/include',
                '/usr/local/cuda/include',
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': [
                    '-O3',
                    '-std=c++14',
                    '--expt-relaxed-constexpr',
                    '--compiler-options', '-fPIC',
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

