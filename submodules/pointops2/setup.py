import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src = 'src'
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith('.cpp') or file.endswith('.cu')
]

setup(
    name='pointops2',
    version='1.0',
    install_requires=["torch", "numpy"],
    packages=["pointops2"],  # Now correctly finds the package
    package_dir={"pointops2": "pointops2"},  # Point directly to renamed folder
    ext_modules=[
        CUDAExtension(
            name='pointops2_cuda',
            sources=sources,
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)