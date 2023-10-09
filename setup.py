from setuptools import setup, Extension
import subprocess

try:
    import cpuinfo
except ImportError:
    subprocess.call(['pip', 'install', 'py-cpuinfo'])
    import cpuinfo

simd_flags_all = [
    'mmx', 'sse', 'sse2', 'sse3', 'ssse3', 'sse4_1', 'sse4_2',
    'avx', 'avx2', 'avx512', 'fma', 'fma4', 'neon', 'asimd', 'vp'
]

simd_flags_supported = [
    'sse','sse2','sse3','ssse3','sse4_1','sse4_2','fma','fma4'
][::-1]

simd_flags = cpuinfo.get_cpu_info()
simd_version = [x for x in simd_flags_supported if x in simd_flags['flags']]

compile_options = {
    "sse4_1":"-msse4.1",
    "fma": "-mfma"
}

module = Extension('neuromorph',sources=['NeuroMorph.c', 'hashmap.c'], extra_compile_args=['-lpthread','-lm']+[option for version, option in compile_options.items() if version in simd_version],define_macros=[(f"nm_{token}", "1") for token in simd_version])

setup(
    name="NeuroMorph",
    version='1.0',
    description='Graphical neural network framework',
    ext_modules=[module]
)


