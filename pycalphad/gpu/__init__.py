from . import gpuify, calculate_gpu
gpuify.parent_globals = globals()
calculate_gpu.parent_globals = globals()
from .gpuify import *
from .calculate_gpu import compute_phase_values_gpu
