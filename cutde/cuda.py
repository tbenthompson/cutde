import logging

import pycuda
import pycuda.compiler
import pycuda.gpuarray

logger = logging.getLogger(__name__)

cuda_initialized = False


def ensure_initialized():
    global cuda_initialized
    if not cuda_initialized:
        cuda_initialized = True
        import pycuda.autoinit

        gpu_idx = pycuda.driver.Context.get_device().get_attribute(
            pycuda._driver.device_attribute.MULTI_GPU_BOARD_GROUP_ID
        )
        logger.info("Initialized CUDA on gpu: " + str(gpu_idx))


def ptr(arr):
    if type(arr) is pycuda.gpuarray.GPUArray:
        return arr.gpudata
    return arr


def to_gpu(arr, float_type):
    ensure_initialized()
    if type(arr) is pycuda.gpuarray.GPUArray:
        return arr
    to_type = arr.astype(float_type)
    return pycuda.gpuarray.to_gpu(to_type)


def empty_gpu(shape, float_type):
    ensure_initialized()
    return pycuda.gpuarray.empty(shape, float_type)


def zeros_gpu(shape, float_type):
    ensure_initialized()
    return pycuda.gpuarray.zeros(shape, float_type)


class CUDAContextWrapper(object):
    def __init__(self, context):
        self.ctx = context

    def __enter__(self):
        self.ctx.push()
        return self

    def __exit__(self, evalue, etype, etraceback):
        self.ctx.pop()


def threaded_get(arr):
    import pycuda.autoinit

    with CUDAContextWrapper(pycuda.autoinit.context):
        return arr.get()


class ModuleWrapper:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        kernel = self.module.get_function(name)

        def wrapper(*args, **kwargs):
            arg_ptrs = [ptr(a) for a in args]
            return kernel(*arg_ptrs, **kwargs)

        return wrapper


def compile(code):
    ensure_initialized()
    compiler_args = ["--use_fast_math", "--restrict"]
    return ModuleWrapper(pycuda.compiler.SourceModule(code, options=compiler_args))


cluda_preamble = """
#include <stdio.h>
#define CUDA
// taken from pycuda._cluda
#define LOCAL_BARRIER __syncthreads()
#define WITHIN_KERNEL __device__
#define KERNEL extern "C" __global__
#define GLOBAL_MEM /* empty */
#define LOCAL_MEM __shared__
#define LOCAL_MEM_DYNAMIC extern __shared__
#define LOCAL_MEM_ARG /* empty */
#define CONSTANT __constant__
#define INLINE __forceinline__
#define SIZE_T unsigned int
#define VSIZE_T unsigned int
// used to align fields in structures
#define ALIGN(bytes) __align__(bytes)

//IF EVER NEEDED, THESE CAN BE CONVERTED TO DEFINES SO THAT THERE IS LESS PER
// KERNEL OVERHEAD.
WITHIN_KERNEL SIZE_T get_local_id(unsigned int dim)
{
    if(dim == 0) return threadIdx.x;
    if(dim == 1) return threadIdx.y;
    if(dim == 2) return threadIdx.z;
    return 0;
}
WITHIN_KERNEL SIZE_T get_group_id(unsigned int dim)
{
    if(dim == 0) return blockIdx.x;
    if(dim == 1) return blockIdx.y;
    if(dim == 2) return blockIdx.z;
    return 0;
}
WITHIN_KERNEL SIZE_T get_local_size(unsigned int dim)
{
    if(dim == 0) return blockDim.x;
    if(dim == 1) return blockDim.y;
    if(dim == 2) return blockDim.z;
    return 1;
}
WITHIN_KERNEL SIZE_T get_num_groups(unsigned int dim)
{
    if(dim == 0) return gridDim.x;
    if(dim == 1) return gridDim.y;
    if(dim == 2) return gridDim.z;
    return 1;
}
WITHIN_KERNEL SIZE_T get_global_size(unsigned int dim)
{
    return get_num_groups(dim) * get_local_size(dim);
}
WITHIN_KERNEL SIZE_T get_global_id(unsigned int dim)
{
    return get_local_id(dim) + get_group_id(dim) * get_local_size(dim);
}
"""
