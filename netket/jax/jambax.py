"""Call Numba from jitted JAX functions.

# The interface

To call your Numba function from JAX, you have to implement:

  1. A Numba function following our calling convention.
  2. A function for abstractly evaluating the function, i.e., for specifying
     the output shapes and dtypes from the input ones.

## 1. The Numba function

The Numba function has to accept a *single* tuple argument and do not return
anythin, i.e. have type `Callable[tuple[numba.carray], None]`. The output and
input arguments are stored consecutively in the tuple. For example, if you want
to implement a function that takes three arrays and returns two, the Numba
function should look like:

```py
@numba.jit
def add_and_mul(args):
  output_1, output_2, input_1, input_2, input_3 = args
  # Now edit output_1 and output_2 *in place*.
  output_1.fill(0)
  output_2.fill(0)
  output_1 += input_1 + input_2
  output_2 += input_1 * input_3
```

Note that the output arguments have to be modified *in-place*. These arrays are
allocated and owned by XLA.

## 2. The abstract evaluation function

You also have to implement a function that tells JAX how to compute the shapes
and types of the outputs from the inputs.For more information, please refer to

https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#Abstract-evaluation-rules

For example, for the above function, the corresponding abstract eval function is

```py
def add_and_mul_shape_fn(input_1, input_2, input_3):
  assert input_1.shape == input_2.shape
  assert input_1.shape == input_3.shape
  return (jax.abstract_arrays.ShapedArray(input_1.shape, input_1.dtype),
          jax.abstract_arrays.ShapedArray(input_1.shape, input_1.dtype))
```

# Conversion

Now, what is left is to convert the function:

```py
add_and_mul_jax = jax.experimental.jambax.numba_to_jax(
    "add_and_mul", add_and_mul, add_and_mul_shape_fn)
```

You can JIT compile the function as
```py
add_and_mul_jit = jax.jit(add_and_mul_jax)
```

# Optional
## Derivatives

You can define a gradient for your function as if you were definining a custom
gradient for any other JAX function. You can follow the tutorial at:

https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html

## Batching / vmap

Batching along the first axes is implemented via jax.lax.map. To implement your
own bathing rule, see the documentation of `numba_to_jax`.
"""

import collections
import ctypes
from functools import partial  # pylint:disable=g-importing-member

import jax
from jax.interpreters import batching
from jax.interpreters import xla
from jax.lib import xla_client
from jaxlib import xla_extension
import numba
from numba import types as nb_types
import numba.typed as nb_typed
import numpy as np


def _shape_size(shape):
    sz = 1
    for dim in shape:
        sz *= dim
    return sz


def _xla_shape_to_abstract(xla_shape):
    return jax.abstract_arrays.ShapedArray(
        xla_shape.dimensions(), xla_shape.element_type()
    )


def _create_xla_target_capsule(ptr):
    xla_capsule_magic = b"xla._CUSTOM_CALL_TARGET"
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    return ctypes.pythonapi.PyCapsule_New(ptr, xla_capsule_magic, None)


def _np_evaluation_rule(call_fn, abstract_eval_fn, *args, **kwargs):
    output_shapes = abstract_eval_fn(*args)
    outputs = tuple(np.empty(shape.shape, dtype=shape.dtype) for shape in output_shapes)
    inputs = tuple(np.asarray(arg) for arg in args)
    call_fn(outputs + inputs, **kwargs)
    return tuple(outputs)


def _naive_batching(call_fn, args, batch_axes):
    # TODO(josipd): Check that the axes are all zeros. Add support when only a
    #               subset of the arguments have to be batched.
    # TODO(josipd): Do this smarter than n CustomCalls.
    return tuple(jax.lax.map(lambda x: call_fn(*x), args)), batch_axes


def _xla_translation_cpu(numba_fn, abstract_eval_fn, xla_builder, *args):
    """Returns the XLA CustomCall for the given numba function.

    Args:
      numba_fn: A numba function. For its signature, see the module docstring.
      abstract_eval_fn: The abstract shape evaluation function.
      xla_builder: The XlaBuilder instance.
      *args: The positional arguments to be passed to `numba_fn`.
    Returns:
      The XLA CustomCall operation calling into the numba function.
    """
    print("encoding jamax cpu")
    input_shapes = [xla_builder.get_shape(arg) for arg in args]
    # TODO(josipd): Check that the input layout is the numpy default.
    output_abstract_arrays = abstract_eval_fn(
        *[_xla_shape_to_abstract(shape) for shape in input_shapes]
    )
    output_shapes = tuple(array.shape for array in output_abstract_arrays)
    output_shapes_flattened = tuple(
        dim for array in output_abstract_arrays for dim in array.shape
    )
    output_ndims = tuple(array.ndim for array in output_abstract_arrays)
    output_ndims_offsets = tuple(np.cumsum(np.concatenate([[0], output_ndims])))
    output_dtypes = tuple(array.dtype for array in output_abstract_arrays)
    layout_for_shape = lambda shape: range(len(shape) - 1, -1, -1)
    output_layouts = map(layout_for_shape, output_shapes)
    xla_output_shapes = [
        xla_client.Shape.array_shape(*arg)
        for arg in zip(output_dtypes, output_shapes, output_layouts)
    ]
    xla_output_shape = xla_client.Shape.tuple_shape(xla_output_shapes)

    input_dtypes = tuple(shape.element_type() for shape in input_shapes)
    input_dimensions = tuple(shape.dimensions() for shape in input_shapes)

    output_i = tuple(i for i in range(len(output_shapes)))
    input_i = tuple(i for i in range(len(input_dimensions)))

    n_out = len(output_shapes)
    n_in = len(input_dimensions)

    xla_call_sig = nb_types.void(
        nb_types.CPointer(nb_types.voidptr),  # output_ptrs
        nb_types.CPointer(nb_types.voidptr),  # input_ptrs
    )

    @numba.cfunc(xla_call_sig)
    def xla_custom_call_target(output_ptrs, input_ptrs):
        # manually unroll input and output args because numba is
        # relatively dummb and cannot always infer getitem on inhomogeneous tuples
        if n_out == 1:
            args_out = (
                numba.carray(output_ptrs[0], output_shapes[0], dtype=output_dtypes[0]),
            )
        elif n_out == 2:
            args_out = (
                numba.carray(output_ptrs[0], output_shapes[0], dtype=output_dtypes[0]),
                numba.carray(output_ptrs[1], output_shapes[1], dtype=output_dtypes[1]),
            )
        elif n_out == 3:
            args_out = (
                numba.carray(output_ptrs[0], output_shapes[0], dtype=output_dtypes[0]),
                numba.carray(output_ptrs[1], output_shapes[1], dtype=output_dtypes[1]),
                numba.carray(output_ptrs[2], output_shapes[2], dtype=output_dtypes[2]),
            )
        elif n_out == 4:
            args_out = (
                numba.carray(output_ptrs[0], output_shapes[0], dtype=output_dtypes[0]),
                numba.carray(output_ptrs[1], output_shapes[1], dtype=output_dtypes[1]),
                numba.carray(output_ptrs[2], output_shapes[2], dtype=output_dtypes[2]),
                numba.carray(output_ptrs[3], output_shapes[3], dtype=output_dtypes[3]),
            )

        if n_in == 1:
            args_in = (
                numba.carray(input_ptrs[0], input_dimensions[0], dtype=input_dtypes[0]),
            )
        elif n_in == 2:
            args_in = (
                numba.carray(input_ptrs[0], input_dimensions[0], dtype=input_dtypes[0]),
                numba.carray(input_ptrs[1], input_dimensions[1], dtype=input_dtypes[1]),
            )
        elif n_in == 3:
            args_in = (
                numba.carray(input_ptrs[0], input_dimensions[0], dtype=input_dtypes[0]),
                numba.carray(input_ptrs[1], input_dimensions[1], dtype=input_dtypes[1]),
                numba.carray(input_ptrs[2], input_dimensions[2], dtype=input_dtypes[2]),
            )
        elif n_in == 4:
            args_in = (
                numba.carray(input_ptrs[0], input_dimensions[0], dtype=input_dtypes[0]),
                numba.carray(input_ptrs[1], input_dimensions[1], dtype=input_dtypes[1]),
                numba.carray(input_ptrs[2], input_dimensions[2], dtype=input_dtypes[2]),
                numba.carray(input_ptrs[3], input_dimensions[3], dtype=input_dtypes[3]),
            )

        numba_fn(args_out + args_in)

    target_name = xla_custom_call_target.native_name.encode("ascii")
    capsule = _create_xla_target_capsule(xla_custom_call_target.address)
    xla_client.register_custom_call_target(target_name, capsule, "cpu")
    # xla_extension.register_custom_call_target(target_name, capsule, "Host")
    return xla_client.ops.CustomCallWithLayout(
        xla_builder,
        target_name,
        operands=args,
        shape_with_layout=xla_output_shape,
        operand_shapes_with_layout=input_shapes,
    )


##

try:
    import ctypes
    from numba.extending import get_cython_function_address
    import cupy_backends

    addr = get_cython_function_address("cupy_backends.cuda.api.runtime", "memcpy")
    functype = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    )
    cudaMemcpy = functype(addr)

    #
    addr = get_cython_function_address("cupy_backends.cuda.api.runtime", "memcpyAsync")
    functype = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    cudaMemcpyAsync = functype(addr)

    #
    addr = get_cython_function_address(
        "cupy_backends.cuda.api.runtime", "streamSynchronize"
    )
    functype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
    cudaStreamSynchronize = functype(addr)

    has_cupy = True

    def _xla_translation_gpu(numba_fn, abstract_eval_fn, xla_builder, *args):
        """Returns the XLA CustomCall for the given numba function.

        Args:
          numba_fn: A numba function. For its signature, see the module docstring.
          abstract_eval_fn: The abstract shape evaluation function.
          xla_builder: The XlaBuilder instance.
          *args: The positional arguments to be passed to `numba_fn`.
        Returns:
          The XLA CustomCall operation calling into the numba function.
        """
        print("encoding jambax gpu")
        input_shapes = [xla_builder.get_shape(arg) for arg in args]
        input_dtypes = tuple(shape.element_type() for shape in input_shapes)
        input_dimensions = tuple(shape.dimensions() for shape in input_shapes)
        input_byte_size = tuple(
            np.prod(shape) * dtype.itemsize
            for (shape, dtype) in zip(input_dimensions, input_dtypes)
        )

        input_i = tuple(i for i in range(len(input_dimensions)))
        n_in = len(input_dimensions)

        # TODO(josipd): Check that the input layout is the numpy default.
        output_abstract_arrays = abstract_eval_fn(
            *[_xla_shape_to_abstract(shape) for shape in input_shapes]
        )
        output_shapes = tuple(array.shape for array in output_abstract_arrays)

        output_ndims = tuple(array.ndim for array in output_abstract_arrays)
        output_ndims_offsets = tuple(np.cumsum(np.concatenate([[0], output_ndims])))
        output_dtypes = tuple(array.dtype for array in output_abstract_arrays)
        output_byte_size = tuple(
            np.prod(shape) * dtype.itemsize
            for (shape, dtype) in zip(output_shapes, output_dtypes)
        )

        layout_for_shape = lambda shape: range(len(shape) - 1, -1, -1)
        output_layouts = map(layout_for_shape, output_shapes)
        xla_output_shapes = [
            xla_client.Shape.array_shape(*arg)
            for arg in zip(output_dtypes, output_shapes, output_layouts)
        ]
        xla_output_shape = xla_client.Shape.tuple_shape(xla_output_shapes)

        output_i = tuple(i for i in range(len(output_shapes)))

        n_out = len(output_shapes)

        xla_call_sig = nb_types.void(
            nb_types.voidptr,  # cudaStream_t* stream
            nb_types.CPointer(nb_types.voidptr),  # void** buffers
            nb_types.voidptr,  # const char* opaque
            nb_types.uint64,  # size_t opaque_len
        )

        print(f"With N_in={n_in} and n_out={n_out}")

        @numba.cfunc(xla_call_sig)
        def xla_custom_call_target(stream, inout_gpu_ptrs, opaque, opaque_len):
            # manually unroll input and output args because numba is
            # relatively dummb and cannot always infer getitem on inhomogeneous tuples
            if n_out == 1:
                args_out = (np.empty(output_shapes[0], dtype=output_dtypes[0]),)
            elif n_out == 2:
                args_out = (
                    np.empty(output_shapes[0], dtype=output_dtypes[0]),
                    np.empty(output_shapes[1], dtype=output_dtypes[1]),
                )
            elif n_out == 3:
                args_out = (
                    np.empty(output_shapes[0], dtype=output_dtypes[0]),
                    np.empty(output_shapes[1], dtype=output_dtypes[1]),
                    np.empty(output_shapes[2], dtype=output_dtypes[2]),
                )
            elif n_out == 4:
                args_out = (
                    np.empty(output_shapes[0], dtype=output_dtypes[0]),
                    np.empty(output_shapes[1], dtype=output_dtypes[1]),
                    np.empty(output_shapes[2], dtype=output_dtypes[2]),
                    np.empty(output_shapes[3], dtype=output_dtypes[3]),
                )

            if n_in == 1:
                args_in = (np.empty(input_dimensions[0], dtype=input_dtypes[0]),)
            #    cudaMemcpy(args_in[0].ctypes.data, inout_gpu_ptrs[0], input_byte_size[0], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyDeviceToHost))
            elif n_in == 2:
                args_in = (
                    np.empty(input_dimensions[0], dtype=input_dtypes[0]),
                    np.empty(input_dimensions[1], dtype=input_dtypes[1]),
                )
                cudaMemcpy(
                    args_in[0].ctypes.data,
                    inout_gpu_ptrs[0],
                    1,
                    nb_types.int32(cupy_backends.cuda.api.runtime.memcpyDeviceToHost),
                )
                cudaMemcpy(
                    args_in[1].ctypes.data,
                    inout_gpu_ptrs[1],
                    1,
                    nb_types.int32(cupy_backends.cuda.api.runtime.memcpyDeviceToHost),
                )
            elif n_in == 3:
                args_in = (
                    np.empty(input_dimensions[0], dtype=input_dtypes[0]),
                    np.empty(input_dimensions[1], dtype=input_dtypes[1]),
                    np.empty(input_dimensions[2], dtype=input_dtypes[2]),
                )
            #    cudaMemcpy(args_in[0].ctypes.data, inout_gpu_ptrs[0], input_byte_size[0], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyDeviceToHost))
            #    cudaMemcpy(args_in[1].ctypes.data, inout_gpu_ptrs[1], input_byte_size[1], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyDeviceToHost))
            #    cudaMemcpy(args_in[2].ctypes.data, inout_gpu_ptrs[2], input_byte_size[2], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyDeviceToHost))
            elif n_in == 4:
                args_in = (
                    np.empty(input_dimensions[0], dtype=input_dtypes[0]),
                    np.empty(input_dimensions[1], dtype=input_dtypes[1]),
                    np.empty(input_dimensions[2], dtype=input_dtypes[2]),
                    np.empty(input_dimensions[3], dtype=input_dtypes[3]),
                )
            #    cudaMemcpy(args_in[0].ctypes.data, inout_gpu_ptrs[0], input_byte_size[0], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyDeviceToHost))
            #    cudaMemcpy(args_in[1].ctypes.data, inout_gpu_ptrs[1], input_byte_size[1], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyDeviceToHost))
            #    cudaMemcpy(args_in[2].ctypes.data, inout_gpu_ptrs[2], input_byte_size[2], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyDeviceToHost))
            #    cudaMemcpy(args_in[3].ctypes.data, inout_gpu_ptrs[3], input_byte_size[3], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyDeviceToHost))

            # numba_fn(args_out + args_in)

            # if n_out == 1:
            #    cudaMemcpy(inout_gpu_ptrs[n_in+0], args_out[0].ctypes.data, output_byte_size[0], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyHostToDevice))
            # elif n_out == 2:
            #    cudaMemcpy(inout_gpu_ptrs[n_in+0], args_out[0].ctypes.data, output_byte_size[0], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyHostToDevice))
            #    cudaMemcpy(inout_gpu_ptrs[n_in+1], args_out[1].ctypes.data, output_byte_size[1], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyHostToDevice))
            # elif n_out == 3:
            #    cudaMemcpy(inout_gpu_ptrs[n_in+0], args_out[0].ctypes.data, output_byte_size[0], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyHostToDevice))
            #    cudaMemcpy(inout_gpu_ptrs[n_in+1], args_out[1].ctypes.data, output_byte_size[1], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyHostToDevice))
            #    cudaMemcpy(inout_gpu_ptrs[n_in+2], args_out[2].ctypes.data, output_byte_size[2], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyHostToDevice))
            # elif n_out == 4:
            #    cudaMemcpy(inout_gpu_ptrs[n_in+0], args_out[0].ctypes.data, output_byte_size[0], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyHostToDevice))
            #    cudaMemcpy(inout_gpu_ptrs[n_in+1], args_out[1].ctypes.data, output_byte_size[1], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyHostToDevice))
            #    cudaMemcpy(inout_gpu_ptrs[n_in+2], args_out[2].ctypes.data, output_byte_size[2], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyHostToDevice))
            #    cudaMemcpy(inout_gpu_ptrs[n_in+3], args_out[3].ctypes.data, output_byte_size[3], nb_types.int32(cupy_backends.cuda.api.runtime.memcpyHostToDevice))

        target_name = xla_custom_call_target.native_name.encode("ascii")
        capsule = _create_xla_target_capsule(xla_custom_call_target.address)
        xla_client.register_custom_call_target(target_name, capsule, "gpu")
        return xla_client.ops.CustomCallWithLayout(
            xla_builder,
            target_name,
            operands=args,
            shape_with_layout=xla_output_shape,
            operand_shapes_with_layout=input_shapes,
        )


except:
    has_cupy = False


def numba_to_jax(name: str, numba_fn, abstract_eval_fn, batching_fn=None):
    """Create a jittable JAX function for the given Numba function.

    Args:
      name: The name under which the primitive will be registered.
      numba_fn: The function that can be compiled with Numba.
      abstract_eval_fn: The abstract evaluation function.
      batching_fn: If set, this function will be used when vmap-ing the returned
        function.
    Returns:
      A jitable JAX function.
    """
    primitive = jax.core.Primitive(name)
    primitive.multiple_results = True

    def abstract_eval_fn_always(*args, **kwargs):
        # Special-casing when only a single tensor is returned.
        shapes = abstract_eval_fn(*args, **kwargs)
        if not isinstance(shapes, collections.abc.Collection):
            return [shapes]
        else:
            return shapes

    primitive.def_abstract_eval(abstract_eval_fn_always)
    primitive.def_impl(partial(_np_evaluation_rule, numba_fn, abstract_eval_fn_always))

    def _primitive_bind(*args):
        result = primitive.bind(*args)
        output_shapes = abstract_eval_fn(*args)
        # Special-casing when only a single tensor is returned.
        if not isinstance(output_shapes, collections.abc.Collection):
            assert len(result) == 1
            return result[0]
        else:
            return result

    if batching_fn is not None:
        batching.primitive_batchers[primitive] = batching_fn
    else:
        batching.primitive_batchers[primitive] = partial(
            _naive_batching, _primitive_bind
        )
    xla.backend_specific_translations["cpu"][primitive] = partial(
        _xla_translation_cpu, numba_fn, abstract_eval_fn_always
    )

    if has_cupy:
        xla.backend_specific_translations["gpu"][primitive] = partial(
            _xla_translation_gpu, numba_fn, abstract_eval_fn_always
        )

    return _primitive_bind


def njit4jax(output_shapes):

    abstract_eval = lambda *args: output_shapes

    def decorator(fun):
        jitted_fun = numba.njit(fun)
        fn_name = "numba_fun_{}".format(hash(jitted_fun))
        print("jambax:", fn_name)
        return numba_to_jax(fn_name, jitted_fun, abstract_eval)

    return decorator
