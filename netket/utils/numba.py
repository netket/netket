# Workaround to NUMBA#5111
# https://github.com/numba/numba/pull/5111
# bug arising when using in a jit compiled function a constant gloabal
# numba does not know how to lower it.

# Below is our attempt at lowering constant jitclasses. It is based on the unboxing code from jitclass/boxing.
# It comes with (at least) two wrinkles

# 1 : use of add_dynamic_addr breaks caching
# (we netket do not care about 1)
# 2 : the lowering code does not aquire a reference on pyval, only when the native result is assigned the meminfo gets inrefed (which will keep pyval alive for the lifetime of the memnfo). HOWEVER there is a chance that pyval will get collected between leaving _lower_constant_class_instance and use of the native value
# this might be issue


from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.imputils import lower_constant
from numba.experimental.jitclass import _box


@lower_constant(types.ClassInstanceType)
def _lower_constant_class_instance(context, builder, typ, pyval):
    def access_member(obj, member_offset):
        # Access member by byte offset
        offset = context.get_constant(types.uintp, member_offset)
        llvoidptr = ir.IntType(8).as_pointer()
        ptr = cgutils.pointer_add(builder, obj, offset)
        casted = builder.bitcast(ptr, llvoidptr.as_pointer())
        return builder.load(casted)

    struct_cls = cgutils.create_struct_proxy(typ)
    inst = struct_cls(context, builder)

    # get a pointer to pyval
    obj = context.add_dynamic_addr(builder, id(pyval), "")

    # load from Python object
    ptr_meminfo = access_member(obj, _box.box_meminfoptr_offset)
    ptr_dataptr = access_member(obj, _box.box_dataptr_offset)

    # store to native structure
    inst.meminfo = builder.bitcast(ptr_meminfo, inst.meminfo.type)
    inst.data = builder.bitcast(ptr_dataptr, inst.data.type)

    return inst._getvalue()
