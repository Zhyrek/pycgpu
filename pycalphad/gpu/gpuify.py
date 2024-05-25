import numpy as np
import cupy as cp
import sympy as sp
import pathlib
import inspect
from cupyx import jit
import linecache
from cupyx.jit._cuda_types import ArrayBase, TypeBase, PointerBase
from typing import Optional
from cupyx.jit._internal_types import BuiltinFunc, Data
from cupy_backends.cuda.api import runtime
from cupy.cuda import device
from cupyx.jit import _cuda_types
from cupyx.jit import _compile

parent_globals = None

#routines to lambdify expressions into GPU device functions

def convert_raw_source(source):
    lines = source.splitlines()
    source = lines[0]+"\n"+lines[2]
    arrayname = lines[1].split("=")[1].strip(" ")
    varnames = lines[1].split("=")[0].strip(" []").split(", ")
    for i, name in enumerate(varnames):
        source = source.replace(name, arrayname+f"[{i}]")
    return source

#https://stackoverflow.com/questions/48709104/how-do-i-specify-multiple-types-for-a-parameter-using-type-hints
#https://stackoverflow.com/questions/47533787/typehinting-tuples-in-python
from typing import Union, Tuple

def to_ctype(t) -> _cuda_types.TypeBase:
    if isinstance(t, _cuda_types.TypeBase):
        return t
    return _cuda_types.Scalar(np.dtype(t))

class ContiguousArray(PointerBase):
    #used to define ndarray of local memory, in <dtype>[3][4] format
    def __init__(self, size: Tuple[int, ...], child_type: TypeBase) -> None:
        self.base_type = child_type
        if(type(self.base_type) is ContiguousArray):
            self.base_type = child_type.base_type
        
        super().__init__(child_type)
        self._size = size
        self.dtype = child_type
        self._c_contiguous = True
        self._index_32_bits = True
    
    def declvar(self, x: str, init: Optional['Data']) -> str:
        s = f'{self.base_type} (*{x})'
        for i in range(1, len(self._size)):
            s += f'[{self._size[i]}]'
        if(init is None):
            return s
        elif(f"{init.code}" == "!!lmem"):
            s = f'{self.base_type} {x}'
            for var in self._size:
                s += f'[{var}]'
            return f"{s}"
        else:
            return f'{s} = {init.code}'

class LocalMem(ArrayBase):

    def __init__(
            self,
            child_type: TypeBase,
            size: Union[int, Tuple[int, ...]],
            alignment: Optional[int] = None,
    ) -> None:
        if not (isinstance(size, int) or isinstance(size, tuple)):
            raise 'size of local_memory must be integer, or a tuple of integers'
        if not (isinstance(alignment, int) or alignment is None):
            raise 'alignment must be integer or `None`'
        self._size = size
        self._alignment = alignment
        super().__init__(child_type, 1)

    def declvar(self, x: str, init: Optional['Data']) -> str:
        assert init is None
        return "//define local memory later" #emit a comment here, build local memory later
    
class LocalMemory(BuiltinFunc):

    def __call__(self, size, dtype, alignment=None):
        """Allocates local memory and returns it as a 1-D array.

        Args:
            dtype (dtype):
                The dtype of the returned array.
            size (int or tuple):
                If ``int`` type, the size of static local memory.
                If ``tuple`` type, the size of a multi-dimensional static local memory.
                Does not use __extern__ keyword like shared memory does
            alignment (int or None): Enforce the alignment via __align__(N).
        """
        super().__call__()

    def call_const(self, env, size, dtype, alignment=None):
        name = env.get_fresh_variable_name(prefix='_lmem')
        ctype = to_ctype(dtype)
        var = Data(name, LocalMem(ctype, size, alignment))
        env.decls[name] = var
        env.locals[name] = var
        if(type(size) is int):
            size = (size,)
        return_ctype = ctype
        for i in range(len(size)):
            return_ctype = ContiguousArray(size[len(size)-(i+1):], return_ctype)
        return Data("!!lmem", return_ctype) #non-valid name in order to intercept creation of this
    
local_memory = LocalMemory()

def better_exec(code_, globals_=None, locals_=None, /):
    import ast
    import linecache

    if not hasattr(better_exec, "saved_sources"):
        old_getlines = linecache.getlines
        better_exec.saved_sources = []

        def patched_getlines(filename, module_globals=None):
            if "<exec#" in filename:
                index = int(filename.split("#")[1].split(">")[0])
                return better_exec.saved_sources[index].splitlines(True)
            else:
                return old_getlines(filename, module_globals)

        linecache.getlines = patched_getlines
    better_exec.saved_sources.append(code_)
    compiled_code = compile(
            ast.parse(code_),
            filename=f"<exec#{len(better_exec.saved_sources) - 1}>",
            mode="exec",
        )
    exec(
        compiled_code,
        globals_,
        locals_,
    )
    
def create_sf_from_model(model, nparray=True):
    comps = model.components
    comps.sort() #just to be absolutely sure...
    sf = []
    for i in range(len(model.constituents)):
        d = model.constituents[i]
        l = list(d)
        l.sort()
        for s in l:
            index = comps.index(s)
            sf.append([i, index])
    if(nparray):
        return np.array(sf)
    else:
        return sf

def create_cupy_ufunc_from_tdb(model, attribute, has_vv=False, param_dict=None, debug_level=0):
    """
    Creates a numba nopython ufunc from the given phase/components of the tdb
    """
    global cupy_ufunc
    index = model.phase_name
    caller_globals = dict(inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
    if not hasattr(create_cupy_ufunc_from_tdb, "num_funcs"):
        create_cupy_ufunc_from_tdb.num_funcs = 0
    expr = sp.parse_expr(str(getattr(model, attribute)))
    syms = expr.free_symbols
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    for sym in syms: 
        if(sym.name == "T"):
            syms.remove(sym)
            syms.append(sym) #explicitly ensure T is always the last symbol, in case there is a phase "Zeta" or something
    sympy_ufunc = sp.lambdify([syms], expr, "math")
    raw_source = sympy_ufunc.__doc__.split("Source code:")[1].strip("\n").split("Imported modules:")[0].strip("\n")
    raw_source = raw_source.replace("log(", "cp.log(")
    raw_source = convert_raw_source(raw_source)
    sc = compile(raw_source, f"<pycgpu_test_function{create_cupy_ufunc_from_tdb.num_funcs}>", "exec")
    exec(sc, globals())
    linecache.cache[f"<pycgpu_test_function{create_cupy_ufunc_from_tdb.num_funcs}>"] = (len(raw_source), None, raw_source.splitlines(True), f"<pycgpu_test_function{create_cupy_ufunc_from_tdb.num_funcs}>")
    create_cupy_ufunc_from_tdb.num_funcs += 1
    cupy_ufunc = jit.rawkernel(device=True)(_lambdifygenerated) #_lambdifygenerated is dynamically defined in sc
    better_exec(f"global _pycgpu_ufunc_{index}_{attribute}_raw2\n"\
         f"_pycgpu_ufunc_{index}_{attribute}_raw2 = cupy_ufunc", globals(), locals())
    if(has_vv and attribute == "GM"): #vv terms only modify GM for now...
        s = f"global _pycgpu_ufunc_{index}_{attribute}_raw\n"\
            f"@jit.rawkernel(device=True)\n"\
            f"def _pycgpu_ufunc_{index}_{attribute}_raw(c, vv_array):\n"\
            f"    e = _pycgpu_ufunc_{index}_{attribute}_raw2(c)\n"\
            f"    T = c[{len(sf)}]\n"
        sfl = create_sf_from_model(model, nparray=False)
        for i in range(len(param_dict["is_G"])):
            is_G = param_dict["is_G"][i]
            if(is_G):
                modifier = 1./np.sum(model.site_ratios)
                s2 = "    e += "
                for j in range(len(param_dict["constituents"][i])):
                    c_index = sfl.index([j, param_dict["constituents"][i][j]])
                    s2 += f"c[{c_index}]*"
                s2 += param_dict["coeff"][i]
                s2 += f"vv_array[{param_dict['id'][i]}]*{modifier}\n"
                s += s2
            else:
                found_double = False
                s2 = "    e += "
                s3 = "("
                for j in range(len(param_dict["constituents"][i])):
                    l = param_dict["constituents"][i][j]
                    if(len(l) == 2):
                        c_index1 = sfl.index([j, l[0]])
                        c_index2 = sfl.index([j, l[1]])
                        s2 += f"c[{c_index1}]*c[{c_index2}]*"
                        order = param_dict["order"][i]
                        if(order == 0):
                            s3 += "1)\n"
                        else:
                            s3 += f"c[{c_index1}]-c[{c_index2}])**{order}\n"
                        if(found_double):
                            raise Exception("Cannot currently deal with L terms of higher order!")
                        found_double = True
                    else:
                        c_index1 = sfl.index([j, l[0]])
                        s2 += f"c[{c_index1}]*"
                s2 += param_dict["coeff"][i]
                s2 += f"vv_array[{param_dict['id'][i]}]*"
                s2 += s3
                s += s2
        s += "    return e"
        if(debug_level > 2):
            print(s)
            print("")
            print("###############################################################################################################\n")
        better_exec(s, globals())
                
    else:
        if(debug_level > 2):
            s = str(sympy_ufunc.__doc__).split("Source code:")[1].split("Imported modules:")[0].strip(" \n")
            s2 = f"_pycgpu_ufunc_{index}_{attribute}_raw"
            s = s.replace("_lambdifygenerated", s2)
            print(f"global _pycgpu_ufunc_{index}_{attribute}_raw")
            print("@jit.rawkernel(device=True)")
            print(s)
            print("")
            print("###############################################################################################################\n")
        better_exec(f"global _pycgpu_ufunc_{index}_{attribute}_raw\n"\
             f"_pycgpu_ufunc_{index}_{attribute}_raw = cupy_ufunc", globals())
    parent_globals[f"_pycgpu_ufunc_{index}_{attribute}_raw"] = globals()[f"_pycgpu_ufunc_{index}_{attribute}_raw"]
    
def create_calculate_kernel(model):
    create_cupy_ufunc_from_tdb(model, "GM")
    name = model.phase_name
    sf = create_sf_from_model(model)
    sr = np.array(model.site_ratios)
    sr /= np.sum(sr)
    s = f"global _pycgpu_{name}_kernel\n"\
        f"@jit.rawkernel()\n"\
        f"def _pycgpu_{name}_kernel(Ts, points, output, mass_output):\n"\
        f"    startx = jit.grid(1)\n"\
        f"    stridex = jit.gridsize(1)\n"\
        f"    pshape = points.shape[0]\n"\
        f"    tshape = Ts.shape[0]\n"\
        f"    loop_length = pshape*tshape\n"\
        f"    l = local_memory({len(sf)+1}, np.float64)\n"\
        f"    for i in range(startx, loop_length, stridex):\n"\
        f"        pi = i%pshape\n"\
        f"        Ti = i//pshape\n"\
        f"        for j in range({len(sf)}):\n"\
        f"            l[j] = points[pi][j]\n"\
        f"        l[{len(sf)}] = Ts[Ti]\n"\
        f"        output[i] = _pycgpu_ufunc_{name}_GM_raw(l)\n"\
        f"        for j in range({len(model.components)}):\n"\
        f"            mass_output[i][j] = 0.\n"
    for i in range(len(sf)):
        s += f"        mass_output[i][{sf[i][1]}] += {sr[sf[i][0]]}*points[pi][{i}]\n"
    better_exec(s, globals())
    parent_globals[f"_pycgpu_{name}_kernel"] = globals()[f"_pycgpu_{name}_kernel"]