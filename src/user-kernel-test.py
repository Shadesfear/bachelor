import bohrium as bh
from itertools import product

ctype = {
    'float32': 'float',
    'float64': 'double',
    'complex64': 'float complex',
    'complex128':'double complex',
    'int8':  'int8_t',
    'int16': 'int16_t',
    'int32': 'int32_t',
    'int64': 'int64_t',
    'uint8':  'uint8_t',
    'uint16': 'uint16_t',
    'uint32': 'uint32_t',
    'uint64': 'uint64_t',
    'bool': 'uint8_t'
}


def axis_split(A,axis):
    (Am,Ai,An) = (product(A.shape[:axis]),  A.shape[axis], product(A.shape[axis+1:]));
    return (Am.__next__,Ai,An.__next__)

def take_cpu(A,I,axis=0):
    (Am,Ai,An) = axis_split(A,axis)
    Ar = bh.user_kernel.make_behaving(A.reshape(Am,Ai,An),dtype=A.dtype);

    I  = bh.user_kernel.make_behaving(I,dtype=np.int64);

    Ii = product(I.shape)

    new_shape = list(A.shape[:axis])+list(I.shape)+list(A.shape[axis+1:]);
    cmd = bh.user_kernel.get_default_compiler_command()
    kernel = read_file("lookup-cpu.c") % {'ctype':ctype[A.dtype.name],'Am':Am,'Ai':Ai,'Ii':Ii,'An':An};

    AI = bh.empty(new_shape,dtype=A.dtype)
    bh.user_kernel.execute(kernel, [AI, A, I], compiler_command=cmd)



    return AI.reshape(new_shape);



a  = np.arange(1,5);


ix = a[:,None]-a[None,:];

print(take_cpu(a,ix, 0))
