import benchpress as bp
from benchpress.suite_util import BP_ROOT
import random

scripts = [
    ('Userkernel', 'bohrium_kmeans', ["3*0","4*0","5*0", "6*0", "7*0", "8*0"]),
    ('Bohrium', 'bohrium_kmeans', ["3*1","4*1","5*1", "6*1", "7*1", "8*1"]),
    # ('numpy_version', 'pure-numpy-kmeans', ["10", "20", "30", "40", "50", "100", "500"])

]

cmd_list = []
for label, name, sizes in scripts:
    for size in sizes:
        full_label = "%s/%s" % (label, size)
        bash_cmd = "python /home/cca/bachelor/src/{script}.py {size}" \
                    .format(root=BP_ROOT, script=name, size=size)
        cmd_list.append(bp.command(bash_cmd, full_label))

# Finally, we build the Benchpress suite, which is written to `--output`
bp.create_suite(cmd_list)
