import benchpress as bp
from benchpress.suite_util import BP_ROOT
import random

scripts = [
    ('Bohrium', 'bohrium_kmeans', ["3*5*0","4*5*0","5*5*0", "6*5*0", "7*5*0", "8*5*0"]),
    ('numpy', 'bohrium_kmeans', ["3*5*1","4*5*1","5*5*1", "6*5*1", "7*5*1", "8*5*1"]),
    # ('numpy_version', 'pure-numpy-kmeans', ["10", "20", "30", "40", "50", "100", "500"])

]

cmd_list = []
for label, name, sizes in scripts:
    for size in sizes:
        full_label = "%s/%s" % (label, size)
        bash_cmd = "python /home/chris/Documents/bachelor/src/{script}.py {size}" \
                    .format(root=BP_ROOT, script=name, size=size)
        cmd_list.append(bp.command(bash_cmd, full_label))

# Finally, we build the Benchpress suite, which is written to `--output`
bp.create_suite(cmd_list)
