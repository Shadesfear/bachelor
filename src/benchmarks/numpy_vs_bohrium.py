import benchpress as bp
from benchpress.suite_util import BP_ROOT
import random

scripts = [
    ('Bohrium', 'bohrium_kmeans', ["3*5","4*5","5*5", "6*5", "7*5", "8*5","3*6","5*6"]),
    ('Numpy', 'pure-numpy-kmeans', ["3*5","4*5","5*5", "6*5", "7*5", "8*5","3*6", "5*6"]),
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
