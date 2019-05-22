import benchpress as bp
from benchpress.suite_util import BP_ROOT
import random

scripts = [
    ('CPU', 'bohrium_kmeans', ["10*0", "20*0", "30*0", "40*0", "50*0", "100*0", "500*0"]),
    ('GPU', 'bohrium_kmeans', ["10*1", "20*1", "30*1", "40*1", "50*1", "100*1", "500*1"]),
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
