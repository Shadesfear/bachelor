import benchpress as bp
from benchpress.suite_util import BP_ROOT
import random

points = [random.randint(0, 100) for i in range(1000)]

scripts = [

    ('Move Centroids',  'bohrium_kmeans',  [points, "10"])
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
