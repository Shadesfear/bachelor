import benchpress as bp
from benchpress.suite_util import BP_ROOT
import random

scripts = [
    ('kmeans', 'bohrium_kmeans',  ["10", "20", "30", "40", "50", "100", "1000"]),
    ('python_version', 'legacy/myownkmeans', ["10", "20", "30", "40", "50", "100", "1000"])
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
