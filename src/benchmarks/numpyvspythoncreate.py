import benchpress as bp
from benchpress.suite_util import BP_ROOT
import random

scripts = [
    ('Numpy', 'numpy_random_centroids', ["10**6", "10**7", "2*10**7", "5*10**7", "8*10**7"]),
    ('Python', 'python_random_centroids', ["10**6", "10**7", "2*10**7", "5*10**7", "8*10**7"]),
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
