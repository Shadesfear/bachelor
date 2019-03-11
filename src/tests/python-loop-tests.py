import time
import numpy as np

N = 100000
s = 0
t = 0
# A = [i for i in range(N)]
# B = [i for i in range(N)]
A = np.arange(N)
B = np.arange(N)
n_tests = 1

bad_loop_time = 0
good_loop_time = 0

start = time.time()
for i in range(N):
    s += A[i]
    t += B[i]

end = time.time()

bad_loop_time += end - start



start = time.time()
for i in range(N):
    s += A[i]

for i in range(N):
    t += B[i]
end = time.time()

good_loop_time += end - start

bad_loop_time /= n_tests
good_loop_time /= n_tests

print("\nBad Loop ", bad_loop_time)
print("Good Loop ", good_loop_time)
