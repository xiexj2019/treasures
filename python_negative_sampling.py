
import os

from multiprocessing import Pool

def worker(*args, **kwds):
    cnt, pid = 0, os.getpid()
    with open("pros/{}.txt".format(pid), "w") as fout:
        pass
        # ...
        # cnt += 1
        # if cnt % 100 == 0:
        #     print(pid, cnt)
        #
        # fout.write(... + "\n")

def manager(*args, **kwds):
    with Pool(20) as p:
        p.map(worker)
        # p.map(worker, args)
        p.close()
        p.join()
