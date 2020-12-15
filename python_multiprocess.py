
import os
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

        
from multiprocessing import Pool
def manager(*args, **kwds):
    with Pool(20) as p:
        p.map(worker)
        # p.map(worker, args)
        p.close()
        p.join()

        
from multiprocessing import Process
def manager2(*args, **kwds):
    for _ in range(20):
        p = Process(target=worker)
        # p = Process(target=worker, args=(variable, constant, list, query, ...))
        p.start()
        p.join()
