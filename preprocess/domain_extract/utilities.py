import os
import shutil
from fnmatch import fnmatch
from itertools import islice
import multiprocessing as mp
import time
from tqdm import tqdm


def RecurseListDir(root: str, pattern: list[str]):
    f = []
    for p in pattern:
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, p):
                    f.append(os.path.join(path, name))
    return f


def IterChunk(arr_range, arr_size) -> list[str]:
    arr_range = iter(arr_range)
    return list(iter(lambda: tuple(islice(arr_range, arr_size)), ()))


def CleanDir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        return
    for file_obj in os.listdir(path):
        file_obj_path = os.path.join(path, file_obj)
        if (os.path.isfile(file_obj_path)) or (os.path.islink(file_obj_path)):
            os.unlink(file_obj_path)
        else:
            shutil.rmtree(file_obj_path)

def MultiProcessing(Worker, args:tuple, n_worker:int):
    print(f"MultiProcess: {n_worker}")
    ps = []
    for i in range(n_worker):
        p = mp.Process(
            target=Worker,
            args=args
        )
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

def ProgressBar(queue: mp.Queue, total:int, name:str="Progress"):
    with tqdm(total = total) as pbar:
        pbar.set_description(name)
        while True:
            current = queue.qsize()
            pbar.n = total - current
            pbar.refresh()
            time.sleep(1)
            if queue.empty():
                pbar.close()
                exit()

def GetNumberOfWorker()->int:
    ncpu = mp.cpu_count()
    print(f"Number of CPU: {ncpu}")
    nworker = ncpu - 2 if ncpu > 2 else 1
    return nworker