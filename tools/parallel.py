import multiprocessing
import os

from tqdm import tqdm


class ForkPool:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers if num_workers and num_workers > 0 else max(1, (os.cpu_count() or 2) - 2)

    def map(self, worker, tasks, initializer=None, initargs=(), desc=None):
        tasks = list(tasks)

        if self.num_workers <= 1:
            if initializer is not None:
                initializer(*initargs)
            return [worker(task) for task in tqdm(tasks, total=len(tasks), desc=desc)]

        context = multiprocessing.get_context("fork")
        results = []
        with context.Pool(self.num_workers, initializer=initializer, initargs=initargs) as pool:
            for result in tqdm(pool.imap(worker, tasks), total=len(tasks), desc=desc):
                results.append(result)

        return results


class Batcher:
    def __init__(self, batch_size):
        self.batch_size = max(int(batch_size), 1)

    def split(self, items):
        items = list(items)
        return [items[start:start + self.batch_size] for start in range(0, len(items), self.batch_size)]
