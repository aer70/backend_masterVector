from __future__ import annotations

import os

from rq import Connection, SimpleWorker
from rq.timeouts import TimerDeathPenalty

from backend.main import get_job_queue, init_runtime


def main() -> None:
    init_runtime()
    queue = get_job_queue()
    worker_name = f"bmp2svg-worker-{os.getpid()}"
    with Connection(queue.connection):
        worker = SimpleWorker([queue], name=worker_name)
        worker.death_penalty_class = TimerDeathPenalty
        worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
