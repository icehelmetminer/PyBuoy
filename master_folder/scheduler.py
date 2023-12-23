import time
import threading

class Scheduler:
    def __init__(self):
        self.tasks = []

    def schedule_task(self, task, interval_minutes):
        def task_wrapper():
            while True:
                task()
                time.sleep(interval_minutes * 60)

        t = threading.Thread(target=task_wrapper)
        t.start()
        self.tasks.append(t)

    def run_continuously(self):
        while True:
            time.sleep(1)
