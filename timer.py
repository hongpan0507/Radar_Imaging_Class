# timer.py
# an example from: https://realpython.com/python-timer/

import time

class TimerError(Exception):
    print("Exception")


class Timer:
    def __init__(self, text="Elapsed time: {:0.4f} seconds"):
        self.start_time = None
        self.initial_time = time.perf_counter()
        self.text = text

    def start(self):
        if self.start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise TimerError(f"Timer is not running. User .start_time to start")
        total_time = time.perf_counter() - self.initial_time
        self.start_time = None
        print(f"Total time: {total_time:0.4f} seconds")

    def count(self):
        elapsed_time = time.perf_counter() - self.start_time
        self.start_time = time.perf_counter()
        print(self.text.format(elapsed_time))
