import time

class MeasureTimer:
    def __init__(self):
        self.count = 0
        self.elapsed = 0
        pass

    def start(self):
        self.start_ = time.time()
    def end(self):
        self.elapsed += time.time() - self.start_
        self.count += 1
        self.start_ = None

timers = {}

def startTime(name):
    if not name in timers:
        timers[name] = MeasureTimer()
    timers[name].start()

def endTime(name):
    timers[name].end()

def printTime():
    for name in timers.keys():
        timer = timers[name]
        print(name,timer.count,timer.elapsed,timer.elapsed/timer.count)

