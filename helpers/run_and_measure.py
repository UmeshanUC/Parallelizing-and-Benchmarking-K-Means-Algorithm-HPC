### Get the time taken to run the passed funtion/method in MiliSeconds
import time


def run_and_measure(func, args):

    start =  time.time()
    func(args)
    duration = (time.time()-start) * 1000
    return duration