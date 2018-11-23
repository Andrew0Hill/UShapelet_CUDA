import numpy as np
from matplotlib import pyplot as plt

class TimeKeeper(object):
    def __init__(self):
        self.runtimes = []

    def accum(self, subseq_time, sax_time, collision_time, filter_time, gap_time, search_time):
        rlist = [subseq_time, sax_time, collision_time,filter_time, gap_time, search_time]
        self.runtimes.append(rlist)

    def clear(self):
        self.runtimes = []

    def print_summary(self):
        times = np.array(self.runtimes)
        mean_times = np.mean(times,axis=0)
        normalized = 100*mean_times/np.sum(mean_times)
        plt.figure(figsize=(9,7))
        plt.title("CPU Time per Section")
        plt.pie(normalized,labels=["Subsequence","SAX","Collision","Filtering","Gap Score", "Search"],autopct="%2.2f%%",explode=(0.2,0.2,0.2,0.2,0.2,0.2))
        plt.savefig("/tmp/pycharm_project_681/usage_chart.png")