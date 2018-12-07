import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
class TimeKeeper(object):
    def __init__(self):
        self.runtimes = []

    def accum(self, subseq_time, sax_time, collision_time, filter_time, gap_time, search_time):
        rlist = [subseq_time,sax_time,search_time,collision_time,filter_time,gap_time]
        self.runtimes.append(rlist)

    def clear(self):
        self.runtimes = []

    def print_summary(self):
        times = np.array(self.runtimes)
        mean_times = np.mean(times,axis=0)
        names = ["Subsequence","SAX","Search","Collision","Filtering","Gap Score"]
        colors = [cm.tab10(i) for i in range(len(names))]
        plt.figure(figsize=(10,9))
        plt.title("Runtime per Section")
        plt.ylabel("Runtime (s)")
        plt.xlabel("Section")
        plt.bar(range(len(names)),mean_times,color=colors)
        plt.xticks(range(len(names)),labels=names)
        plt.savefig("/tmp/pycharm_project_926/bar_chart.png")
        normalized = 100*mean_times/np.sum(mean_times)
        plt.figure(figsize=(10,9))
        plt.title("Percentage of Total Runtime by Section")
        plt.pie(normalized,labels=names,autopct="%2.2f%%",explode=(0.15,0.15,0.15,0.15,0.15,0.15),colors=colors)
        plt.savefig("/tmp/pycharm_project_926/usage_chart.png")