import os
import json
import time
import torch
import operator
from functools import reduce
import matplotlib.pyplot as plt
import collections
from .utils import xmkdir


class TotalAverage():
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_value = 0.
        self.mass = 0.
        self.sum = 0.

    def update(self, value, mass=1):
        self.last_value = value
        self.mass += mass
        self.sum += value * mass

    def get(self):
        return self.sum / self.mass

class MovingAverage():
    def __init__(self, inertia=0.9):
        self.inertia = inertia
        self.reset()
        self.last_value = None

    def reset(self):
        self.last_value = None
        self.average = None

    def update(self, value, mass=1):
        self.last_value = value
        if self.average is None:
            self.average = value
        else:
            self.average = self.inertia * self.average + (1 - self.inertia) * value

    def get(self):
        return self.average

class MetricsTrace():
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = {}

    def append(self, dataset, metric):
        if dataset not in self.data:
            self.data[dataset] = []
        self.data[dataset].append(metric.get_data_dict())

    def load(self, path):
        """Load the metrics trace from the specified JSON file."""
        with open(path, 'r') as f:
            self.data = json.load(f)

    def save(self, path):
        """Save the metrics trace to the specified JSON file."""
        if path is None:
            return
        xmkdir(os.path.dirname(path))
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def plot(self, pdf_path=None):
        """Plots and optionally save as PDF the metrics trace."""
        plot_metrics(self.data, pdf_path=pdf_path)

    def get(self):
        return self.data

    def __str__(self):
        pass

class Metrics():
    def __init__(self):
        self.iteration_time = MovingAverage(inertia=0.9)
        self.now = time.time()

    def update(self, prediction=None, ground_truth=None):
        self.iteration_time.update(time.time() - self.now)
        self.now = time.time()

    def get_data_dict(self):
        return {"objective" : self.objective.get(), "iteration_time" : self.iteration_time.get()}

class StandardMetrics(Metrics):
    def __init__(self, m=None):
        super(StandardMetrics, self).__init__()
        self.metrics = m or {}
        self.speed = MovingAverage(inertia=0.9)

    def update(self, metric_dict, mass=1):
        super(StandardMetrics, self).update()
        for metric, val in metric_dict.items():
            if torch.is_tensor(val):
                val = val.item()
            if metric not in self.metrics:
                self.metrics[metric] = TotalAverage()
            self.metrics[metric].update(val, mass)
        self.speed.update(mass / self.iteration_time.last_value)

    def get_data_dict(self):
        data_dict = {k: v.get() for k,v in self.metrics.items()}
        data_dict['speed'] = self.speed.get()
        return data_dict

    def __str__(self):
        pstr = '%7.1fHz\t' %self.speed.get()
        pstr += '\t'.join(['%s: %6.5f' %(k,v.get()) for k,v in self.metrics.items()])
        return pstr

def plot_metrics(stats, pdf_path=None, fig=1, datasets=None, metrics=None):
    """Plot metrics. `stats` should be a dictionary of type

          stats[dataset][t][metric][i]

    where dataset is the dataset name (e.g. `train` or `val`), t is an iteration number,
    metric is the name of a metric (e.g. `loss` or `top1`),  and i is a loss dimension.

    Alternatively, if a loss has a single dimension, `stats[dataset][t][metric]` can
    be a scalar.

    The supported options are:

    - pdf_file: path to a PDF file to store the figure (default: None)
    - fig: MatPlotLib figure index (default: 1)
    - datasets: list of dataset names to plot (default: None)
    - metrics: list of metrics to plot (default: None)
    """
    plt.figure(fig)
    plt.clf()
    linestyles = ['-', '--', '-.', ':']
    datasets = list(stats.keys()) if datasets is None else datasets
    # Filter out empty datasets
    datasets = [d for d in datasets if len(stats[d]) > 0]
    duration = len(stats[datasets[0]])
    metrics = list(stats[datasets[0]][0].keys()) if metrics is None else metrics
    for m, metric in enumerate(metrics):
        plt.subplot(len(metrics),1,m+1)
        legend_content = []
        for d, dataset in enumerate(datasets):
            ls = linestyles[d % len(linestyles)]
            if isinstance(stats[dataset][0][metric], collections.Iterable):
                metric_dimension = len(stats[dataset][0][metric])
                for sl in range(metric_dimension):
                    x = [stats[dataset][t][metric][sl] for t in range(duration)]
                    plt.plot(x, linestyle=ls)
                    name = f'{dataset} {metric}[{sl}]'
                    legend_content.append(name)
            else:
                x = [stats[dataset][t][metric] for t in range(duration)]
                plt.plot(x, linestyle=ls)
                name = f'{dataset} {metric}'
                legend_content.append(name)
        plt.legend(legend_content, loc=(1.04,0))
        plt.grid(True)
    if pdf_path is not None:
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.draw()
    plt.pause(0.0001)
