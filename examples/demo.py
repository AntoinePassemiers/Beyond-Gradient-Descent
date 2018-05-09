# -*- coding: utf-8 -*-
# autoencoder.py: Implement an interactive window for autoencoder demo
# author : Antoine Passemiers, Robin Petit

import pickle
import tkinter as tk

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import numpy as np

from autoencoder import get_trained_autoencoder, mnist
import bgd


COLORS = ['r', 'g', 'b', 'c', 'limegreen', 'y', 'k', 'purple', 'orange', 'pink']

class Window:
    def __init__(self, nn, master=None):
        if master is None:
            self.master = tk.Tk()
        else:
            self.master = master
        self.nn = nn
        self.fig = Figure(figsize=(20, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.scatter_ax = self.fig.add_axes([.05, .05, .45, .9])
        self.imshow_ax = self.fig.add_axes([.55, .05, .40, .9])
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.callbacks.connect('button_press_event',
                                      lambda event: self.on_click(event))
        
    def on_click(self, event):
        if not event.inaxes is self.scatter_ax:
            return
        X = np.asarray([event.xdata, event.ydata])
        Y = self.nn.eval(X, start=6).reshape(28, 28)
        self.imshow_ax.imshow(Y, interpolation='nearest')
        self.canvas.draw()
    
    def scatterplot(self, mnist, N=8192):
        indices = np.arange(mnist.data.shape[0])
        np.random.shuffle(indices)
        indices = indices[:N]
        Xs = mnist.data[indices]
        ys = mnist.target[indices]
        dots = self.nn.eval(Xs, stop=6)
        assert dots.shape == (N, 2)
        cmap = matplotlib.cm.autumn
        for c, y in enumerate(np.unique(ys)):
            color = COLORS[c]
            self.scatter_ax.plot(dots[ys == y,0], dots[ys == y,1], marker='${:d}$'.format(int(y)),
                                 color=color, lw=0)
        self.canvas.draw()
    
    def start(self):
        self.master.mainloop()

if __name__ == '__main__':
    autoencoder = get_trained_autoencoder()
    #autoencoder = pickle.load(open('../autoencoder.pickle', 'rb'))
    window = Window(autoencoder)
    window.scatterplot(mnist)
    window.start()