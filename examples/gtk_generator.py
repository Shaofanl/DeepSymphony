from DeepSymphony.models import StackedRNN
import keras.backend as K
from keras.layers import LSTM

from gi.repository import Gtk, GObject
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk3cairo import \
    FigureCanvasGTK3Cairo as FigureCanvas


LEN = 2000
DIM = 128+128+100+7


class VisualizeWindow(Gtk.Window):
    def __init__(self, model, lstm_layers,
                 heatmap_height=4):
        Gtk.Window.__init__(self, title="Visualize Generator")

        self.model = model
        self.lstm_layers = lstm_layers

        self.set_default_size(1600, 400)
        self.set_border_width(10)
        grid = Gtk.Grid(column_homogeneous=True,
                        row_homogeneous=True,
                        column_spacing=10,
                        row_spacing=10)
        self.add(grid)

        self.figs = []
        self.axs = []
        self.imshows = []
        self.canvas = []
        for i in range(len(lstm_layers)):
            sw, fig, ax, imshow, canvas = self.figure_placeholder((16, 32))
            imshow.set_cmap('Reds')
            grid.attach(sw, i, 0, 1, heatmap_height)

            self.figs.append(fig)
            self.axs.append(ax)
            self.imshows.append(imshow)
            self.canvas.append(canvas)

        sw, fig, ax, imshow, canvas = self.figure_placeholder((1, 128),
                                                              aspect='auto')
        grid.attach(sw, 0, heatmap_height, len(lstm_layers), 1)
        imshow.set_cmap('binary')
        self.figs.append(fig)
        self.axs.append(ax)
        self.imshows.append(imshow)
        self.canvas.append(canvas)
        self.keyboard = np.zeros((1, 128))

        btn_start = Gtk.Button(label="Block and Generate")
        btn_start.connect("clicked", self.on_click)
        hb = Gtk.HeaderBar(title="generate and visualize")
        hb.pack_end(btn_start)
        self.set_titlebar(hb)

        # testing
        # self.timeout_id = GObject.timeout_add(10, self.on_timeout, None)

    def on_click(self, widget):
        kwargs = {'weight_path': 'temp/simple_rnn.h5',
                  'seed': 20,
                  'verbose': 0,
                  'length': 1000,
                  'callbacks': [self]}
        self.model.generate(**kwargs)

    def on_timeout(self, user_data):
        # print '>>>', self.keyboard.sum()
        return True

    def figure_placeholder(self,
                           shape=(10, 1),
                           aspect=None):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
        ax.axis('off')
        imshow = ax.imshow(np.random.rand(*shape),
                           animated=True,
                           interpolation=None, aspect=aspect)
        sw = Gtk.ScrolledWindow()
        canvas = FigureCanvas(fig)
        sw.add_with_viewport(canvas)
        return sw, fig, ax, imshow, canvas

    def _generator_callback(self, generator, note):
        for ind, layer in enumerate(self.lstm_layers):
            cell = layer.states[1].eval(session=K.get_session())
            self.imshows[ind].set_data(cell.reshape(16, 32))
            self.canvas[ind].draw()

        note = note.argmax()
        if note <= 127:
            self.keyboard[0, note] = 1
        elif note <= 128+127:
            self.keyboard[0, note-128] = 0
        self.imshows[-1].set_data(self.keyboard)
        self.canvas[-1].draw()


if __name__ == '__main__':
    model = StackedRNN(timespan=LEN,
                       input_dim=DIM,
                       output_dim=DIM,
                       cells=[512, 512, 512])
    generator = model.build_generator()
    lstm_layers = []
    for layer in generator.layers:
        if isinstance(layer, LSTM):
            lstm_layers.append(layer)
    print lstm_layers

    window = VisualizeWindow(model=model,
                             lstm_layers=lstm_layers)
    window.connect("delete-event", Gtk.main_quit)
    window.show_all()
    Gtk.main()
