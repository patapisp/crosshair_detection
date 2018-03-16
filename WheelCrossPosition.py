import numpy as np
from tkinter import *
from astropy.io import fits
import os
from scipy.ndimage import imread
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import matplotlib
from matplotlib.widgets import Cursor
from skimage.color import rgb2gray


matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from crosshair_definitions import find_crosshair_center

class CrossCenterFinder:
    """
    GUI for caclulating and viewing the center position of a slot of the
    APW ERIS wheel
    """
    def __init__(self, root):
        self.master = Frame(root)
        self.master.grid(column=0, row=0, sticky=(N, W, E, S))
        root.title('Position center finder')


        self.image = np.zeros((1288, 1936, 3))
        self.detectorX = 22.7  # mm
        self.detectorY = 15.1  # mm

        self.centerY = int(self.image.shape[0]/2)
        self.centerX = int(self.image.shape[1]/2)
        self.dfx = 0
        self.dfy = 0
        self.fig, self.ax = plt.subplots()
        self.im = plt.imshow(self.image, interpolation='nearest')
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.lx1 = plt.plot(self.centerX, self.centerY, color="r")[0]
        self.lx2 = plt.plot(self.centerX, self.centerY, color="r")[0]
        self.ly1 = plt.plot(self.centerX, self.centerY, color="r")[0]
        self.ly2 = plt.plot(self.centerX, self.centerY, color="r")[0]

        self.centerplot = plt.plot(self.centerX, self.centerY, 'rx')[0]
        #self.crossh = self.ax.axhline(self.centerX, color="red", linewidth=1.2)
        #self.crossv = self.ax.axvline(self.centerY, color="red", linewidth=1.2)
        self.center_text = self.ax.text(0.4, 1.1, 'Center X:%.2f, Center Y:%.2f'%(self.centerX, self.centerY),
                                        ha='center', va='center',
                                        size=12, transform=self.ax.transAxes,
                                        bbox=dict(facecolor='white', alpha=0.5))
        self.defocus_text = self.ax.text(0.4, 1.2, 'Defocus X:%.2f, Defocus Y:%.2f'%(0, 0),
                                        ha='center', va='center',
                                        size=12, transform=self.ax.transAxes,
                                        bbox=dict(facecolor='white', alpha=0.5))
        # get image plot onto canvas and app
        self.data_plot = FigureCanvasTkAgg(self.fig, master=self.master)
        self.data_plot.get_tk_widget().configure(borderwidth=0)
        #self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=2)

        self.data_plot.show()
        #self.fig.canvas.mpl_connect('button_press_event', self.click_callback)

        self.controlsFrame = ttk.Frame(self.master)
        self.import_btn = ttk.Button(self.controlsFrame, text="Import image",
                                     command=self.import_image)
        self.analyse_btn = ttk.Button(self.controlsFrame, text="Update",
                                      command=self.check_updated_img)
        self.defpos_btn = ttk.Button(self.controlsFrame, text="Set default pos",
                                      command=self.set_defpos)

        self.mask_nr_label = Label(self.controlsFrame, text="Mask number:")
        self.mask_nr = StringVar()
        self.mask_nr.set('1')
        self.mask_nr_entry = Entry(self.controlsFrame, textvariable=self.mask_nr, justify='center')
        self.mask_nr_entry.bind("<Return>", self.mask_nr_entry_callback)
        self.import_btn.grid(column=0, row=0)
        self.analyse_btn.grid(column=0, row=1)
        self.defpos_btn.grid(column=1, row=1)
        self.mask_nr_label.grid(column=0, row=2)
        self.mask_nr_entry.grid(column=1, row=2)

        self.refpositions_dict = {'1': 0}
        self.refpos = 1

    def set_defpos(self):
        pass

    def mask_nr_entry_callback(self, event):
        self.refpos = self.refpositions_dict[self.mask_nr.get()]
        return

    def import_image(self):
        self.filename = filedialog.askopenfilename(title="Select image")
        self.initfiletime = os.path.getmtime(self.filename)
        self.image = imread(self.filename)
        self.img_gray = rgb2gray(self.image)
        self.update_plot()
        self.analyse_image()
        return

    def check_updated_img(self):
        self.filetime = os.path.getmtime(self.filename)
        if self.filetime != self.initfiletime:
            self.initfiletime = self.filetime
            self.image = imread(self.filename)
            self.analyse_image()
        return

    def update_plot(self):
        self.center_text.set_text('Center X:%.2f, Center Y:%.2f'%(self.centerX, self.centerY))
        self.defocus_text.set_text('Defocus X:%.2f, Defocus Y:%.2f'%(self.dfx, self.dfy))
        self.im.set_data(self.image)
        self.fig.canvas.draw()
        return

    def analyse_image(self):
        cx, cy, zx, l1x, l2x, zy, l1y, l2y, dfx, dfy = find_crosshair_center(self.img_gray)
        self.centerX = cx
        self.centerY = cy
        self.dfx = dfx
        self.dfy = dfy

        self.centerplot.set_xdata(self.centerX)
        self.centerplot.set_ydata(self.centerY)

        self.lx1.set_xdata(zx)
        self.lx1.set_ydata(l1x)
        self.lx2.set_xdata(zx)
        self.lx2.set_ydata(l2x)

        self.ly1.set_xdata(l1y)
        self.ly1.set_ydata(zy)
        self.ly2.set_xdata(l2y)
        self.ly2.set_ydata(zy)
        self.update_plot()
        return




if __name__ == '__main__':

    def on_closing(master):
        master.quit()
        master.destroy()
        return

    root = Tk()
    print("Created root")
    window = CrossCenterFinder(root)
    print("Initialized window")
    window.data_plot.get_tk_widget().grid(column=2, row=0, columnspan=4, rowspan=5)
    window.controlsFrame.grid(column=0, row=0, columnspan=2, rowspan=2)
    print("Created data plot")
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))  # make sure window close properly

    while True:
        try:
            root.mainloop()
            break
        except UnicodeDecodeError:
            pass
