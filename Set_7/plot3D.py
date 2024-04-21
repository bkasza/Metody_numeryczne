import sys
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout,
                             QLabel, QSlider, QVBoxLayout, QWidget)
from PyQt6.QtGui import QAction

class ApplicationWindow(QMainWindow):
    def __init__(self,X,Y,Z):
        super().__init__()

        # Central widget
        self._main = QWidget()
        self.setCentralWidget(self._main)

        # Main menu bar
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        
        exit = QAction("Exit", self)
        exit.triggered.connect(self.close)
        self.menu_file.addAction(exit)

        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About Qt", self)
        about.triggered.connect(QApplication.aboutQt)
        self.menu_about.addAction(about)

        # Figure (Left)
        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)

        # Sliders (Left)
        min = 0
        max = 360
        self.slider_azim = QSlider(Qt.Orientation.Horizontal)
        self.slider_azim.setMinimum(min)
        self.slider_azim.setMaximum(max)

        self.slider_elev = QSlider(Qt.Orientation.Horizontal)
        self.slider_elev.setMinimum(min)
        self.slider_elev.setMaximum(max)

        self.slider_azim_layout = QHBoxLayout()
        self.slider_azim_layout.addWidget(QLabel(f"{min}"))
        self.slider_azim_layout.addWidget(self.slider_azim)
        self.slider_azim_layout.addWidget(QLabel(f"{max}"))

        self.slider_elev_layout = QHBoxLayout()
        self.slider_elev_layout.addWidget(QLabel(f"{min}"))
        self.slider_elev_layout.addWidget(self.slider_elev)
        self.slider_elev_layout.addWidget(QLabel(f"{max}"))

        # Left layout
        llayout = QVBoxLayout()
        llayout.addWidget(self.canvas, 88)
        llayout.addWidget(QLabel("Azimuth:"), 1)
        llayout.addLayout(self.slider_azim_layout, 5)
        llayout.addWidget(QLabel("Elevation:"), 1)
        llayout.addLayout(self.slider_elev_layout, 5)

        # Main layout
        layout = QHBoxLayout(self._main)
        layout.addLayout(llayout)

        # Signal and Slots connections
        self.slider_azim.valueChanged.connect(self.rotate_azim)
        self.slider_elev.valueChanged.connect(self.rotate_elev)

        # Initial setup
        self.plot_surface(X,Y,Z)
        self._ax.view_init(30, 30)
        self.slider_azim.setValue(30)
        self.slider_elev.setValue(30)
        self.fig.canvas.mpl_connect("button_release_event", self.on_click)

    def on_click(self, event):
        azim, elev = self._ax.azim, self._ax.elev
        self.slider_azim.setValue(azim + 180)
        self.slider_elev.setValue(elev + 180)

    def set_canvas_configuration(self):
        self.fig.set_canvas(self.canvas)
        self._ax = self.canvas.figure.add_subplot(projection="3d")
        self._ax.set_xlabel("U_A")
        self._ax.set_ylabel("U_B")
        self._ax.set_zlabel("Alice Payoff")

    def plot_surface(self, X, Y, Z):
        # X, Y, Z = axes3d.get_test_data(0.03)
        # self.set_canvas_configuration()
        # self._ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, cmap="viridis")
        # self._ax.set_xlim(left=np.min(X), right=np.max(X))  
        # self._ax.set_ylim(bottom=np.min(Y), top=np.max(Y))
        # self._ax.set_zlim(bottom=np.min(Z), top=np.max(Z))
        # self.canvas.draw()
        self.set_canvas_configuration()
        self._ax.plot_surface(X, Y, Z, cmap="viridis")
        self._ax.set_xlim(left=np.min(X), right=np.max(X))  
        self._ax.set_ylim(bottom=np.min(Y), top=np.max(Y))
        self._ax.set_zlim(bottom=np.min(Z), top=np.max(Z))
        self.canvas.draw()


    def rotate_azim(self, value):
        self._ax.view_init(self._ax.elev, value)
        self.fig.set_canvas(self.canvas)
        self.canvas.draw()

    def rotate_elev(self, value):
        self._ax.view_init(value, self._ax.azim)
        self.fig.set_canvas(self.canvas)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ApplicationWindow()
    w.setFixedSize(1280, 720)
    w.show()
    sys.exit(app.exec())
