import numpy as np
import napari

import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
from qtpy.QtCore import Signal

from napari_properties_plotter.property_plotter import (
    LayerSelector, VariablePicker, DataSelector, BinningSpinbox, PyQtGraphWrapper
)


class SingleDataSelector(DataSelector): 
    new_selection = Signal(float)

    def toggle_selection(self, activated):
        if activated:
            self.toggle.setText('Stop Selecting')
            sele = pg.InfiniteLine(movable=True)
            sele.sigPositionChangeFinished.connect(self.on_selection_changed)
            self.sele = sele
            self.plot.addItem(sele)
        else:
            self.toggle.setText('Select Area')
            if self.sele is not None:
                self.sele.sigRegionChangeFinished.disconnect()
                self.plot.removeItem(self.sele)
                self.sele = None
            self.on_selection_changed(None)
    
    def on_selection_changed(self, line_changed):
        if line_changed is None:
            self.abort_selection.emit()
        else:
            self.new_selection.emit(line_changed.value())


class PropertyPlotter(QWidget):
   # MODIFIED from: napari_properties_plotter
    """
    Napari widget that plots layer properties
    """
    def __init__(self, viewer: napari.Viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = viewer

        # buttons
        self.setLayout(QHBoxLayout())
        self.left = QVBoxLayout()
        self.layout().addLayout(self.left)

        self.left.addWidget(QLabel('Layer:'))
        self.layer_selector = LayerSelector(self.viewer.layers)
        self.left.addWidget(self.layer_selector)

        self.picker = VariablePicker(self)
        self.left.addWidget(self.picker)

        # plot
        self.plot = PyQtGraphWrapper(self)
        self.layout().addWidget(self.plot)

        self.data_selector = SingleDataSelector(self.plot.plotter, self)
        self.left.addWidget(self.data_selector)

        self.binning_spinbox = BinningSpinbox(self.plot, self)
        self.left.addWidget(self.binning_spinbox)

        # events
        self.layer_selector.changed.connect(self.on_layer_changed)

        self.plot.continuous.connect(self.data_selector.toggle_enabled)
        self.plot.binned.connect(self.binning_spinbox.setVisible)

        self.picker.changed.connect(self.plot.update)
        self.picker.removed.connect(self.plot.remove)
        self.picker.x_changed.connect(self.plot.set_x)

        self.data_selector.new_selection.connect(self.on_selection_changed)
        self.data_selector.abort_selection.connect(self.on_selection_changed)

        # trigger first time
        self.on_layer_changed(self.layer_selector.layer)

    def on_layer_changed(self, layer):
        # disable selection
        self.data_selector.toggle.setChecked(False)
        if isinstance(layer, napari.layers.Tracks):
            self.data_selector.setVisible(True)
        else:
            self.data_selector.setVisible(False)
        properties = getattr(layer, 'properties', {})
        self.picker.set_dataframe(properties)

    def on_selection_changed(self, value=None):
        pass
   