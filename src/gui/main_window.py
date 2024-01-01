import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from src.gui import gui_utils as gu
import src.vtk_utils as vu
import time
import math
from src.config import *


class MainWindow(QtWidgets.QMainWindow, QtWidgets.QApplication):
    """
    Defines the main GUI class
    """

    # region Constructor
    def __init__(self,
                 app):
        """
        Constructor
        :param app: app
        """
        QtWidgets.QMainWindow.__init__(self, None)

        # Store the app
        self.app = app

        # Create the base UI components
        # 1. Qt objects set up
        # QFrame - provides a container for other widgets.
        self.frame = QtWidgets.QFrame()
        self.frame.setAutoFillBackground(True)

        # 2. Vtk set up
        # QVTKRenderWindowInteractor - provided by VTK for integration with the Qt framework.
        # Specifically, it is a Qt widget that enables the embedding of VTK rendering windows and interactors
        # within Qt applications.
        self.vtkWidget = QVTKRenderWindowInteractor()

        # vtkRenderWindowInteractor - responsible for connecting the vtkInteractorStyle with the rendering window.
        # It captures and processes user events (e.g., mouse clicks, key presses) and triggers the corresponding
        # actions defined by the current vtkInteractorStyle.
        self.vtkInteractor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # vtkRenderWindow - represents the rendering window or viewport in which the visualization scene is displayed.
        self.vtk_render_window = self.vtkWidget.GetRenderWindow()

        # vtkRenderer - manages the rendering of geometric and actor objects in a scene.
        self.vtk_renderer = vtk.vtkRenderer()

        # Add the renderer to the rendering window
        self.vtk_render_window.AddRenderer(self.vtk_renderer)

        # Set the interactor style
        # See: https://vtk.org/doc/nightly/html/classvtkInteractorStyleTrackballCamera.html
        self.vtkInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # 3. Initialize the volume/mask and other properties
        # MRI volume object
        self.volume = None
        # Segmentation mask object
        self.mask = None
        # Volume slicer properties list
        self.volume_slicer_props = []
        # Mask slicer properties list
        self.mask_slicer_props = []
        # Look up table list - corresponding to each color scheme
        self.luts = []

        # 4. GUI widgets
        # General
        self.rendering_menu = gu.create_combo_box([], True)
        # MRI Volume
        self.color_scheme_menu = gu.create_combo_box([], False)
        self.threshold_box = gu.create_picker(is_enabled=False)
        self.volume_opacity_box = gu.create_slider(is_enabled=False)
        self.smoothness_box = gu.create_slider(is_enabled=False)
        self.intensity_box = gu.create_slider(is_enabled=False)
        self.slicer_widgets = []
        # Segmentation mask
        self.mask_opacity_box = gu.create_slider(is_enabled=False)
        self.mask_labels = []
        # VTK View
        self.view_group_box = QtWidgets.QGroupBox()

        # 5. Set up the grid and its widgets
        self.grid = QtWidgets.QGridLayout()
        # Set up the grid
        self.add_widgets()

        # Flags:
        # Processing flag - whether there is already a threshold/smoothness change in progress
        self.processing = False
        # volume_loaded - whether the MRI is loaded
        self.volume_loaded = False
        # mask_loaded - whether the mask volume is loaded
        self.mask_loaded = False

        # Show the gui
        # Set the window title and icon
        self.setWindowTitle('3D NIfTI Viewer')
        self.setWindowIcon(QIcon(QPixmap(ICON_PATH)))
        # Set the layout and the central widget
        self.frame.setLayout(self.grid)
        self.setCentralWidget(self.frame)
        # Initialize the vtk Interactor
        self.vtkInteractor.Initialize()
        # Display the GUI
        self.show()
    # endregion

    # region Widgets
    def add_widgets(self):
        """
        Adds all the necessary widgets to the grid
        """
        # 1. Loading section
        load_group_box = QtWidgets.QGroupBox("Load files")
        load_group_layout = QtWidgets.QGridLayout()
        # Add volume button
        add_volume_button = QtWidgets.QPushButton("Volume")
        add_volume_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        add_volume_button.clicked.connect(self.add_volume)
        load_group_layout.addWidget(add_volume_button, 0, 0)
        # Add mask button
        add_mask_button = QtWidgets.QPushButton("Segmentation mask")
        add_mask_button.clicked.connect(self.add_mask)
        add_mask_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        load_group_layout.addWidget(add_mask_button, 0, 1)
        load_group_box.setLayout(load_group_layout)
        load_group_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.grid.addWidget(load_group_box, 0, 0, 1, 2)

        # 2. General settings section
        general_group_box = QtWidgets.QGroupBox("General settings")
        general_group_layout = QtWidgets.QGridLayout()
        # Add rendering modes (surface, slice)
        rendering_modes = ['Surface', 'Slice']
        # Rendering dropdown menu
        self.rendering_menu.addItems(rendering_modes)
        self.rendering_menu.currentIndexChanged.connect(self.render_mode_changed)
        self.rendering_menu.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        general_group_layout.addWidget(QtWidgets.QLabel("Render Mode"), 0, 0)
        general_group_layout.addWidget(self.rendering_menu, 0, 1)
        general_group_box.setLayout(general_group_layout)
        general_group_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.grid.addWidget(general_group_box, 1, 0, 1, 2)

        # 3. MRI volume section
        volume_group_box = QtWidgets.QGroupBox("MRI Volume settings")
        volume_group_layout = QtWidgets.QGridLayout()
        # Add color schemes
        volume_group_layout.addWidget(QtWidgets.QLabel("Color scheme"), 0, 0)
        color_schemes = ['Gray Scale', 'Rainbow (blue-red)', 'Rainbow (red-blue)', 'High contrast']
        self.color_scheme_menu.addItems(color_schemes)
        self.color_scheme_menu.currentIndexChanged.connect(self.color_scheme_changed)
        volume_group_layout.addWidget(self.color_scheme_menu, 0, 1)
        # Add threshold
        volume_group_layout.addWidget(QtWidgets.QLabel("Threshold"), 1, 0)
        self.threshold_box.valueChanged.connect(self.threshold_changed)
        volume_group_layout.addWidget(self.threshold_box, 1, 1)
        # Add opacity
        volume_group_layout.addWidget(QtWidgets.QLabel("Opacity"), 2, 0)
        self.volume_opacity_box.valueChanged.connect(self.volume_opacity_changed)
        volume_group_layout.addWidget(self.volume_opacity_box, 2, 1)
        # Add smoothness
        volume_group_layout.addWidget(QtWidgets.QLabel("Smoothness"), 3, 0)
        self.smoothness_box.valueChanged.connect(self.smoothness_changed)
        volume_group_layout.addWidget(self.smoothness_box, 3, 1)
        # Add intensity
        volume_group_layout.addWidget(QtWidgets.QLabel("Intensity"), 4, 0)
        self.intensity_box.valueChanged.connect(self.intensity_changed)
        volume_group_layout.addWidget(self.intensity_box, 4, 1)
        volume_group_box.setLayout(volume_group_layout)
        load_group_layout.addWidget(gu.create_separator(), 5, 1)
        self.grid.addWidget(volume_group_box, 2, 0, 1, 2)
        # Add slicing sliders
        slider_callbacks = [self.sagittal_slice_changed, self.axial_slice_changed, self.coronal_slice_changed]
        slider_titles = ['Sagittal Slice', 'Axial Slice', 'Coronal Slice']
        for i in range(3):
            slice_widget = gu.create_slider(callback=slider_callbacks[i],
                                            is_enabled=False)
            self.slicer_widgets.append(slice_widget)
            volume_group_layout.addWidget(QtWidgets.QLabel(slider_titles[i]), 6 + i, 0)
            volume_group_layout.addWidget(slice_widget, 6 + i, 1, 1, 2)

        # 4. Segmentation mask settings
        mask_group_box = QtWidgets.QGroupBox("Segmentation mask settings")
        mask_group_layout = QtWidgets.QGridLayout()
        # Add opacity
        mask_group_layout.addWidget(QtWidgets.QLabel("Opacity"), 0, 0)
        self.mask_opacity_box.valueChanged.connect(self.mask_opacity_changed)
        self.mask_opacity_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        mask_group_layout.addWidget(self.mask_opacity_box, 0, 1)
        mask_group_layout.addWidget(gu.create_separator(), 1, 0, 1, 2)
        # Add label checkboxes
        labels_area = QtWidgets.QScrollArea()
        labels_area.setWidgetResizable(True)
        labels_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        labels_area_content = QtWidgets.QGroupBox()
        labels_area_layout = QtWidgets.QGridLayout()
        c_row, c_col = 0, 0
        for i in range(1, 96):
            check_box = QtWidgets.QCheckBox(str(i))
            check_box.setEnabled(False)
            # Set size policy to prevent stretching
            check_box.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self.mask_labels.append(check_box)
            labels_area_layout.addWidget(self.mask_labels[i - 1], c_row, c_col)
            c_row = c_row + 1 if c_col == 3 else c_row
            c_col = 0 if c_col == 3 else c_col + 1
        labels_area_content.setLayout(labels_area_layout)
        labels_area.setWidget(labels_area_content)
        mask_group_layout.addWidget(labels_area, 2, 0, 1, 2)
        mask_group_box.setLayout(mask_group_layout)
        self.grid.addWidget(mask_group_box, 3, 0, 1, 2)

        # 5. View
        view_layout = QtWidgets.QGridLayout()
        axial_view_button = QtWidgets.QPushButton("Axial")
        axial_view_button.clicked.connect(self.set_axial_view)
        view_layout.addWidget(axial_view_button, 0, 0)
        coronal_view_button = QtWidgets.QPushButton("Coronal")
        coronal_view_button.clicked.connect(self.set_coronal_view)
        view_layout.addWidget(coronal_view_button, 0, 1)
        sagittal_view_button = QtWidgets.QPushButton("Sagittal")
        sagittal_view_button.clicked.connect(self.set_sagittal_view)
        view_layout.addWidget(sagittal_view_button, 0, 2)
        view_layout.addWidget(self.vtkWidget, 1, 0, 1, 3)
        self.view_group_box.setLayout(view_layout)
        self.grid.addWidget(self.view_group_box, 0, 2, 5, 5)
        self.grid.setColumnMinimumWidth(1, 200)
        self.grid.setColumnMinimumWidth(2, 700)
    # endregion

    # region Volume
    def add_volume(self):
        """
        Loads the MRI volume
        """
        # Browse for the NIfTI file
        selected_file = self.get_nifti_file()

        if selected_file:
            # Get the Volume
            self.volume = vu.setup_volume(selected_file, self.vtk_renderer)

            # Set up the slicer
            self.volume_slicer_props = vu.setup_slicer(self.vtk_renderer,
                                                       self.volume)

            # Set slicers' extent
            extent_index = 5
            for slice_widget in self.slicer_widgets:
                slice_widget.setRange(self.volume.extent[extent_index - 1], self.volume.extent[extent_index])
                slice_widget.setValue(self.volume.extent[extent_index] // 2)
                extent_index -= 2

            # Display surface/slicers depending on the current rendering mode:
            if self.rendering_menu.currentIndex() == 0:
                # Display the surface volume
                self.display_volume_surface()
            else:
                # Display slicers
                self.display_volume_slicers()

            # Setup volume settings
            self.setup_volume_settings()

            # Update settings
            self.update_volume_settings()

            # Set up the LUTS
            self.luts = vu.create_luts(self.volume)

            # Set volume flag up
            self.volume_loaded = True

            # Set coronal view and render
            self.set_coronal_view()

    def setup_volume_settings(self,
                              enabled: bool = True):
        """
        Sets up the volume settings
        :param enabled: Boolean parameter indicating whether the volume settings should be enabled or not.
        """
        # Enable the color scheme menu
        self.color_scheme_menu.setEnabled(enabled)
        # Enable and set up the threshold box
        gu.set_picker(self.threshold_box,
                      self.volume.scalar_range[0],
                      self.volume.scalar_range[1],
                      5,
                      sum(self.volume.scalar_range) / 2,
                      None)
        # Enable and set up the volume slider
        gu.set_slider(self.volume_opacity_box,
                      0,
                      100,
                      BRAIN_OPACITY,
                      None,
                      True)
        # Enable and set up the smoothness slider
        gu.set_slider(self.smoothness_box,
                      100,
                      1000,
                      BRAIN_SMOOTHNESS,
                      None)
        # Enable and set up the intensity slider
        gu.set_slider(self.intensity_box,
                      0,
                      300,
                      200,
                      None)

    def update_volume_settings(self):
        """
        Updates volume settings according to the rendering mode
        """
        # Get the current rendering mode: 0 - Surface, 1 - Slice
        rendering_mode = self.rendering_menu.currentIndex()

        # Update settings
        self.color_scheme_menu.setEnabled(rendering_mode)
        self.intensity_box.setEnabled(rendering_mode)
        self.threshold_box.setEnabled(not rendering_mode)
        self.smoothness_box.setEnabled(not rendering_mode)

    def display_volume_surface(self):
        """
        Displays the vtk volume actor
        """
        vu.display_actor(self.vtk_renderer,
                         self.volume.labels[0].actor)

    def hide_volume_surface(self):
        """
        Removes the vtk volume actor from the scene
        """
        vu.remove_actor(self.vtk_renderer,
                        self.volume.labels[0].actor)

    def display_volume_slicers(self):
        """
        Displays the volume slicer widgets
        """
        # Determine the opacity
        opacity = 0.5 if self.mask_loaded else 1.0

        # Display slicers
        vu.set_props_opacity(self.volume_slicer_props,
                             opacity)

    def hide_volume_slicers(self):
        """
        Hides the volume slicer widgets
        """
        vu.set_props_opacity(self.volume_slicer_props,
                             0.0)
    # endregion

    # region Mask
    def add_mask(self):
        """
        Loads the segmentation mask
        """
        # Browse for the NIfTI file
        selected_file = self.get_nifti_file()

        if selected_file:
            # Get the Volume
            self.mask = vu.setup_mask(selected_file, self.vtk_renderer)

            # Set up thr slicer:
            self.mask_slicer_props = vu.setup_slicer(self.vtk_renderer,
                                                     self.mask)

            # Display surface/slicers depending on the current rendering mode:
            if self.rendering_menu.currentIndex() == 0:
                # Display the mask surface
                self.display_mask_surface()
            else:
                # Display slicers
                self.display_mask_slicers()

            # Enable volume settings
            self.setup_mask_settings()

            # Set mask flag up
            self.mask_loaded = True

            # If volume is missing
            if not self.volume_loaded:
                # Set slicers' extent
                extent_index = 5
                for slice_widget in self.slicer_widgets:
                    slice_widget.setRange(self.mask.extent[extent_index - 1], self.mask.extent[extent_index])
                    slice_widget.setValue(self.mask.extent[extent_index] // 2)
                    extent_index -= 2

                # Set coronal view
                self.set_coronal_view()

    def setup_mask_settings(self,
                            enabled: bool = True):
        """
        Sets up the segmentation mask settings
        :param enabled: Boolean parameter indicating whether the volume settings should be enabled or not.
        """
        # Enable label checkboxes
        for i, cb in enumerate(self.mask_labels):
            if i < len(self.mask.labels) and self.mask.labels[i].actor:
                cb.setEnabled(True)
                cb.setChecked(True)
                cb.clicked.connect(self.mask_label_checked)
            else:
                cb.setDisabled(True)

        # Enable and set up the opacity slider
        gu.set_slider(self.mask_opacity_box,
                      0,
                      100,
                      MASK_OPACITY,
                      None,
                      enabled)

    def display_mask_surface(self):
        """
        Displays the segmentation mask rendered surface
        """
        for label in self.mask.labels:
            vu.display_actor(self.vtk_renderer,
                             label.actor)

    def hide_mask_surface(self):
        """
        Hides the segmentation mask rendered surface
        """
        vu.remove_actor(self.vtk_renderer,
                        self.volume.labels[0].actor)

    def display_mask_slicers(self):
        """
        Displays the mask's slicer widgets
        """
        # Determine the opacity
        opacity = 0.5 if self.volume_loaded else 1.0

        # Display slicers
        vu.set_props_opacity(self.mask_slicer_props,
                             opacity)

    def hide_mask_slicers(self):
        """
        Hides the mask's slicer widgets
        """
        vu.set_props_opacity(self.mask_slicer_props,
                             0.0)

    # endregion

    # region Callbacks
    def render_mode_changed(self, index):
        """
        Rendering mode changed callback
        :param index: the selected rendering mode
                      (0 - Surface,
                      1- Slice)
        """
        # Enable/Disable the slicing widgets
        if self.volume_loaded or self.mask_loaded:
            for widget in self.slicer_widgets:
                widget.setEnabled(index)

        # Get the opacity
        if index == 0:
            opacity = 0
        elif self.volume_loaded and self.mask_loaded:
            opacity = 0.5
        elif self.volume_loaded or self.mask_loaded:
            opacity = 1
        else:
            opacity = 0

        if self.volume_loaded:
            # Hide/show the surface volume
            if index == 0:
                self.vtk_renderer.AddActor(self.volume.labels[0].actor)
            else:
                self.vtk_renderer.RemoveActor(self.volume.labels[0].actor)

            # Set volume slicer opacity
            for prop in self.volume_slicer_props:
                prop.GetProperty().SetOpacity(opacity)

        if self.mask_loaded:
            # Hide/show the surface volume
            for label_idx in range(len(MASK_COLORS)):
                if index == 0:
                    self.vtk_renderer.AddActor(self.mask.labels[label_idx].actor)
                else:
                    self.vtk_renderer.RemoveActor(self.mask.labels[label_idx].actor)

            # Set volume slicer opacity
            for prop in self.mask_slicer_props:
                prop.GetProperty().SetOpacity(opacity)

        # Update settings
        if self.volume_loaded:
            self.update_volume_settings()

        # Render the window
        self.vtk_render_window.Render()

    def color_scheme_changed(self, index):
        """
        Color scheme changed callback
        :param index: the selected color scheme
        """
        if self.rendering_menu.currentIndex() == 1:
            # If slice mode => change lut
            self.volume.image_mapper.SetLookupTable(self.luts[index])
            self.volume.image_mapper.Update()
            self.vtk_render_window.Render()

    def threshold_changed(self, value):
        """
        Threshold changed callback
        :param value: threshold level
        """
        self.process_changes()
        self.volume.labels[0].extractor.SetValue(0, value)
        self.vtk_renderer.Render()
        self.processing = False

    def volume_opacity_changed(self, value):
        """
        Volume changed callback
        :param value: opacity level in percentages (0 - 100)
        """
        # Bring the value to [0, 1]
        opacity = round(value / 100, 2)

        if self.rendering_menu.currentIndex() == 0:
            # Surface rendering mode
            self.volume.labels[0].property.SetOpacity(opacity)
        else:
            # Slice rendering mode
            for prop in self.volume_slicer_props:
                prop.GetProperty().SetOpacity(opacity)
        self.vtk_render_window.Render()

    def smoothness_changed(self, value):
        """
        Smoothness changed callback
        :param value: smoothness value
        """
        self.process_changes()
        self.volume.labels[0].smoother.SetNumberOfIterations(value)
        self.vtk_render_window.Render()

    def intensity_changed(self, value):
        """
        Intensity changed callback
        :param value: intensity value in percentages (0 - 100)
        """
        # Get the current lookup table
        lut = self.volume.image_mapper.GetLookupTable()

        # Bring the newvalue to [0, 1]
        new_lut_value = round(value / 100, 2)

        # Set lut's value range and update
        lut.SetValueRange(0.0, new_lut_value)
        lut.Build()
        self.volume.image_mapper.SetLookupTable(lut)
        self.volume.image_mapper.Update()
        self.vtk_render_window.Render()

    def mask_opacity_changed(self, value):
        """
        Mask opacity changed callback
        :param value: opacity level in percentages (0 - 100)
        """
        # Compute opacity (ranges between 0 - 1)
        opacity = round(value / 100, 2)

        if self.rendering_menu.currentIndex() == 0:
            # Surface rendering mode
            for i, label in enumerate(self.mask.labels):
                if label.property and self.mask_labels[i].isChecked():
                    label.property.SetOpacity(opacity)
        else:
            # Slice rendering mode
            for prop in self.mask_slicer_props:
                prop.GetProperty().SetOpacity(opacity)
        self.vtk_render_window.Render()

    def mask_label_checked(self):
        """
        Mask label checkbox checked callback
        """
        # Get the LookupTable
        lut = self.mask.image_mapper.GetLookupTable()

        # Color list
        color_list = list(MASK_COLORS.items())

        for i, cb in enumerate(self.mask_labels):
            if cb.isChecked():
                # Set the surface opacity
                self.mask.labels[i].property.SetOpacity(self.mask_opacity_box.value())

                # Add color value in lut
                r, g, b = color_list[i][1]
                lut.SetTableValue(color_list[i][0], r, g, b, 1.0)
            elif cb.isEnabled():
                # Set the surface opacity to 0
                self.mask.labels[i].property.SetOpacity(0)

                # Change the color value from lut
                lut.SetTableValue(color_list[i][0], 0.0, 0.0, 0.0, 0.0)

        # Build and update
        lut.Build()
        self.mask.image_mapper.SetLookupTable(lut)
        self.mask.image_mapper.Update()
        self.vtk_render_window.Render()

    def sagittal_slice_changed(self, value):
        """
        Sagittal slice changed callback
        :param value: the selected slice
        """
        # Change the volume slicer widget's extent
        if len(self.volume_slicer_props) > 0:
            self.volume_slicer_props[0].SetDisplayExtent(self.volume.extent[0], self.volume.extent[1],
                                                         self.volume.extent[2], self.volume.extent[3], value, value)
        # Change the mask slicer widget's extent
        if len(self.mask_slicer_props) != 0:
            self.mask_slicer_props[0].SetDisplayExtent(self.mask.extent[0], self.mask.extent[1],
                                                       self.mask.extent[2], self.mask.extent[3], value, value)
        self.vtk_render_window.Render()

    def axial_slice_changed(self, value):
        """
        Axial slice changed callback
        :param value: the selected slice
        """
        # Change the volume slicer widget's extent
        if len(self.volume_slicer_props) > 0:
            self.volume_slicer_props[1].SetDisplayExtent(self.volume.extent[0], self.volume.extent[1], value, value,
                                                         self.volume.extent[4], self.volume.extent[5])
        # Change the mask slicer widget's extent
        if len(self.mask_slicer_props) != 0:
            self.mask_slicer_props[1].SetDisplayExtent(self.mask.extent[0], self.mask.extent[1], value, value,
                                                       self.mask.extent[4], self.mask.extent[5])
        self.vtk_render_window.Render()

    def coronal_slice_changed(self, value):
        """
        Coronal slice changed callback
        :param value: the selected slice
        """
        # Change the volume slicer widget's extent
        if len(self.volume_slicer_props) > 0:
            self.volume_slicer_props[2].SetDisplayExtent(value, value, self.volume.extent[2], self.volume.extent[3],
                                                         self.volume.extent[4], self.volume.extent[5])
        # Change the mask slicer widget's extent
        if len(self.mask_slicer_props) != 0:
            self.mask_slicer_props[2].SetDisplayExtent(value, value, self.mask.extent[2], self.mask.extent[3],
                                                       self.mask.extent[4], self.mask.extent[5])
        self.vtk_render_window.Render()

    def process_changes(self):
        """
        Processes threshold/smoothness changes avoiding blocking and excessive calls
        """
        if not self.processing:
            self.processing = True
            for _ in range(10):
                self.app.processEvents()
                time.sleep(0.1)
    # endregion

    # region View orientation
    def set_axial_view(self):
        """
        Sets the axial view
        """
        self.vtk_renderer.ResetCamera()
        fp = self.vtk_renderer.GetActiveCamera().GetFocalPoint()
        p = self.vtk_renderer.GetActiveCamera().GetPosition()
        dist = math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        # self.vtk_renderer.GetActiveCamera().SetPosition(fp[0], fp[1], fp[2] + dist)
        self.vtk_renderer.GetActiveCamera().SetPosition(fp[0], fp[2] + dist, fp[1])
        self.vtk_renderer.GetActiveCamera().SetViewUp(0.0, 0.0, 0.0)
        self.vtk_renderer.GetActiveCamera().Zoom(1.6)
        self.vtk_render_window.Render()

    def set_coronal_view(self):
        """
        Sets the coronal view
        """
        self.vtk_renderer.ResetCamera()
        fp = self.vtk_renderer.GetActiveCamera().GetFocalPoint()
        p = self.vtk_renderer.GetActiveCamera().GetPosition()
        dist = math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        # self.vtk_renderer.GetActiveCamera().SetPosition(fp[0], fp[2] - dist, fp[1])
        self.vtk_renderer.GetActiveCamera().SetPosition(fp[2] + dist, fp[0], fp[1])
        self.vtk_renderer.GetActiveCamera().SetViewUp(0.0, 0.0, 0.0)
        self.vtk_renderer.GetActiveCamera().Zoom(1.6)
        self.vtk_render_window.Render()

    def set_sagittal_view(self):
        """
        Sets the sagittal view
        """
        self.vtk_renderer.ResetCamera()
        fp = self.vtk_renderer.GetActiveCamera().GetFocalPoint()
        p = self.vtk_renderer.GetActiveCamera().GetPosition()
        dist = math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        # self.vtk_renderer.GetActiveCamera().SetPosition(fp[2] + dist, fp[0], fp[1])
        self.vtk_renderer.GetActiveCamera().SetPosition(fp[0], fp[1], fp[2] + dist)
        self.vtk_renderer.GetActiveCamera().SetViewUp(0.0, 0.0, 0.0)
        self.vtk_renderer.GetActiveCamera().Zoom(1.6)
        self.vtk_render_window.Render()
    # endregion

    # region NIfTI files
    def get_nifti_file(self):
        """
        Browses and returns the NIfTI file name
        """
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setNameFilter("NIfTI files (*.nii *.nii.gz);;All Files (*)")

        selected_file, _ = file_dialog.getOpenFileName(self, caption='Open file', directory='',
                                                       filter='NIfTI files (*.nii *.nii.gz);;All Files (*)')
        return selected_file
    # endregion
