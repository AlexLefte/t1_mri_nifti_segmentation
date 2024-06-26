import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QObject, QTimer
from PyQt5.QtGui import QMovie
import nibabel
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from src.model.src.inference import main as segment_main
import vtk

from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction, vtkVolumeProperty, vtkVolume
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from src.nii_utils import *

from src.gui import gui_utils as gu
import src.vtk_utils as vu
import time
import math
from src.config import *
import os
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper


class DialogWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Perform Segmentation?")
        self.setStyleSheet("background-color: white;")

        # Setează layout-ul principal pentru dialog
        self.layout = QVBoxLayout()
        self.label = QLabel("Would you also like to perform segmentation now?", self)
        self.layout.addWidget(self.label)

        # Setează layout-ul pentru butoane
        self.buttonLayout = QtWidgets.QHBoxLayout()  # Layout orizontal pentru centrarea butoanelor

        # Adaugă butoanele de confirmare și anulare
        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Yes | QtWidgets.QDialogButtonBox.No,
                                                    self)
        yes_button = self.buttonBox.button(QtWidgets.QDialogButtonBox.Yes)
        no_button = self.buttonBox.button(QtWidgets.QDialogButtonBox.No)
        button_style = """
                    QPushButton {
                       background-color: #3b9ebf;
                       color: white;
                       border-radius: 1px;
                       padding: 8px 25px 8px 25px;
                       margin: 10px 5px 10px 0px
                    }
                    QPushButton:hover {
                       background-color: #ffffff;
                       color: #3b9ebf;
                       border: 1px solid #3b9ebf;
                    }
                """
        yes_button.setStyleSheet(button_style)
        no_button.setStyleSheet(button_style)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject_by_button)

        self.buttonLayout.addStretch(1)  # Adaugă un stretch pentru a împinge butoanele la centru
        self.buttonLayout.addWidget(self.buttonBox)
        self.buttonLayout.addStretch(1)  # Adaugă un stretch pentru a menține butoanele la centru

        # Adaugă layout-ul orizontal la layout-ul vertical principal
        self.layout.addLayout(self.buttonLayout)

        self.layout.setContentsMargins(15, 15, 15, 15)
        self.setLayout(self.layout)
        self.cancel_pressed = False

    def reject_by_button(self):
        # Setează starea cancel ca fiind apăsată și apelează reject
        self.cancel_pressed = True
        self.reject()


class ModalWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Processing Status")

        self.setStyleSheet("background-color: white;")
        self.layout = QVBoxLayout()
        self.status_label = None
        self.cancel_button = None
        self.setMinimumSize(250, 100)
        self.init_ui()
        self.set_geometry()

    def init_ui(self):
        self.layout = QVBoxLayout(self)

        # Add status label
        self.status_label = QLabel("MRI segmentation started.\nThis process may take up to several minutes...", self)
        self.status_label.setWordWrap(True)
        self.layout.addWidget(self.status_label)

        self.layout.setContentsMargins(15, 15, 15, 15)
        self.setLayout(self.layout)

    def update_message(self, message):
        self.status_label.setText(message)
        # self.adjustSize()

    def set_geometry(self):
        if self.parent():
            parent_geometry = self.parent().geometry()
            self.move(
                parent_geometry.x() + (parent_geometry.width() - self.width()) // 2,
                parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            )

    def showEvent(self, event):
        super().showEvent(event)
        self.setFixedSize(self.size())

    def closeEvent(self, event):
        self.parent().close_modal()
        super().closeEvent(event)


class Worker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, file_path="",
                 output_path="",
                 device="",
                 config_path="",
                 lut_path=""):
        super().__init__()
        self.file_path = file_path
        self.output_path = output_path
        self.device = device
        self.config_path = config_path
        self.lut_path = lut_path

    def run_segment(self):
        try:
            # Start recording the segmentation time
            start_time = time.time()
            # Call the segmentation function
            segment_main(self.file_path,
                         labels_path=None,
                         output_path=self.output_path,
                         device=self.device,
                         config_path=self.config_path,
                         lut_path=self.lut_path)
            elapsed_time = round((time.time() - start_time) / 60, 2)
            self.finished.emit(elapsed_time)
        except Exception as e:
            self.error.emit(str(e))

    def run_volume_setup(self, standard_image, renderer):
        try:
            volume = vu.setup_volume(standard_image, renderer)
            self.finished.emit(volume)
        except Exception as e:
            self.error.emit(str(e))

    def run_mask_setup(self, segmentation_image, renderer):
        try:
            mask = vu.setup_mask(segmentation_image, renderer)
            self.finished.emit(mask)
        except Exception as e:
            self.error.emit(str(e))


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

        # Set a custom font for the entire application
        font = QtGui.QFont("Segoe UI", 10, QtGui.QFont.Normal, False)
        app.setFont(font)

        # Set a stylesheet for the entire application
        self.app.setStyleSheet("""
                   QMainWindow {
                       background-color: #ffffff;
                   }
                   QPushButton {
                       background-color: #3b9ebf;
                       color: white;
                       border-radius: 1px;
                       padding: 5px;
                   }
                   QPushButton:hover {
                       background-color: #ffffff;
                       color: #3b9ebf;
                       border: 2px solid #3b9ebf;
                   }
                   QFrame {
                       background-color: #ffffff;
                   }
                   
                   QDoubleSpinBox
                    {
                    border : 2px solid #3b9ebf;
                    background : white;
                    }
                    QDoubleSpinBox::hover
                    {
                    border : 2px solid #000080;
                    background : #d3d3d3;
                    }
                    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                    subcontrol-origin: border;
                    subcontrol-position: top right;
                    }
                    
                    QDoubleSpinBox::up-button {
                        width: 16px;  /* Width of the up-button */
                        height: 16px; /* Height of the up-button, ensuring it's square */
                        border-bottom: none;
                    }
                    
                    QDoubleSpinBox::down-button {
                        width: 16px;
                        height: 16px;
                        border-top: none;
                    }
                    
                    QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
                        width: 10px;
                        height: 10px;
                    }
                    
                    QDoubleSpinBox::up-arrow {
                        image: url('images/arrow_up.png');
                    }
                    
                    QComboBox {
                        border: 1px solid #bdc3c7;
                        border-radius: 4px;
                        padding: 5px;
                        background: white;
                        min-width: 150px;
                    }
                    QComboBox:hover {
                        border-color: #3498db;
                    }
                    QComboBox::drop-down {
                        border-image: none;  /* Nu afișează nicio imagine */
                        width: 0px;
                    }
                    QComboBox QAbstractItemView {
                        border: 2px solid #3498db;
                        selection-background-color: #bdc3c7;
                        color: #2c3e50;
                        background-color: white;
                    }
                    """)

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
        self.color_scheme_label = QtWidgets.QLabel("Color scheme")
        self.threshold_box = gu.create_picker(is_enabled=False)
        self.threshold_button = gu.create_button('Set Threshold')
        self.threshold_label = QtWidgets.QLabel("Threshold")
        self.volume_opacity_box = gu.create_slider(is_enabled=False)
        self.opacity_label = QtWidgets.QLabel("Opacity")
        self.smoothness_box = gu.create_slider(is_enabled=False)
        # self.intensity_box = gu.create_slider(is_enabled=False)
        # self.intensity_label = QtWidgets.QLabel("Intensity")
        self.slicer_widgets = []
        self.slicer_label_widgets = []
        self.display_all_masks = False
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

        self.thread = None
        self.worker = None
        self.modal = None
        self.volume_lambda = None
        self.mask_lambda = None
        self.modal_closed = False
        self.segment = True

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

    # region Modal Window
    def close_modal(self):
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        self.modal_closed = True

    # endregion

    # region Widgets
    def add_widgets(self):
        """
        Adds all the necessary widgets to the grid
        """
        # 1. Loading section
        load_group_box = QtWidgets.QGroupBox()
        load_group_layout = QtWidgets.QGridLayout()
        # Add volume button
        add_volume_button = QtWidgets.QPushButton("Load MRI")
        add_volume_button.setFont(QFont("Segoe UI", 10, QtGui.QFont.Bold, False))
        add_volume_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        add_volume_button.clicked.connect(self.load_segment_volume)
        load_group_layout.addWidget(add_volume_button, 0, 0)
        # Add mask button
        add_mask_button = QtWidgets.QPushButton("Segmentation mask")
        add_mask_button.setFont(QFont("Segoe UI", 10, QtGui.QFont.Bold, False))
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
        rendering_modes = ['Slice', 'Surface']
        # Rendering dropdown menu
        self.rendering_menu.addItems(rendering_modes)
        self.rendering_menu.currentIndexChanged.connect(self.render_mode_changed)
        self.rendering_menu.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        general_group_layout.addWidget(QtWidgets.QLabel("Render Mode"), 0, 0)
        general_group_layout.addWidget(self.rendering_menu, 0, 1)
        general_group_box.setLayout(general_group_layout)
        general_group_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.grid.addWidget(general_group_box, 1, 0, 1, 2)

        # 3. Section widgets
        volume_group_box = QtWidgets.QGroupBox("MRI Volume settings")
        volume_group_layout = QtWidgets.QGridLayout()

        # Add opacity
        self.volume_opacity_box.valueChanged.connect(self.volume_opacity_changed)
        volume_group_layout.addWidget(self.opacity_label, 1, 0)
        volume_group_layout.addWidget(self.volume_opacity_box, 1, 1, 1, 2)

        # Set up the color scheme menu
        # color_schemes = ['Gray Scale', 'Rainbow (blue-red)', 'Rainbow (red-blue)', 'High contrast']
        color_schemes = ['Gray Scale', 'Rainbow (blue-red)', 'Rainbow (red-blue)']
        self.color_scheme_menu.addItems(color_schemes)
        self.color_scheme_menu.currentIndexChanged.connect(self.color_scheme_changed)
        volume_group_layout.addWidget(self.color_scheme_label, 0, 0)
        volume_group_layout.addWidget(self.color_scheme_menu, 0, 1)

        # # Add smoothness
        # volume_group_layout.addWidget(QtWidgets.QLabel("Smoothness"), 3, 0)
        # self.smoothness_box.valueChanged.connect(self.smoothness_changed)
        # volume_group_layout.addWidget(self.smoothness_box, 3, 1)

        # Add intensity
        # self.intensity_box.valueChanged.connect(self.intensity_changed)
        # volume_group_layout.addWidget(self.intensity_label, 2, 0)
        # volume_group_layout.addWidget(self.intensity_box, 2, 1)

        # Add slicing sliders
        slider_titles = ['Axial Slice', 'Coronal Slice', 'Sagittal Slice']
        for i in range(3):
            # slice_widget = gu.create_slider(callback=slider_callbacks[i],
            #                                 is_enabled=False)
            slice_widget = gu.create_slider(callback=None,
                                            is_enabled=False)
            self.slicer_widgets.append(slice_widget)
            self.slicer_label_widgets.append(QtWidgets.QLabel(slider_titles[i]))
            volume_group_layout.addWidget(self.slicer_label_widgets[i], 2 + i, 0)
            volume_group_layout.addWidget(self.slicer_widgets[i], 2 + i, 1, 1, 2)

        # Add surface-related controls
        # self.threshold_box.valueChanged.connect(self.threshold_changed)
        self.threshold_button.clicked.connect(self.change_volume_threshold)
        volume_group_layout.addWidget(self.threshold_label, 2, 0)
        volume_group_layout.addWidget(self.threshold_box, 2, 1)
        volume_group_layout.addWidget(self.threshold_button, 2, 2)
        volume_group_box.setLayout(volume_group_layout)
        self.grid.addWidget(volume_group_box, 2, 0, 1, 2)

        # Set widget visibility according to the initial rendering type (slice)
        self.set_widget_visibility(0)

        # 4. Segmentation mask settings
        mask_group_box = QtWidgets.QGroupBox("Segmentation mask settings")
        mask_group_layout = QtWidgets.QGridLayout()
        # Add opacity
        mask_group_layout.addWidget(QtWidgets.QLabel("Opacity"), 0, 0)
        self.mask_opacity_box.valueChanged.connect(self.mask_opacity_changed)
        self.mask_opacity_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        mask_group_layout.addWidget(self.mask_opacity_box, 0, 1)
        spacer = QtWidgets.QSpacerItem(5, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        mask_group_layout.addItem(spacer, 1, 0, 1, 2)
        # mask_group_layout.addWidget(gu.create_separator(), 1, 0, 1, 2)
        # Add label checkboxes
        labels_area = QtWidgets.QScrollArea()
        labels_area.setWidgetResizable(True)
        labels_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        labels_area_content = QtWidgets.QGroupBox()
        labels_area_layout = QtWidgets.QGridLayout()
        c_row, c_col = 0, 0
        for i in range(96):
            if i == 0:
                check_box = QtWidgets.QCheckBox('All')
            else:
                check_box = QtWidgets.QCheckBox(str(i))
            check_box.setEnabled(False)
            # Set size policy to prevent stretching
            check_box.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self.mask_labels.append(check_box)
            labels_area_layout.addWidget(self.mask_labels[i], c_row, c_col)
            c_row = c_row + 1 if c_col == 4 else c_row
            c_col = 0 if c_col == 4 else c_col + 1
        labels_area_content.setLayout(labels_area_layout)
        labels_area.setWidget(labels_area_content)
        mask_group_layout.addWidget(labels_area, 2, 0, 1, 2)
        mask_group_box.setLayout(mask_group_layout)
        self.grid.addWidget(mask_group_box, 3, 0, 1, 2)

        # 5. View
        view_layout = QtWidgets.QGridLayout()
        axial_view_button = QtWidgets.QPushButton("Axial")
        axial_view_button.setFont(QFont("Segoe UI", 10, QtGui.QFont.Bold, False))
        axial_view_button.clicked.connect(self.set_axial_view)
        view_layout.addWidget(axial_view_button, 0, 0)
        coronal_view_button = QtWidgets.QPushButton("Coronal")
        coronal_view_button.setFont(QFont("Segoe UI", 10, QtGui.QFont.Bold, False))
        coronal_view_button.clicked.connect(self.set_coronal_view)
        view_layout.addWidget(coronal_view_button, 0, 1)
        sagittal_view_button = QtWidgets.QPushButton("Sagittal")
        sagittal_view_button.setFont(QFont("Segoe UI", 10, QtGui.QFont.Bold, False))
        sagittal_view_button.clicked.connect(self.set_sagittal_view)
        view_layout.addWidget(sagittal_view_button, 0, 2)
        view_layout.addWidget(self.vtkWidget, 1, 0, 1, 3)
        self.view_group_box.setLayout(view_layout)
        self.grid.addWidget(self.view_group_box, 0, 2, 5, 5)
        self.grid.setColumnMinimumWidth(1, 200)
        self.grid.setColumnMinimumWidth(2, 700)

    def set_widget_visibility(self, mode):
        """
        Adds widgets based on the rendering mode
        :return:
        """
        # Set color scheme widgets visibility
        self.color_scheme_label.setVisible(not mode)
        self.color_scheme_menu.setVisible(not mode)

        # Set intensity widgets visibility
        # self.intensity_label.setVisible(not mode)
        # self.intensity_box.setVisible(not mode)

        # Set slicing sliders visibility
        for i in range(3):
            self.slicer_label_widgets[i].setVisible(not mode)
            self.slicer_widgets[i].setVisible(not mode)

        # Set surface-related widgets visibility
        self.threshold_label.setVisible(mode)
        self.threshold_box.setVisible(mode)
        self.threshold_button.setVisible(mode)

    # endregion

    # region Volume
    def load_segment_volume(self):
        # Ask whether segmentation is needed
        dialog = DialogWindow(self)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            self.segment = True
        elif dialog.cancel_pressed:
            self.segment = False
        else:
            return

        # Browse for the NIfTI file
        selected_file = self.get_nifti_file()

        if selected_file:
            # Clear the workspace
            if self.volume is not None:
                self.clear_volume()
        else:
            return

        if self.segment:
            # Get the Volume
            self.modal = ModalWindow(self)
            self.modal.show()
            self.modal.update_message("MRI segmentation started.\nThis process may take several minutes...")

            # Perform the segmentation and save the results
            if self.thread is None:
                self.thread = QThread()
            self.worker = Worker(file_path=selected_file,
                                 output_path='output/',
                                 device='cuda',
                                 config_path='src/model/config/config.json',
                                 lut_path='src/model/config/FastSurfer_ColorLUT.tsv')
            self.worker.moveToThread(self.thread)
            self.worker.finished.connect(self.onFinishedSegment)
            self.worker.error.connect(self.onError)
            self.thread.started.connect(self.worker.run_segment)
            self.thread.start()
        else:
            # Get the Volume
            self.modal = ModalWindow(self)
            self.modal.show()
            self.modal.update_message("Loading the MRI image...")
            time.sleep(1)

            # Just load the volume
            if self.thread is None:
                self.thread = QThread()
            if self.worker is None:
                self.worker = Worker()
            self.worker.moveToThread(self.thread)
            self.worker.finished.connect(self.onFinishedVolumeSetup)
            self.volume_lambda = lambda: self.worker.run_volume_setup(selected_file,
                                                                      self.vtk_renderer)
            self.thread.started.connect(self.volume_lambda)
            if not self.thread.isRunning():
                self.thread.start()
            else:
                print("Could not start the new thread.")

    def onFinishedSegment(self, time_taken):
        if self.modal_closed:
            self.modal.hide()
            self.modal_closed = False
            return

        minutes = int(time_taken)
        seconds = (time_taken - minutes) * 60

        # Format the string to include minutes and seconds
        formatted_time = f"{minutes} minutes {seconds:.0f} seconds"

        # Update your message
        self.modal.update_message(f"Segmentation performed successfully.\n"
                                  f"Total time taken: {formatted_time}.\n\n"
                                  f"Loading the MRI image...")

        self.worker.finished.disconnect(self.onFinishedSegment)
        self.thread.started.disconnect(self.worker.run_segment)
        self.thread.quit()
        self.thread.wait()

        # Check if standard image is present and load it
        standard_image = 'output/standard_image.nii'
        if not os.path.isfile(standard_image):
            self.modal.hide()
            print("Could not find the images.")
            return

        self.worker.finished.connect(self.onFinishedVolumeSetup)
        self.volume_lambda = lambda: self.worker.run_volume_setup(standard_image,
                                                                  self.vtk_renderer)
        self.thread.started.connect(self.volume_lambda)
        if not self.thread.isRunning():
            self.thread.start()
        else:
            print("Could not start the new thread.")

    def onFinishedVolumeSetup(self, volume):
        if self.modal_closed:
            self.modal.hide()
            self.modal_closed = False
            return

        # Set the volume
        self.volume = volume

        # Set volume flag up
        self.volume_loaded = True

        # Set up the slicer
        self.volume_slicer_props = vu.setup_slicer(self.vtk_renderer,
                                                   self.volume)
        if self.mask_loaded:
            # Remove the mask actors and add them back again, over the MRI slices
            # for a better visibility
            for slice_actor in self.mask_slicer_props:
                vu.remove_actor(self.vtk_renderer, slice_actor)
                vu.display_actor(self.vtk_renderer, slice_actor)


        # Set slicers' extent
        extent_index = 5
        for slice_widget in self.slicer_widgets:
            slice_widget.setRange(self.volume.extent[extent_index - 1], self.volume.extent[extent_index])
            slice_widget.setValue(self.volume.extent[extent_index] // 2)
            extent_index -= 2

        # Display surface/slicers depending on the current rendering mode:
        if self.rendering_menu.currentIndex() == 0:
            # Display slicers
            self.display_volume_slicers()
        else:
            # Display the surface volume
            self.display_volume_surface()

        # Setup volume settings
        self.setup_volume_settings()

        # Update settings
        self.update_volume_settings()

        # Set up the LUTS
        self.luts = vu.create_luts(self.volume)

        # Load the segmentation mask
        if self.segment:
            segmentation_image = 'output/aggregated.nii'
            if not os.path.isfile(segmentation_image):
                self.modal.hide()
                print("Could not find the images.")
                return
            self.modal.update_message(f"Loading the segmentation mask...")
            self.worker.finished.disconnect(self.onFinishedVolumeSetup)
            self.thread.started.disconnect(self.volume_lambda)
            self.thread.quit()
            self.thread.wait()

            self.worker.finished.connect(self.onFinishedMaskSetup)
            self.mask_lambda = lambda: self.worker.run_mask_setup(segmentation_image,
                                                                  self.vtk_renderer)
            self.thread.started.connect(self.mask_lambda)
            if not self.thread.isRunning():
                self.thread.start()
            else:
                print("Could not start the new thread.")
        else:
            # Set coronal view
            self.vtk_render_window.Render()
            self.set_coronal_view()

            # Hide the modal window
            self.modal.hide()

    def onFinishedMaskSetup(self, mask):
        if self.modal_closed:
            self.clear_volume()
            self.modal.hide()
            self.modal_closed = False
            return

        # Disconnect handlers
        self.worker.finished.disconnect(self.onFinishedMaskSetup)
        self.thread.started.disconnect(self.mask_lambda)
        self.thread.quit()
        self.thread.wait()

        # Clear the workspace
        if self.mask is not None:
            self.clear_mask()
            self.mask_slicer_props = None

        # Get the mask
        self.mask = mask

        # Set mask flag up
        self.mask_loaded = True

        # Set up thr slicer:
        self.mask_slicer_props = vu.setup_slicer(self.vtk_renderer,
                                                 self.mask)

        # Display surface/slicers depending on the current rendering mode:
        if self.rendering_menu.currentIndex() == 0:
            # Display slicers
            self.display_mask_slicers()
        else:
            # Display the mask surface
            self.display_mask_surface()

        # Enable volume settings
        self.setup_mask_settings()

        # Update settings
        self.update_volume_settings()

        # If volume is missing
        if not self.volume_loaded:
            # Set slicers' extent
            extent_index = 5
            for slice_widget in self.slicer_widgets:
                slice_widget.setRange(self.mask.extent[extent_index - 1], self.mask.extent[extent_index])
                slice_widget.setValue(self.mask.extent[extent_index] // 2)
                extent_index -= 2

        # Set coronal view
        self.vtk_render_window.Render()
        self.set_coronal_view()

        # Hide the modal window
        self.modal.hide()

    def onError(self, message):
        print("Error:", message)
        self.modal.update_message("Error during segmentation: " + message)

    # def add_volume(self):
    #     """
    #     Loads the MRI volume
    #     """
    #     # Check if standard image is present and load it
    #     standard_image = 'output/standard_image.nii'
    #     segmentation_image = 'output/aggregated.nii'
    #     if not os.path.isfile(standard_image) or not os.path.isfile(segmentation_image):
    #         # self.dialog.hide()
    #         print("Could not find images")
    #         return
    #     self.volume = vu.setup_volume(standard_image,
    #                                   self.vtk_renderer)
    #
    #     # Set volume flag up
    #     self.volume_loaded = True
    #
    #     # Set up the slicer
    #     self.volume_slicer_props = vu.setup_slicer(self.vtk_renderer,
    #                                                self.volume)
    #
    #     # Set slicers' extent
    #     extent_index = 5
    #     for slice_widget in self.slicer_widgets:
    #         slice_widget.setRange(self.volume.extent[extent_index - 1], self.volume.extent[extent_index])
    #         slice_widget.setValue(self.volume.extent[extent_index] // 2)
    #         extent_index -= 2
    #
    #     # Display surface/slicers depending on the current rendering mode:
    #     if self.rendering_menu.currentIndex() == 0:
    #         # Display slicers
    #         self.display_volume_slicers()
    #     else:
    #         # Display the surface volume
    #         self.display_volume_surface()
    #
    #     # Setup volume settings
    #     self.setup_volume_settings()
    #
    #     # Update settings
    #     self.update_volume_settings()
    #
    #     # Set up the LUTS
    #     self.luts = vu.create_luts(self.volume)
    #
    #     # Load the segmentation mask
    #     self.modal.update_message(f"Loading the segmentation mask...")
    #     # self.add_mask(segmentation_image)
    #
    #     # Set coronal view and render
    #     # self.set_coronal_view()
    #
    #     # self.dialog.hide()
    #     self.modal.hide()

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
                      10,
                      sum(self.volume.scalar_range) / 2,
                      # self.thrs.GetUpperThreshold(),
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
        # gu.set_slider(self.intensity_box,
        #               0,
        #               100,
        #               self.volume.image_mapper.GetLookupTable().GetValueRange()[1],
        #               None)

    def update_volume_settings(self):
        """
        Updates volume settings according to the rendering mode
        """
        # Get the current rendering mode: 0 - Slice, 1 - Surface
        rendering_mode = self.rendering_menu.currentIndex()

        # Update settings
        self.color_scheme_menu.setEnabled(not rendering_mode)
        # self.intensity_box.setEnabled(not rendering_mode)
        self.threshold_box.setEnabled(rendering_mode)
        # self.smoothness_box.setEnabled(rendering_mode)
        slider_callbacks = [self.axial_slice_changed, self.coronal_slice_changed, self.sagittal_slice_changed]
        if self.volume_loaded or self.mask_loaded:
            for i, widget in enumerate(self.slicer_widgets):
                widget.setEnabled(not rendering_mode)
                widget.valueChanged.connect(slider_callbacks[i])

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
        # opacity = 0.5 if self.mask_loaded else 1.0

        # Set opacity slicers
        vu.set_props_opacity(self.volume_slicer_props,
                             1.0)

    def hide_volume_slicers(self):
        """
        Hides the volume slicer widgets
        """
        vu.set_props_opacity(self.volume_slicer_props,
                             0.0)

    def remove_volume_slicers(self):
        """
        Removes the volume slicer widgets
        """
        for slice_prop in self.volume_slicer_props:
            self.vtk_renderer.RemoveActor(slice_prop)

    def clear_volume(self):
        """
        Removes the volume actors (surface and slice props) from the renderer
        :return:
        """
        # Check the current rendering mode
        if self.rendering_menu.currentIndex() == 0:
            # Slice mode -> remove the slicer props from the renderer
            self.remove_volume_slicers()
        else:
            # Surface mode -> remove the surface actor from the renderer
            self.hide_volume_surface()

        # Reinitialize the slicer props
        self.volume_slicer_props = []

        # Reinitialize the volume
        self.volume = None

    # endregion

    # region Mask
    def add_mask(self):
        """
        Loads the segmentation mask
        """
        # Browse for the NIfTI file
        selected_file = self.get_nifti_file()

        if not os.path.isfile(selected_file):
            print("Could not find the images.")
            return

        # Create the modal window
        if self.modal is None:
            self.modal = ModalWindow(self)
        self.modal.show()

        # Setup the worker thread
        if self.thread is None:
            self.thread = QThread()
        if self.worker is None:
            self.worker = Worker()
        self.modal.update_message(f"Loading the segmentation mask...")
        try:
            self.worker.finished.disconnect(self.onFinishedVolumeSetup)
            self.thread.started.disconnect(self.volume_lambda)
        except:
            print('No connection to disconnect')
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        # Connect the new handlers
        self.worker.finished.connect(self.onFinishedMaskSetup)
        self.mask_lambda = lambda: self.worker.run_mask_setup(selected_file,
                                                              self.vtk_renderer)
        self.thread.started.connect(self.mask_lambda)
        if not self.thread.isRunning():
            self.thread.start()
        else:
            print("Could not start the new thread.")

        # # if selected_file:
        # # Clear the workspace
        # if self.mask is not None:
        #     self.clear_mask()
        #     self.mask_slicer_props = None
        #
        # # Get the Mask
        # self.mask = vu.setup_mask(selected_file, self.vtk_renderer)
        #
        # # Set up thr slicer:
        # self.mask_slicer_props = vu.setup_slicer(self.vtk_renderer,
        #                                          self.mask)
        #
        # # Display surface/slicers depending on the current rendering mode:
        # if self.rendering_menu.currentIndex() == 0:
        #     # Display slicers
        #     self.display_mask_slicers()
        # else:
        #     # Display the mask surface
        #     self.display_mask_surface()
        #
        # # Enable volume settings
        # self.setup_mask_settings()
        #
        # # Set mask flag up
        # self.mask_loaded = True
        #
        # # If volume is missing
        # if not self.volume_loaded:
        #     # Set slicers' extent
        #     extent_index = 5
        #     for slice_widget in self.slicer_widgets:
        #         slice_widget.setRange(self.mask.extent[extent_index - 1], self.mask.extent[extent_index])
        #         slice_widget.setValue(self.mask.extent[extent_index] // 2)
        #         extent_index -= 2
        #
        #     # Set coronal view
        #     # self.set_coronal_view()

    def setup_mask_settings(self,
                            enabled: bool = True):
        """
        Sets up the segmentation mask settings
        :param enabled: Boolean parameter indicating whether the volume settings should be enabled or not.
        """
        # Set up the `All` button:
        self.mask_labels[0].setChecked(True)
        self.mask_labels[0].setEnabled(True)
        self.mask_labels[0].clicked.connect(self.mask_label_checked)
        self.display_all_masks = True

        # Enable label checkboxes
        for i, cb in enumerate(self.mask_labels[1:]):
            if i < len(self.mask.labels) and self.mask.labels[i].actor:
                cb.setEnabled(True)
                cb.setChecked(True)
                cb.clicked.connect(self.mask_label_checked)
            else:
                cb.setDisabled(True)

        # Enable slicing widgets


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
        for label in self.mask.labels:
            vu.remove_actor(self.vtk_renderer,
                            label.actor)

    def display_mask_slicers(self):
        """
        Displays the mask's slicer widgets
        """
        # Determine the opacity
        opacity = 0.5 if self.volume_loaded else 1.0

        # Display slicers
        vu.set_props_opacity(self.mask_slicer_props,
                             1.0)

    def hide_mask_slicers(self):
        """
        Hides the mask's slicer widgets
        """
        vu.set_props_opacity(self.mask_slicer_props,
                             0.0)

    def remove_mask_slicers(self):
        """
        Removes the mask slicer widgets
        """
        for slice_prop in self.mask_slicer_props:
            self.vtk_renderer.RemoveActor(slice_prop)

    def clear_mask(self):
        """
        Removes the mask actors from the renderer
        """
        # Check the current rendering mode
        if self.rendering_menu.currentIndex() == 0:
            # Slice mode -> remove the slicer props from the renderer
            self.hide_mask_slicers()
        else:
            # Surface mode -> remove the surface actor from the renderer
            self.hide_mask_surface()

        # Reinitialize the slicer props
        self.mask_slicer_props = []

        # Reinitialize the volume
        self.mask = None

    # endregion

    # region Callbacks
    def render_mode_changed(self, index):
        """
        Rendering mode changed callback
        :param index: the selected rendering mode
                      (0 - Slice,
                      1 - Surface)
        """
        # Update settings
        self.set_widget_visibility(index)

        # Enable/Disable the slicing widgets
        if self.volume_loaded or self.mask_loaded:
            for widget in self.slicer_widgets:
                widget.setEnabled(not index)

        # # Get the opacity
        # if index == 1:
        #     opacity = 0
        # elif self.volume_loaded and self.mask_loaded:
        #     opacity = 0.5
        # elif self.volume_loaded or self.mask_loaded:
        #     opacity = 1
        # else:
        #     opacity = 0

        volume_opacity = round(self.volume_opacity_box.value() / 100, 2)
        mask_opacity = round(self.mask_opacity_box.value() / 100, 2)

        if self.volume_loaded:
            # Hide/show the surface volume
            if index == 0:
                self.vtk_renderer.RemoveActor(self.volume.labels[0].actor)
                # Set volume slicer opacity
                for prop in self.volume_slicer_props:
                    prop.GetProperty().SetOpacity(volume_opacity)
            else:
                self.vtk_renderer.AddActor(self.volume.labels[0].actor)
                self.volume.labels[0].property.SetOpacity(volume_opacity)
                # Set volume slicer opacity
                for prop in self.volume_slicer_props:
                    prop.GetProperty().SetOpacity(0)

        if self.mask_loaded:
            # Hide/show the surface volume
            for label_idx in range(len(MASK_COLORS)):
                if index == 0:
                    self.vtk_renderer.RemoveActor(self.mask.labels[label_idx].actor)
                    for prop in self.mask_slicer_props:
                        prop.GetProperty().SetOpacity(mask_opacity)
                else:
                    self.vtk_renderer.AddActor(self.mask.labels[label_idx].actor)
                    # self.mask.labels[label_idx].property.SetOpacity(mask_opacity)
                    # Set volume slicer opacity
                    for i, cb in enumerate(self.mask_labels[1:]):
                        if cb.isChecked():
                            # Set the surface opacity
                            self.mask.labels[i].property.SetOpacity(self.mask_opacity_box.value())
                    # Set volume slicer opacity
                    for prop in self.mask_slicer_props:
                        prop.GetProperty().SetOpacity(0)

        # Update settings
        if self.volume_loaded:
            self.update_volume_settings()

        # Render the window
        # self.worker = Worker(self.vtk_render_window)
        # self.worker.finished.connect(self.on_render_ready)
        # self.worker.start()
        self.vtk_render_window.Render()
        # self.render()

    def render(self):
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker(self.vtk_render_window)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        # self.worker.progress.connect(self.reportProgress)
        # Step 6: Start the thread
        self.thread.start()

    def color_scheme_changed(self, index):
        """
        Color scheme changed callback
        :param index: the selected color scheme
        """
        if self.rendering_menu.currentIndex() == 0:
            # If slice mode => change lut
            self.volume.image_mapper.SetLookupTable(self.luts[index])
            self.volume.image_mapper.Update()
            self.vtk_render_window.Render()

    def change_volume_threshold(self):
        # Pich current value
        value = self.threshold_box.value()

        # Disable the button until rendering is done
        self.threshold_box.setEnabled(False)

        # Apply the new threshold value and render
        if self.volume is not None:
            self.volume.labels[0].extractor.SetValue(0, value)
        self.vtk_renderer.Render()

        # Enable the threshold button back
        self.threshold_box.setEnabled(True)

    def threshold_changed(self):
        """
        Threshold changed callback
        :param value: threshold level
        """
        if not self.processing:
            self.processing = True
            self.process_changes()
            value = self.threshold_box.value()
            if self.volume is not None:
                self.volume.labels[0].extractor.SetValue(0, value)
            # self.thrs.ThresholdByUpper(value)
            self.vtk_renderer.Render()
            self.processing = False

    def volume_opacity_changed(self, value):
        """
        Volume changed callback
        :param value: opacity level in percentages (0 - 100)
        """
        # Bring the value to [0, 1]
        opacity = round(value / 100, 2)

        if self.rendering_menu.currentIndex() == 1:
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
        if not self.processing:
            self.processing = True
            self.process_changes()
            self.volume.labels[0].smoother.SetNumberOfIterations(value)
            self.vtk_render_window.Render()
            self.processing = False

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

        if self.rendering_menu.currentIndex() == 1:
            # Surface rendering mode
            for i, label in enumerate(self.mask.labels):
                if label.property and self.mask_labels[i+1].isChecked():
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

        if self.mask_labels[0].isChecked() != self.display_all_masks:
            self.display_all_masks = self.mask_labels[0].isChecked()
            for j in range(1, 96):
                if self.mask_labels[j].isEnabled():
                    # Set the surface opacity
                    self.mask_labels[j].setChecked(self.display_all_masks)

        for i, cb in enumerate(self.mask_labels[1:]):
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

    def axial_slice_changed(self, value):
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

    def coronal_slice_changed(self, value):
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

    def sagittal_slice_changed(self, value):
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
        for _ in range(10):
            self.app.processEvents()
            time.sleep(0.2)

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
        # self.vtk_renderer.GetActiveCamera().SetPosition(fp[0], fp[2] + dist, fp[1])
        self.vtk_renderer.GetActiveCamera().SetPosition(fp[0], fp[1], fp[2] + dist)
        self.vtk_renderer.GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)
        self.vtk_renderer.GetActiveCamera().Zoom(2)
        self.vtk_render_window.Render()

    def set_coronal_view(self):
        """
        Sets the coronal view
        """
        self.vtk_renderer.ResetCamera()
        fp = self.vtk_renderer.GetActiveCamera().GetFocalPoint()
        p = self.vtk_renderer.GetActiveCamera().GetPosition()
        dist = math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.vtk_renderer.GetActiveCamera().SetPosition(fp[0], fp[2] + dist, fp[1])
        # self.vtk_renderer.GetActiveCamera().SetPosition(fp[2] + dist, fp[0], fp[1])
        # self.vtk_renderer.GetActiveCamera().SetPosition(fp[0], fp[1] + dist, fp[2])
        self.vtk_renderer.GetActiveCamera().SetViewUp(0.0, 0.0, 1.0)
        self.vtk_renderer.GetActiveCamera().Zoom(2)
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
        # self.vtk_renderer.GetActiveCamera().SetPosition(fp[0], fp[1], fp[2] + dist)
        self.vtk_renderer.GetActiveCamera().SetPosition(fp[0] - dist, fp[1], fp[2])
        self.vtk_renderer.GetActiveCamera().SetViewUp(0.0, 0.0, 1.0)
        self.vtk_renderer.GetActiveCamera().Zoom(2)
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
