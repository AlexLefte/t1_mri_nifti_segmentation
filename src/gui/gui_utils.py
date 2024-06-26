import PyQt5.QtCore as Qt
import PyQt5.QtWidgets as QtWidgets


def create_button(title):
    """
    Creates a button
    :param title:
    :param handler:
    :return:
    """
    button = QtWidgets.QPushButton(title)
    button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    return button


def create_separator():
    """
    Creates a separator line
    :return: A QWidget representing a line separator
    """
    separator = QtWidgets.QWidget()
    separator.setFixedHeight(1)
    separator.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    separator.setStyleSheet("background-color: #c8c8c8;")
    return separator


def create_combo_box(values: list,
                     enabled: bool = True):
    """
    Creates a combobox widget, given a list of values
    :param values: A list of strings representing the options to be displayed in the dropdown menu.
    :param enabled: Enable flag
    :return: A QComboBox widget representing the created dropdown menu.
    """
    # Initialize the combo box
    dropdown_menu = QtWidgets.QComboBox()

    # Set enabled
    dropdown_menu.setEnabled(enabled)

    # Add new items
    for value in values:
        dropdown_menu.addItem(value)

    # Return
    return dropdown_menu


def create_picker(min_value=0,
                  max_value=100,
                  step: int = 1,
                  value=50,
                  changed_callback=None,
                  is_enabled: bool = True):
    """
    Returns a QSpinBox, configured with the desired parameters
    :param min_value: the minimum value
    :param max_value: the maximum value
    :param step: the advancing/retracting step
    :param value: initial value
    :param changed_callback: callback
    :param is_enabled: enable flag
    :return: picker
    """
    if is_enabled is False:
        picker = QtWidgets.QDoubleSpinBox()
        picker.setEnabled(False)
        return picker
    # elif min_value is int:
    #     picker = QtWidgets.QSpinBox()
    # else:
    #     picker = QtWidgets.QDoubleSpinBox()
    picker = QtWidgets.QDoubleSpinBox()
    picker.setMinimum(min_value)
    picker.setMaximum(max_value)
    picker.setSingleStep(step)
    picker.setValue(value)

    if changed_callback is not None:
        picker.valueChanged.connect(changed_callback)
    return picker


def set_picker(picker,
               min_value=0,
               max_value=100,
               step: int = 1,
               value=50,
               changed_callback=None,
               is_enabled: bool = True):
    """
    Returns a QSpinBox, configured with the desired parameters
    :param picker: picker widget
    :param min_value: the minimum value
    :param max_value: the maximum value
    :param step: the advancing/retracting step
    :param value: initial value
    :param changed_callback: callback
    :param is_enabled: enable flag
    :return: picker
    """
    picker.setEnabled(is_enabled)
    picker.setMinimum(min_value)
    picker.setMaximum(max_value)
    picker.setSingleStep(step)
    picker.setValue(value)

    if changed_callback is not None:
        picker.valueChanged.connect(changed_callback)
    return picker


def create_slider(min_value: int = 0,
                  max_value: int = 100,
                  value: int = 50,
                  callback=None,
                  is_enabled: bool = False):
    """
    Returns a QSlider, configured with the desired parameters
    :param min_value: the minimum value
    :param max_value: the maximum value
    :param value: the current value
    :param callback: the changed callback
    :param is_enabled: enable flag
    :return: slider_widget
    """
    slider_widget = QtWidgets.QSlider(Qt.Qt.Horizontal)
    slider_widget.setEnabled(is_enabled)
    if callback is not None:
        slider_widget.valueChanged.connect(callback)
    slider_widget.setRange(min_value, max_value)
    slider_widget.setValue(value)
    return slider_widget


def set_slider(slice_widget,
               min_value: int = 0,
               max_value: int = 100,
               value: int = 50,
               callback=None,
               is_enabled: bool = False):
    """
    Sets the slider widget's properties
    :param slice_widget: QSlider-type object
    :param min_value: the minimum value
    :param max_value: the maximum value
    :param value: the current value
    :param callback: the changed callback
    :param is_enabled: enable flag
    """
    min_value = int(min_value)
    max_value = int(max_value)
    value = int(value)
    slice_widget.setRange(min_value, max_value)
    slice_widget.setEnabled(is_enabled)
    slice_widget.setValue(value)
    if callback is not None:
        slice_widget.valueChanged.connect(callback)
