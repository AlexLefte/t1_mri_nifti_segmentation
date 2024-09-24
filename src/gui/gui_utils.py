import PyQt5.QtCore as Qt
import PyQt5.QtWidgets as QtWidgets

def create_button(title):
    """
    Creates a button with the specified title.

    Parameters
    ----------
    title : str
        The title of the button.

    Returns
    -------
    button : QPushButton
        The created button.
    """
    button = QtWidgets.QPushButton(title)
    button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    return button

def create_separator():
    """
    Creates a line separator.

    Returns
    -------
    separator : QWidget
        A QWidget representing the line separator.
    """
    separator = QtWidgets.QWidget()
    separator.setFixedHeight(1)
    separator.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    separator.setStyleSheet("background-color: #c8c8c8;")
    return separator

def create_combo_box(values: list, enabled: bool = True):
    """
    Creates a combobox widget with given values.

    Parameters
    ----------
    values : list of str
        The options to be displayed in the dropdown menu.
    enabled : bool, optional
        Flag to enable or disable the combo box (default is True).

    Returns
    -------
    dropdown_menu : QComboBox
        The created combobox widget.
    """
    # Initialize the combo box
    dropdown_menu = QtWidgets.QComboBox()
    dropdown_menu.setEnabled(enabled)

    # Add new items
    for value in values:
        dropdown_menu.addItem(value)

    return dropdown_menu

def create_picker(min_value=0, max_value=100, step: int = 1, value=50, changed_callback=None, is_enabled: bool = True):
    """
    Returns a QSpinBox configured with the desired parameters.

    Parameters
    ----------
    min_value : int, optional
        The minimum value (default is 0).
    max_value : int, optional
        The maximum value (default is 100).
    step : int, optional
        The step size (default is 1).
    value : int, optional
        The initial value (default is 50).
    changed_callback : callable, optional
        The callback function for value change events.
    is_enabled : bool, optional
        Flag to enable or disable the picker (default is True).

    Returns
    -------
    picker : QSpinBox
        The created picker widget.
    """
    # Initialize the picker
    picker = QtWidgets.QDoubleSpinBox()
    picker.setMinimum(min_value)
    picker.setMaximum(max_value)
    picker.setSingleStep(step)
    picker.setValue(value)
    picker.setEnabled(is_enabled)

    if changed_callback is not None:
        picker.valueChanged.connect(changed_callback)
    return picker

def set_picker(picker, min_value=0, max_value=100, step: int = 1, value=50, changed_callback=None, is_enabled: bool = True):
    """
    Configures an existing picker widget with the desired parameters.

    Parameters
    ----------
    picker : QSpinBox
        The picker widget to configure.
    min_value : int, optional
        The minimum value (default is 0).
    max_value : int, optional
        The maximum value (default is 100).
    step : int, optional
        The step size (default is 1).
    value : int, optional
        The initial value (default is 50).
    changed_callback : callable, optional
        The callback function for value change events.
    is_enabled : bool, optional
        Flag to enable or disable the picker (default is True).

    Returns
    -------
    picker : QSpinBox
        The configured picker widget.
    """
    picker.setEnabled(is_enabled)
    picker.setMinimum(min_value)
    picker.setMaximum(max_value)
    picker.setSingleStep(step)
    picker.setValue(value)

    if changed_callback is not None:
        picker.valueChanged.connect(changed_callback)
    return picker

def create_slider(min_value: int = 0, max_value: int = 100, value: int = 50, callback=None, is_enabled: bool = False):
    """
    Returns a QSlider configured with the desired parameters.

    Parameters
    ----------
    min_value : int, optional
        The minimum value (default is 0).
    max_value : int, optional
        The maximum value (default is 100).
    value : int, optional
        The initial value (default is 50).
    callback : callable, optional
        The callback function for value change events.
    is_enabled : bool, optional
        Flag to enable or disable the slider (default is False).

    Returns
    -------
    slider_widget : QSlider
        The created slider widget.
    """
    # Initialize the slider
    slider_widget = QtWidgets.QSlider(Qt.Qt.Horizontal)
    slider_widget.setEnabled(is_enabled)
    slider_widget.setRange(min_value, max_value)
    slider_widget.setValue(value)

    if callback is not None:
        slider_widget.valueChanged.connect(callback)
    return slider_widget

def set_slider(slider_widget, min_value: int = 0, max_value: int = 100, value: int = 50, callback=None, is_enabled: bool = False):
    """
    Configures an existing slider widget with the desired parameters.

    Parameters
    ----------
    slider_widget : QSlider
        The slider widget to configure.
    min_value : int, optional
        The minimum value (default is 0).
    max_value : int, optional
        The maximum value (default is 100).
    value : int, optional
        The initial value (default is 50).
    callback : callable, optional
        The callback function for value change events.
    is_enabled : bool, optional
        Flag to enable or disable the slider (default is False).
    """
    min_value = int(min_value)
    max_value = int(max_value)
    value = int(value)

    slider_widget.setRange(min_value, max_value)
    slider_widget.setValue(value)
    slider_widget.setEnabled(is_enabled)

    if callback is not None:
        slider_widget.valueChanged.connect(callback)
