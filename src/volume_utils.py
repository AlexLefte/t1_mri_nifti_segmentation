class Volume:
    """
    Represents the volume class.
    This class encapsulates all the necessary components and properties
    to represent a 3D volume in VTK (both the MRI and the segmentation mask).
    """
    def __init__(self):
        self.reader = None  # The NIfTI reader to load the volume data
        self.extent = []  # The extent of the volume data
        self.labels = []  # A list to hold different labels associated with the volume
        self.image_mapper = None  # The image mapper to map volume data to colors
        self.scalar_range = None  # The range of scalar values in the volume data


class Label:
    """
    Represents the label class.
    This class holds properties and components necessary to render a specific
    structure within the volume.
    """
    def __init__(self, color, opacity, smoothness):
        self.extractor = None  # The extractor used to generate the surface of the label
        self.actor = None  # The actor that represents the label in the scene
        self.property = None  # The visual properties of the actor (e.g., color, opacity)
        self.smoothness = smoothness  # The level of smoothness / number of iterations for the vtk data filter
        self.color = color  # The color of the label
        self.opacity = opacity  # The opacity of the label
