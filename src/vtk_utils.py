import vtk
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

from src.volume_utils import *
from src.config import *
import numpy


def add_surface_rendering(volume, label_idx, label_value):
    """
    Renders the surface for a given label in the volume.

    Parameters
    ----------
    volume : Volume
        The volume object containing the data to be rendered.
    label_idx : int
        The index of the label to be rendered.
    label_value : float
        The value of the label to be rendered.

    Returns
    -------
    None
    """
    # Set the label value for the extractor and update it
    volume.labels[label_idx].extractor.SetValue(0, label_value)
    volume.labels[label_idx].extractor.Update()

    # Check if there is any data in the extractor output
    if volume.labels[label_idx].extractor.GetOutput().GetMaxCellSize():
        # Reduce the number of polygons (triangles) in the volume => Speeds up rendering
        reducer = vtk.vtkDecimatePro()
        reducer.SetInputConnection(volume.labels[label_idx].extractor.GetOutputPort())
        reducer.SetTargetReduction(0.35)
        reducer.PreserveTopologyOn()

        # Smooth the render edges
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(reducer.GetOutputPort())
        smoother.SetNumberOfIterations(volume.labels[label_idx].smoothness)

        # Compute polygon normals for better rendering quality
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.SetFeatureAngle(60.0)

        # Create the mapper => Converts the data into a format that can be displayed
        actor_mapper = vtk.vtkPolyDataMapper()
        actor_mapper.SetInputConnection(normals.GetOutputPort())
        actor_mapper.ScalarVisibilityOff()

        # Create the property => Sets visual properties (opacity, color)
        actor_property = vtk.vtkProperty()
        actor_property.SetOpacity(volume.labels[label_idx].opacity)
        actor_property.SetColor(volume.labels[label_idx].color)

        # Create the actor => Represents the data in the scene
        actor = vtk.vtkActor()
        actor.SetMapper(actor_mapper)
        actor.SetProperty(actor_property)

        # Store the actor, smoother, and property in the volume label
        volume.labels[label_idx].actor = actor
        volume.labels[label_idx].smoother = smoother
        volume.labels[label_idx].property = actor_property


def display_actor(renderer: vtk.vtkRenderer,
                  actor: vtk.vtkActor):
    """
    Adds actor to renderer
    :param renderer: R
    enderer
    :param actor: Actor to be displayed
    """
    renderer.AddActor(actor)


def remove_actor(renderer: vtk.vtkRenderer,
                 actor: vtk.vtkActor):
    """
    Adds actor to renderer
    :param renderer: Renderer
    :param actor: Actor to be removed
    """
    renderer.RemoveActor(actor)


def get_mask_extractor(mask: Volume):
    """
    Given the output from mask (vtkNIFTIImageReader) extract it into 3D using
    vtkDiscreteMarchingCubes algorithm (https://www.vtk.org/doc/release/5.0/html/a01331.html).
    This algorithm is specialized for reading segmented volume labels.
    :param mask: a vtkNIFTIImageReader volume containing the mask
    :return: the extracted volume from vtkDiscreteMarchingCubes
    """
    # shrink = vtk.vtkImageShrink3D()
    # shrink.SetInputConnection(mask.reader.GetOutputPort())
    # shrink.SetShrinkFactors(2, 2, 2)
    # shrink.Update()

    mask_extractor = vtk.vtkDiscreteMarchingCubes()
    mask_extractor.SetInputConnection(mask.reader.GetOutputPort())
    # mask_extractor.SetInputConnection(shrink.GetOutputPort())
    mask_extractor.ComputeNormalsOff()
    # mask_extractor.SetNumberOfThreads(vtk.vtkMultiThreader.GlobalDefaultNumberOfThreads())
    return mask_extractor


def create_file_reader(file: str):
    """
    Returns the vtkNIFTIIImageReader instance
    :param file: file path
    :return: reader
    """
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileNameSliceOffset(1)
    reader.SetDataByteOrderToBigEndian()
    reader.SetFileName(file)
    reader.Update()
    return reader


def setup_volume(file: str,
                 renderer: vtk.vtkRenderer):
    """
    Sets up the volume

    Parameters
    ----------
    file: str
        The file path to the NIfTI file.
    renderer: vtk.vtkRenderer
        The VTK renderer to use for rendering the volume.

    Returns
    -------
    Volume
        The configured volume object.
    """
    # Initialize the Volume type object
    volume = Volume()

    # Create the VTK NIfTI image reader
    volume.reader = create_file_reader(file)

    # Append the label corresponding to the MRI image
    volume.labels.append(Label(BRAIN_COLOR, BRAIN_OPACITY, BRAIN_SMOOTHNESS))

    # Create and configure the VTK image shrink filter => Subsamples the volume
    shrink = vtk.vtkImageShrink3D()
    shrink.SetInputConnection(volume.reader.GetOutputPort())
    shrink.SetShrinkFactors(2, 2, 2)
    shrink.Update()

    # Create and configure the VTK volume extractor => Generates the surface
    # from the volumetric data
    volume_extractor = vtk.vtkFlyingEdges3D()
    volume_extractor.SetInputConnection(shrink.GetOutputPort())
    volume_extractor.ComputeNormalsOff()
    volume_extractor.Update()
    volume.labels[0].extractor = volume_extractor

    # Store the data extent for future use
    volume.extent = volume.reader.GetDataExtent()

    # Set up the lookup table (LUT) for color mapping
    scalar_range = volume.reader.GetOutput().GetScalarRange()
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(scalar_range)
    lut.SetSaturationRange(0, 0)
    lut.SetHueRange(0, 0)
    lut.SetValueRange(0, 1)
    lut.Build()

    # Map the image data to colors using the lookup table
    view_colors = vtk.vtkImageMapToColors()
    view_colors.SetInputConnection(volume.reader.GetOutputPort())
    view_colors.SetLookupTable(lut)
    view_colors.Update()
    volume.image_mapper = view_colors
    volume.scalar_range = scalar_range

    # Add surface rendering
    add_surface_rendering(volume, 0, sum(scalar_range) / 2)
    return volume


def setup_mask(file: str,
               renderer: vtk.vtkRenderer):
    """
    Sets up the segmentation mask

    Parameters
    ----------
    file: str
        The file path to the NIfTI file.
    renderer: vtk.vtkRenderer
        The VTK renderer to use for rendering the mask.

    Returns
    -------
    Mask
        The configured mask object.
    """
    # Initialize the mask Volume type object
    mask = Volume()

    # Create the VTK NIfTI image reader => Reads the NIfTI file
    mask.reader = create_file_reader(file)

    # Save the data extent for future use
    mask.extent = mask.reader.GetDataExtent()

    # Set up the lookup table (LUT) for color mapping
    # Mask colors => dictionary containing the RGB color code for each class
    scalar_range = mask.reader.GetOutput().GetScalarRange()
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(int(scalar_range[1])+1)
    lut.SetTableRange(scalar_range)
    lut.SetTableValue(0, 0, 0, 0, 0.0)
    for key, value in MASK_COLORS.items():
        r, g, b = value
        lut.SetTableValue(key, r, g, b, 1.0)
    lut.Build()

    # Map the image data to colors using the lookup table => Applies the LUT to the mask
    image_mapper = vtk.vtkImageMapToColors()
    image_mapper.SetInputConnection(mask.reader.GetOutputPort())
    image_mapper.SetLookupTable(lut)
    image_mapper.Update()
    mask.image_mapper = image_mapper
    mask.scalar_range = scalar_range

    # Add surface rendering for each anatomical structure
    for i, label_idx in enumerate(MASK_COLORS.keys()):
        mask.labels.append(Label(MASK_COLORS[label_idx], MASK_OPACITY, MASK_SMOOTHNESS))
        mask.labels[i].extractor = get_mask_extractor(mask)
        add_surface_rendering(mask, i, label_idx)
    return mask


def setup_slicer(renderer: vtk.vtkRenderer,
                 obj: Volume):
    """
    Sets up the slicing widgets for visualizing different planes of a 3D volume.

    Parameters
    ----------
    renderer : vtk.vtkRenderer
        The VTK renderer responsible for rendering the slicing actors.
    obj : Volume
        The volume to be sliced, containing the image data and its properties.

    Returns
    -------
    list
        A list of slicing actors (axial, coronal, sagittal).
    """
    # Get the extent of the volume (dimensions in each direction)
    _, x, _, y, _, z = obj.extent

    # Create the axial slicing actor
    axial = vtk.vtkImageActor()
    axial_prop = vtk.vtkImageProperty()
    axial_prop.SetOpacity(0)
    axial.SetProperty(axial_prop)
    axial.GetMapper().SetInputConnection(obj.image_mapper.GetOutputPort())
    axial.SetDisplayExtent(0, x, 0, y, int(z/2), int(z/2))
    axial.InterpolateOn()
    axial.ForceOpaqueOn()

    # Create the coronal slicing actor
    coronal = vtk.vtkImageActor()
    cor_prop = vtk.vtkImageProperty()
    cor_prop.SetOpacity(0)
    coronal.SetProperty(cor_prop)
    coronal.GetMapper().SetInputConnection(obj.image_mapper.GetOutputPort())
    coronal.SetDisplayExtent(0, x, int(y/2), int(y/2), 0, z)
    coronal.InterpolateOn()
    coronal.ForceOpaqueOn()

    # Create the sagittal slicing actor
    sagittal = vtk.vtkImageActor()
    sag_prop = vtk.vtkImageProperty()
    sag_prop.SetOpacity(0)
    sagittal.SetProperty(sag_prop)
    sagittal.GetMapper().SetInputConnection(obj.image_mapper.GetOutputPort())
    sagittal.SetDisplayExtent(int(x/2), int(x/2), 0, y, 0, z)
    sagittal.InterpolateOn()
    sagittal.ForceOpaqueOn()

    # Add the slicing actors to the renderer
    renderer.AddActor(axial)
    renderer.AddActor(coronal)
    renderer.AddActor(sagittal)

    # Return the list of slicing actors
    return [axial, coronal, sagittal]


def set_props_opacity(props: list,
                      opacity: float = 1.0):
    """
    Sets the desired opacity of the properties inside the provided list
    :param props: properties list
    :param opacity: opacity value
    """
    for prop in props:
        prop.GetProperty().SetOpacity(opacity)


# region LookUpTables
def create_luts(volume):
    """
    Initializes the default lookup tables
    :param volume: Volume
    :return: A list of default lookup tables
    """
    # Initialize the list
    luts = []

    # Gray scale lookup table
    gray_scale_lut = vtk.vtkLookupTable()
    scalar_range = volume.reader.GetOutput().GetScalarRange()
    gray_scale_lut.SetTableRange(scalar_range)
    gray_scale_lut.SetSaturationRange(0, 0)
    gray_scale_lut.SetHueRange(0, 0)
    gray_scale_lut.SetValueRange(0, 1)
    gray_scale_lut.Build()
    luts.append(gray_scale_lut)

    # Rainbow blue-red lookup table
    rainbow_blue_red_lut = vtk.vtkLookupTable()
    rainbow_blue_red_lut.SetTableRange(scalar_range)
    rainbow_blue_red_lut.SetNumberOfColors(256)
    rainbow_blue_red_lut.SetHueRange(0.667, 0.0)
    rainbow_blue_red_lut.Build()
    luts.append(rainbow_blue_red_lut)

    # Rainbow red-blue lookup table
    rainbow_red_blue_lut = vtk.vtkLookupTable()
    rainbow_red_blue_lut.SetTableRange(scalar_range)
    rainbow_red_blue_lut.SetNumberOfColors(256)
    rainbow_red_blue_lut.SetHueRange(0.0, 0.667)
    rainbow_red_blue_lut.Build()
    luts.append(rainbow_red_blue_lut)

    # High contrast lookup table
    high_contrast_lut = vtk.vtkLookupTable()
    high_contrast_lut.SetTableRange(scalar_range)
    high_contrast_lut.SetNumberOfColors(256)
    # Build the lookup table
    high_contrast_lut.Build()
    # Set color values for high contrast lookup table
    for i in range(16):
        high_contrast_lut.SetTableValue(i * 16, 1, 0, 0, 1)
        high_contrast_lut.SetTableValue(i * 16 + 1, 0, 1, 0, 1)
        high_contrast_lut.SetTableValue(i * 16 + 2, 0, 0, 1, 1)
        high_contrast_lut.SetTableValue(i * 16 + 3, 0, 0, 0, 1)
    # luts.append(high_contrast_lut)

    return luts
# endregion
