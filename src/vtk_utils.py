import vtk
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

from src.nii_utils import *
from src.config import *
import numpy


def add_surface_rendering(nii_object, label_idx, label_value):
    """
    Renders the surface
    :param nii_object: the volume to be rendered
    :param label_idx: the label index
    :param label_value: the label value
    """
    nii_object.labels[label_idx].extractor.SetValue(0, label_value)
    nii_object.labels[label_idx].extractor.Update()

    # Check if there is any label_idx data
    if nii_object.labels[label_idx].extractor.GetOutput().GetMaxCellSize():
        reducer = create_polygon_reducer(nii_object.labels[label_idx].extractor)
        smoother = create_smoother(reducer, nii_object.labels[label_idx].smoothness)
        normals = create_normals(smoother)
        actor_mapper = create_mapper(normals)
        actor_property = create_property(nii_object.labels[label_idx].opacity, nii_object.labels[label_idx].color)
        actor = create_actor(actor_mapper, actor_property)
        nii_object.labels[label_idx].actor = actor
        nii_object.labels[label_idx].smoother = smoother
        nii_object.labels[label_idx].property = actor_property


def create_polygon_reducer(extractor):
    """
    Reduces the number of polygons (triangles) in the volume. This is used to speed up rendering.
    (https://www.vtk.org/doc/nightly/html/classvtkDecimatePro.html)
    :param extractor: an extractor (vtkPolyDataAlgorithm), will be either vtkFlyingEdges3D or vtkDiscreteMarchingCubes
    :return: the decimated volume
    """
    reducer = vtk.vtkDecimatePro()
    reducer.SetInputConnection(extractor.GetOutputPort())
    reducer.SetTargetReduction(0.35)
    reducer.PreserveTopologyOn()
    return reducer


def create_smoother(reducer, smoothness):
    """
    Reorients some points in the volume to smooth the render edges.
    (https://www.vtk.org/doc/nightly/html/classvtkSmoothPolyDataFilter.html)
    :param reducer:
    :param smoothness:
    :return:
    """
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(reducer.GetOutputPort())
    smoother.SetNumberOfIterations(smoothness)
    return smoother


def create_normals(smoother):
    """
    The filter can reorder polygons to insure consistent orientation across polygon neighbors. Sharp edges can be split
    and points duplicated with separate normals to give crisp (rendered) surface definition.
    (https://www.vtk.org/doc/nightly/html/classvtkPolyDataNormals.html)
    :param smoother:
    :return:
    """
    brain_normals = vtk.vtkPolyDataNormals()
    brain_normals.SetInputConnection(smoother.GetOutputPort())
    brain_normals.SetFeatureAngle(60.0)  #
    return brain_normals


def create_mapper(stripper):
    """
    Creates a mapper given the stripper
    :param stripper: vtkPolyDataNormals-type object
    :return vtkPolyDataMapper-type object
    """
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())
    mapper.ScalarVisibilityOff()
    mapper.Update()
    return mapper


def create_property(opacity, color):
    """
    Creates a vtk property
    :param opacity: opacity level
    :param color: color
    :return: vtkProperty-type object
    """
    prop = vtk.vtkProperty()
    prop.SetColor(color[0], color[1], color[2])
    prop.SetOpacity(opacity)
    return prop


def create_actor(mapper, prop):
    """
    Creates a vtk actor
    :param mapper: vtk mapper
    :param prop: vtk property
    :return: vtk actor
    """
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)
    return actor


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


def get_volume_extractor(volume: Volume):
    """
    Given the output from volume (vtkNIFTIImageReader) extract it into 3D using
    vtkFlyingEdges3D algorithm (https://www.vtk.org/doc/nightly/html/classvtkFlyingEdges3D.html)
    :param volume: a vtkNIFTIImageReader volume containing the brain
    :return: the extracted volume from vtkFlyingEdges3D
    """
    shrink = vtk.vtkImageShrink3D()
    shrink.SetInputConnection(volume.reader.GetOutputPort())
    shrink.SetShrinkFactors(2, 2, 2)
    shrink.Update()

    volume_extractor = vtk.vtkFlyingEdges3D()
    # volume_extractor.SetInputConnection(volume.reader.GetOutputPort())
    volume_extractor.SetInputConnection(shrink.GetOutputPort())
    volume_extractor.ComputeNormalsOff()
    volume_extractor.Update()
    return volume_extractor


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


def read_volume(file: str):
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
    :param file: file path
    :param renderer: vtk renderer
    :return: volume
    """
    volume = Volume()
    volume.file = file
    volume.reader = read_volume(file)

    # Check and fix orientation
    # current_orientation_matrix = get_orientation(file)
    # print(f"Current orientation: {current_orientation_matrix}")
    # # Set the desired orientation
    # desired_orientation = numpy.eye(4)  # Replace with the desired orientation
    # volume.reorient = set_orientation(volume.reader, file, desired_orientation)

    volume.labels.append(NiiLabel(BRAIN_COLORS[0], BRAIN_OPACITY, BRAIN_SMOOTHNESS))
    volume.labels[0].extractor = get_volume_extractor(volume)
    volume.extent = volume.reader.GetDataExtent()

    scalar_range = volume.reader.GetOutput().GetScalarRange()
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(scalar_range)
    lut.SetSaturationRange(0, 0)
    lut.SetHueRange(0, 0)
    lut.SetValueRange(0, 1)
    lut.Build()

    view_colors = vtk.vtkImageMapToColors()
    view_colors.SetInputConnection(volume.reader.GetOutputPort())
    view_colors.SetLookupTable(lut)
    view_colors.Update()
    volume.image_mapper = view_colors
    volume.scalar_range = scalar_range

    add_surface_rendering(volume, 0, sum(scalar_range) / 2)
    return volume


def get_orientation(nifti_file):
    """
    Get the orientation of the NIfTI volume, based on the QFormMatrix
    :param nifti_file: NIfTI file's path
    :return: the orientation matrix
    """
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_file)
    reader.Update()

    orientation_matrix = reader.GetQFormMatrix()

    return orientation_matrix


def set_orientation(reader, new_orientation):
    """
    Attempts to set the desired orientation
    :param reader: vtk NIfTI reader
    :param new_orientation: desired orientation matrix
    :return: reorient vtk block
    """
    current_orientation_matrix = reader.GetQFormMatrix()

    # Check if the current and new orientations are different
    if not numpy.array_equal(current_orientation_matrix, new_orientation):
        # Create a filter to reorient the image
        reorient = vtk.vtkImageReslice()
        reorient.SetInputConnection(reader.GetOutputPort())

        # Set the desired orientation
        reorient.SetResliceAxesDirectionCosines(new_orientation[:3, :3].flatten())
        reorient.Update()
        return reorient
    else:
        print("Image is already in the desired orientation.")


def setup_mask(file: str,
               renderer: vtk.vtkRenderer):
    """
    Sets up the mask
    :param file: file path
    :param renderer: vtk renderer
    :return: mask
    """
    mask = Volume()
    mask.file = file
    mask.reader = read_volume(mask.file)
    mask.extent = mask.reader.GetDataExtent()
    scalar_range = mask.reader.GetOutput().GetScalarRange()
    n_labels = int(mask.reader.GetOutput().GetScalarRange()[1])
    # n_labels = n_labels if n_labels <= 10 else 10

    lut = vtk.vtkLookupTable()
    table_size = len(MASK_COLORS) + 1
    # lut.SetNumberOfColors(table_size)
    lut.SetNumberOfTableValues(int(scalar_range[1])+1)
    lut.SetTableRange(scalar_range)

    lut.SetTableValue(0, 0, 0, 0, 0.0)
    for key, value in MASK_COLORS.items():
        r, g, b = value
        lut.SetTableValue(key, r, g, b, 1.0)

    # lut.IndexedLookupOn()
    lut.Build()

    view_colors = vtk.vtkImageMapToColors()
    view_colors.SetInputConnection(mask.reader.GetOutputPort())
    view_colors.SetLookupTable(lut)
    view_colors.Update()
    mask.image_mapper = view_colors
    mask.scalar_range = scalar_range

    for i, label_idx in enumerate(MASK_COLORS.keys()):
        mask.labels.append(NiiLabel(MASK_COLORS[label_idx], MASK_OPACITY, MASK_SMOOTHNESS))
        mask.labels[i].extractor = get_mask_extractor(mask)
        add_surface_rendering(mask, i, label_idx)
    return mask


def setup_slicer(renderer: vtk.vtkRenderer,
                 obj: Volume):
    """
    Sets up the slicing widgets
    :param renderer: vtkRenderer-type object
    :param obj: The volume to be sliced
    :return: the list of slicing actors
    """
    # Get the object's extent
    x = obj.extent[1]
    y = obj.extent[3]
    z = obj.extent[5]

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

    # Add actors to the renderer
    renderer.AddActor(axial)
    renderer.AddActor(coronal)
    renderer.AddActor(sagittal)

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
