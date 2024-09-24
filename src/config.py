import csv

TSV_FILE_PATH = 'data/FastSurfer_ColorLUT_95_classes.tsv'
ICON_PATH = 'images/logo.png'


def is_convertible_to_int(value):
    """
    Checks whether a variable is convertible to int

    Parameters
    ----------
    value : any
        The value to be converted

    Returns
    -------
    bool
        True if convertible, False otherwise
    """
    try:
        _ = int(value)
        return True
    except ValueError:
        return False


def get_mask_colors():
    """
    Gets the segmentation masks' lookup table

    Returns
    -------
    dict
        A dictionary with index as keys and (r, g, b) tuples as values
    """
    rgb_values = {}

    with open(TSV_FILE_PATH, 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        for row in reader:
            # Get the index
            idx = row[0].split('.')[0]

            # Assuming the first column is the index and the next three columns are R, G, B values
            if not is_convertible_to_int(idx) or int(idx) == 0:
                continue
            index = int(idx)
            r, g, b = map(int, row[2:-1])
            rgb_values[index] = (r / 255.0, g / 255.0, b / 255.0)
    return rgb_values


# Default brain settings
BRAIN_SMOOTHNESS = 500
BRAIN_OPACITY = 100
BRAIN_COLOR = (1.0, 0.9, 0.9)

# Default mask settings
MASK_SMOOTHNESS = 500
MASK_COLORS = get_mask_colors()
MASK_OPACITY = 100