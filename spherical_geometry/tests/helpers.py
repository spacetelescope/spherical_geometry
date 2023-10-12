import os

import numpy as np

from spherical_geometry import vector

__all__ = ["ROOT_DIR", "get_point_set", "resolve_imagename"]


ROOT_DIR = os.path.join(os.path.dirname(__file__), 'data')


def get_point_set(density=25):
    points = []
    for i in np.linspace(-85, 85, density, True):
        adjusted_density = int(np.cos(np.deg2rad(i)) * density)
        for j in np.linspace(-180, 180, adjusted_density):
            points.append([j, i])
    points = np.asarray(points)
    return np.dstack(vector.radec_to_vector(points[:,0], points[:,1]))[0]


def resolve_imagename(root, base_name):
    """Resolve image name for tests."""

    image_name = os.path.join(root, base_name)

    # Is it zipped?
    if not os.path.exists(image_name):
        image_name = image_name.replace('.fits', '.fits.gz')

    return image_name
