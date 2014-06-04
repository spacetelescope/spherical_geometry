from __future__ import print_function, absolute_import

# STDLIB
import functools
import itertools
import math
import os
import random
import sys

# THIRD-PARTY
import numpy as np
from numpy.testing import assert_array_almost_equal

# LOCAL
from .. import polygon
from .test_shared import resolve_imagename

GRAPH_MODE = False
ROOT_DIR = os.path.join(os.path.dirname(__file__), 'data')


class intersection_test:
    def __init__(self, lon_0, lat_0, proj='ortho'):
        self._lon_0 = lon_0
        self._lat_0 = lat_0
        self._proj = proj

    def __call__(self, func):
        @functools.wraps(func)
        def run(*args, **kwargs):
            if GRAPH_MODE:
                from mpl_toolkits.basemap import Basemap
                from matplotlib import pyplot as plt

            polys = func(*args, **kwargs)

            intersections = []
            num_permutations = math.factorial(len(polys))
            step_size = int(max(float(num_permutations) / 20.0, 1.0))

            areas = [x.area() for x in polys]

            if GRAPH_MODE:
                print("%d permutations" % num_permutations)
            for method in ('parallel', 'serial'):
                for i, permutation in enumerate(
                    itertools.islice(
                        itertools.permutations(polys),
                        None, None, step_size)):
                    filename = '%s_%s_intersection_%04d.svg' % (
                        func.__name__, method, i)
                    print(filename)

                    intersection = polygon.SphericalPolygon.multi_intersection(
                        permutation, method=method)
                    intersections.append(intersection)
                    intersection_area = intersection.area()
                    if GRAPH_MODE:
                        fig = plt.figure()
                        m = Basemap(projection=self._proj,
                                    lon_0=self._lon_0,
                                    lat_0=self._lat_0)
                        m.drawparallels(np.arange(-90., 90., 20.))
                        m.drawmeridians(np.arange(0., 420., 20.))
                        m.drawmapboundary(fill_color='white')

                        intersection.draw(m, color='red', linewidth=3)
                        for poly in permutation:
                            poly.draw(m, color='blue', alpha=0.5)
                        plt.savefig(filename)
                        fig.clear()

                    assert np.all(intersection_area * 0.9 <= areas)

            lengths = np.array([len(x._points) for x in intersections])
            assert np.all(lengths == [lengths[0]])
            areas = np.array([x.area() for x in intersections])
            assert_array_almost_equal(areas, areas[0], decimal=1)

        return run


@intersection_test(0, 90)
def test1():
    import pyfits
    import os

    fits = pyfits.open(resolve_imagename(ROOT_DIR,'1904-66_TAN.fits'))
    header = fits[0].header

    poly1 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[0, 87])
    poly2 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[20, 89])

    return [poly1, poly2]


@intersection_test(0, 90)
def test2():
    poly1 = polygon.SphericalPolygon.from_cone(0, 60, 8, steps=16)
    poly2 = polygon.SphericalPolygon.from_cone(0, 68, 8, steps=16)
    poly3 = polygon.SphericalPolygon.from_cone(12, 66, 8, steps=16)
    return [poly1, poly2, poly3]


@intersection_test(0, 90)
def test3():
    import pyfits
    fits = pyfits.open(resolve_imagename(ROOT_DIR, '1904-66_TAN.fits'))
    header = fits[0].header

    poly1 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[0, 87])
    poly3 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[175, 89])

    return [poly1, poly3]


def test4():
    import pyfits
    import pywcs

    A = pyfits.open(os.path.join(ROOT_DIR, '2chipA.fits.gz'))
    B = pyfits.open(os.path.join(ROOT_DIR, '2chipB.fits.gz'))

    wcs = pywcs.WCS(A[1].header, fobj=A)
    chipA1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(A[4].header, fobj=A)
    chipA2 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(B[1].header, fobj=B)
    chipB1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(B[4].header, fobj=B)
    chipB2 = polygon.SphericalPolygon.from_wcs(wcs)

    Apoly = chipA1.union(chipA2)
    Bpoly = chipB1.union(chipB2)

    X = Apoly.intersection(Bpoly)


def test5():
    import pyfits
    import pywcs

    A = pyfits.open(os.path.join(ROOT_DIR, '2chipA.fits.gz'))
    B = pyfits.open(os.path.join(ROOT_DIR, '2chipB.fits.gz'))

    wcs = pywcs.WCS(A[1].header, fobj=A)
    chipA1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(A[4].header, fobj=A)
    chipA2 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(B[1].header, fobj=B)
    chipB1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(B[4].header, fobj=B)
    chipB2 = polygon.SphericalPolygon.from_wcs(wcs)

    Apoly = chipA1.union(chipA2)
    Bpoly = chipB1.union(chipB2)

    Apoly.overlap(chipB1)


@intersection_test(0, 90)
def test6():
    import pyfits
    fits = pyfits.open(resolve_imagename(ROOT_DIR, '1904-66_TAN.fits'))
    header = fits[0].header

    poly1 = polygon.SphericalPolygon.from_wcs(
        header, 1)
    poly2 = polygon.SphericalPolygon.from_wcs(
        header, 1)

    return [poly1, poly2]


if __name__ == '__main__':
    if '--profile' not in sys.argv:
        GRAPH_MODE = True
        from mpl_toolkits.basemap import Basemap
        from matplotlib import pyplot as plt

    functions = [(k, v) for k, v in globals().items() if k.startswith('test')]
    functions.sort()
    for k, v in functions:
        v()
