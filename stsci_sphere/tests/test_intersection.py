from __future__ import print_function, absolute_import

# STDLIB
import codecs
import functools
import itertools
import math
import os
import random
import sys

# THIRD-PARTY
from astropy.extern.six.moves import xrange
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
    from astropy.io import fits

    fits = fits.open(resolve_imagename(ROOT_DIR,'1904-66_TAN.fits'))
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
    from astropy.io import fits
    fits = fits.open(resolve_imagename(ROOT_DIR, '1904-66_TAN.fits'))
    header = fits[0].header

    poly1 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[0, 87])
    poly3 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[175, 89])

    return [poly1, poly3]


def test4():
    from astropy.io import fits
    from astropy import wcs as pywcs

    A = fits.open(os.path.join(ROOT_DIR, '2chipA.fits.gz'))
    B = fits.open(os.path.join(ROOT_DIR, '2chipB.fits.gz'))

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


@intersection_test(0, 90)
def test6():
    from astropy.io import fits
    fits = fits.open(resolve_imagename(ROOT_DIR, '1904-66_TAN.fits'))
    header = fits[0].header

    poly1 = polygon.SphericalPolygon.from_wcs(
        header, 1)
    poly2 = polygon.SphericalPolygon.from_wcs(
        header, 1)

    return [poly1, poly2]


def test_intersection_empty():
    p = polygon.SphericalPolygon.from_cone(
        random.randrange(-180, 180),
        random.randrange(20, 90),
        random.randrange(5, 16),
        steps=16)

    p2 = p.intersection(polygon.SphericalPolygon([]))

    assert_array_almost_equal(p2._points, [])


def test_difficult_intersections():
    # Tests a number of intersections of real data that have been
    # problematic in previous revisions of stsci_sphere

    def test_intersection(polys):
        A, B = polys
        A.intersection(B)

    fname = resolve_imagename(ROOT_DIR, "difficult_intersections.txt")
    with open(fname, 'rb') as fd:
        lines = fd.readlines()

    def to_array(line):
        x = np.frombuffer(codecs.decode(line.strip(), 'hex_codec'), dtype='<f8')
        return x.reshape((len(x) // 3, 3))

    for i in range(0, len(lines), 4):
        Apoints, Ainside, Bpoints, Binside = [
            to_array(line) for line in lines[i:i+4]]
        polyA = polygon.SphericalPolygon(Apoints, Ainside)
        polyB = polygon.SphericalPolygon(Bpoints, Binside)
        yield test_intersection, (polyA, polyB)


def test_ordering():
    nrepeat = 10

    A = polygon.SphericalPolygon(
        [[3.532808036921135653e-01, 6.351523005458726834e-01, -6.868582305351954576e-01],
         [3.532781068942476010e-01, 6.351564219435104075e-01, -6.868558064493115456e-01],
         [3.529538811375814156e-01, 6.351027504797477352e-01, -6.870720880104047579e-01],
         [3.533428330964511477e-01, 6.345142927049303161e-01, -6.874157800432978416e-01],
         [3.533486351814376647e-01, 6.345151843837375516e-01, -6.874119745843003670e-01],
         [3.533513056857608414e-01, 6.345111416839894769e-01, -6.874143334620310686e-01],
         [3.536740696809928530e-01, 6.345607036635456666e-01, -6.872025653337667794e-01],
         [3.536713200704008631e-01, 6.345649108795897719e-01, -6.872000954889618818e-01],
         [3.536761865498951884e-01, 6.345656515431040701e-01, -6.871969069700470945e-01],
         [3.536788213460497765e-01, 6.345616140129455296e-01, -6.871992792142280759e-01],
         [3.540056257094351122e-01, 6.346113105009757449e-01, -6.869850810245486938e-01],
         [3.536200722272911379e-01, 6.352081961257413090e-01, -6.866319189293832448e-01],
         [3.536142814048366390e-01, 6.352072452054380314e-01, -6.866357809093986964e-01],
         [3.536116196666648781e-01, 6.352113634102898310e-01, -6.866333419163089813e-01],
         [3.532833767830895755e-01, 6.351574192193063517e-01, -6.868521736876195272e-01],
         [3.532861440234288386e-01, 6.351531838825796861e-01, -6.868546669018701367e-01],
         [3.532808036921135653e-01, 6.351523005458726834e-01, -6.868582305351954576e-01]],
        [3.536414047913637448e-01, 6.348851549491377755e-01, -6.869196436573932196e-01])

    B = polygon.SphericalPolygon(
        [[3.529249199274748783e-01, 6.356925960489819838e-01, -6.865412764158403958e-01],
         [3.533126219535084322e-01, 6.351003877952851040e-01, -6.868898664200949744e-01],
         [3.533173735956686712e-01, 6.351012981906917210e-01, -6.868865805589428053e-01],
         [3.529301898742857047e-01, 6.356935934402119237e-01, -6.865376437853726310e-01],
         [3.532584388080926563e-01, 6.357475490961038700e-01, -6.863188247667159070e-01],
         [3.536441982306618437e-01, 6.351510082118909661e-01, -6.866723948326530769e-01],
         [3.533173735956686712e-01, 6.351012981906917210e-01, -6.868865805589428053e-01],
         [3.533126219535084322e-01, 6.351003877952851040e-01, -6.868898664200949744e-01],
         [3.529898380712340189e-01, 6.350508125724935171e-01, -6.871016225198859351e-01],
         [3.526006883384300017e-01, 6.356389133339014341e-01, -6.867575456003104373e-01],
         [3.529249199274748783e-01, 6.356925960489819838e-01, -6.865412764158403958e-01]],
        [3.532883212044564125e-01, 6.354215160430938258e-01, -6.866053153377369433e-01])

    areas = []
    for i in xrange(nrepeat):
        C = A.intersection(B)
        areas.append(C.area())
    areas = np.array(areas)
    assert_array_almost_equal(areas[:-1], areas[1:])

    def roll_polygon(P, i):
        points = P.points
        points = np.roll(points[:-1], i, 0)
        points = np.append(points, [points[0]], 0)
        return polygon.SphericalPolygon(points, P.inside)

    Aareas = []
    Bareas = []
    Careas = []
    for i in xrange(nrepeat):
        AS = roll_polygon(A, i)
        BS = roll_polygon(B, i)

        C = AS.intersection(BS)

        Aareas.append(A.area())
        Bareas.append(B.area())
        Careas.append(C.area())

        for j in xrange(nrepeat):
            CS = roll_polygon(C, j)
            Careas.append(CS.area())

    Aareas = np.array(Aareas)
    Bareas = np.array(Bareas)
    Careas = np.array(Careas)
    assert_array_almost_equal(Aareas[:-1], Aareas[1:])
    assert_array_almost_equal(Bareas[:-1], Bareas[1:])
    assert_array_almost_equal(Careas[:-1], Careas[1:])


if __name__ == '__main__':
    if '--profile' not in sys.argv:
        GRAPH_MODE = True
        from mpl_toolkits.basemap import Basemap
        from matplotlib import pyplot as plt

    functions = [(k, v) for k, v in globals().items() if k.startswith('test')]
    functions.sort()
    for k, v in functions:
        v()
