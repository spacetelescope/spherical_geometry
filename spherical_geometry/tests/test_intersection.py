# STDLIB
import codecs
import functools
import itertools
import math
import os
from os import path
import random

# THIRD-PARTY
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

# LOCAL
from spherical_geometry import polygon
from spherical_geometry.tests.helpers import ROOT_DIR, resolve_imagename

GRAPH_MODE = False


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

            areas = np.array([x.area() for x in polys])

            if GRAPH_MODE:
                print("%d permutations" % num_permutations)

            for i, permutation in enumerate(
                itertools.islice(
                    itertools.permutations(polys),
                    None, None, step_size)):
                filename = '%s_intersection_%04d.svg' % (func.__name__, i)
                print(filename)

                intersection = polygon.SphericalPolygon.multi_intersection(
                    permutation)
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

            lengths = np.array([
                sum(len(x._points) for x in y)
                for y in intersections])
            assert np.all(lengths == [lengths[0]])
            areas = np.array([x.area() for x in intersections])
            assert_array_almost_equal(areas, areas[0], decimal=1)

        return run


@intersection_test(0, 90)
def test1():
    from astropy import wcs as pywcs
    from astropy.io import fits

    filename = resolve_imagename(ROOT_DIR, "1904-66_TAN.fits")
    header = fits.getheader(filename, ext=0)

    wcsobj1 = pywcs.WCS(header)
    wcsobj1.wcs.crval = [0, 87]
    wcsobj2 = pywcs.WCS(header)
    wcsobj2.wcs.crval = [20, 89]

    poly1 = polygon.SphericalPolygon.from_wcs(wcsobj1, 1)
    poly2 = polygon.SphericalPolygon.from_wcs(wcsobj2, 1)

    return [poly1, poly2]


@intersection_test(0, 90)
def test2():
    poly1 = polygon.SphericalPolygon.from_cone(0, 60, 8, steps=16)
    poly2 = polygon.SphericalPolygon.from_cone(0, 68, 8, steps=16)
    poly3 = polygon.SphericalPolygon.from_cone(12, 66, 8, steps=16)
    return [poly1, poly2, poly3]


@intersection_test(0, 90)
def test3():
    from astropy import wcs as pywcs
    from astropy.io import fits

    filename = resolve_imagename(ROOT_DIR, "1904-66_TAN.fits")
    header = fits.getheader(filename, ext=0)

    wcsobj1 = pywcs.WCS(header)
    wcsobj1.wcs.crval = [0, 87]
    wcsobj2 = pywcs.WCS(header)
    wcsobj2.wcs.crval = [175, 89]

    poly1 = polygon.SphericalPolygon.from_wcs(wcsobj1, 1)
    poly3 = polygon.SphericalPolygon.from_wcs(wcsobj2, 1)

    return [poly1, poly3]


@pytest.mark.filterwarnings("ignore:CPERROR.*")
def test4():
    from astropy.io import fits
    from astropy import wcs as pywcs

    with fits.open(os.path.join(ROOT_DIR, '2chipA.fits.gz')) as A:
        wcs = pywcs.WCS(A[1].header, fobj=A)
        chipA1 = polygon.SphericalPolygon.from_wcs(wcs)
        wcs = pywcs.WCS(A[4].header, fobj=A)
        chipA2 = polygon.SphericalPolygon.from_wcs(wcs)

    with fits.open(os.path.join(ROOT_DIR, '2chipB.fits.gz')) as B:
        wcs = pywcs.WCS(B[1].header, fobj=B)
        chipB1 = polygon.SphericalPolygon.from_wcs(wcs)
        wcs = pywcs.WCS(B[4].header, fobj=B)
        chipB2 = polygon.SphericalPolygon.from_wcs(wcs)

    Apoly = chipA1.union(chipA2)
    Bpoly = chipB1.union(chipB2)

    _ = Apoly.intersection(Bpoly)


@intersection_test(0, 90)
def test6():
    from astropy.io import fits
    filename = resolve_imagename(ROOT_DIR,'1904-66_TAN.fits')
    header = fits.getheader(filename, ext=0)

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

    assert len(p2.polygons) == 0


def test_difficult_intersections():
    # Tests a number of intersections of real data that have been
    # problematic in previous revisions of spherical_geometry

    # def test_intersection(polys):
    #     A, B = polys
    #     A.intersection(B)

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
        # yield test_intersection, (polyA, polyB)
        polyA.intersection(polyB)


def test_self_intersection():
    # Tests intersection between a disjoint polygon and itself
    ra1 = [150.15056635,  150.18472797,  150.18472641, 150.15056557, 150.15056635]
    dec1 = [2.33675579,  2.33675454,  2.30262137,  2.3026226 ,  2.33675579]
    ra2 = [150.18472955,  150.18472798,  150.15056635, 150.15056714, 150.18472955]
    dec2 = [2.37105428,  2.33692121,  2.33692245,  2.37105554,  2.37105428]
    # create a union polygon
    s1 = polygon.SphericalPolygon.from_radec(np.array(ra1), np.array(dec1))
    s2 = polygon.SphericalPolygon.from_radec(np.array(ra2), np.array(dec2))
    s12 = s2.union(s1)
    # asserts self-intersection is same as original
    s12int = s12.intersection(s12)
    assert (abs(s12.area() - s12int.area()) < 1.0e-6)
    # same, with multi_intersection method
    s12int = polygon.SphericalPolygon.multi_intersection([s12, s12, s12])
    assert (abs(s12.area() - s12int.area()) < 1.0e-6)


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
    for i in range(nrepeat):
        C = A.intersection(B)
        areas.append(C.area())
    areas = np.array(areas)
    assert_array_almost_equal(areas[:-1], areas[1:])

    def roll_polygon(P, i):
        polygons = []
        for p in P.polygons:
            points = p.points
            points = np.roll(points[:-1], i, 0)
            points = np.append(points, [points[0]], 0)
            p = polygon.SingleSphericalPolygon(points, p.inside)
            polygons.append(p)
        return polygon.SphericalPolygon(polygons)

    Aareas = []
    Bareas = []
    Careas = []
    for i in range(nrepeat):
        AS = roll_polygon(A, i)
        BS = roll_polygon(B, i)

        C = AS.intersection(BS)

        Aareas.append(A.area())
        Bareas.append(B.area())
        Careas.append(C.area())

        for j in range(nrepeat):
            CS = roll_polygon(C, j)
            Careas.append(CS.area())

    Aareas = np.array(Aareas)
    Bareas = np.array(Bareas)
    Careas = np.array(Careas)
    assert_array_almost_equal(Aareas[:-1], Aareas[1:])
    assert_array_almost_equal(Bareas[:-1], Bareas[1:])
    assert_array_almost_equal(Careas[:-1], Careas[1:])


def test_intersection_crash():
    # Reported by Darren White

    testpoints = np.array(
        [[ 0.3583051 ,  0.04329773,  0.9326    ],
         [ 0.34935525,  0.09059884,  0.9326    ],
         [ 0.41458613,  0.10294118,  0.90416893],
         [ 0.42353598,  0.05564007,  0.90416893],
         [ 0.3583051 ,  0.04329773,  0.9326    ]])

    testcenter = np.array(
        [ 0.38681003,  0.07318841,  0.91925049])

    polypoints = np.array(
        [[ 0.4246086 ,  0.04379716,  0.90431706],
         [ 0.41771239,  0.08700592,  0.90440386],
         [ 0.35406757,  0.06995283,  0.9326    ],
         [ 0.34465183,  0.10538573,  0.93279631],
         [ 0.29364251,  0.08598585,  0.95204018],
         [ 0.27687092,  0.13007964,  0.95206186],
         [ 0.23924242,  0.10865033,  0.96486174],
         [ 0.21881505,  0.14456566,  0.96499779],
         [ 0.19972446,  0.12927807,  0.97128643],
         [ 0.17211517,  0.16374755,  0.97137177],
         [ 0.16418552,  0.15451288,  0.97425299],
         [ 0.12763925,  0.18584761,  0.97425299],
         [ 0.17303981,  0.2387991 ,  0.95552719],
         [ 0.20958607,  0.20746437,  0.95552719],
         [ 0.1861715 ,  0.18013143,  0.96586378],
         [ 0.22383787,  0.21031346,  0.95166426],
         [ 0.253951  ,  0.17275425,  0.95166426],
         [ 0.25269325,  0.17174401,  0.95218177],
         [ 0.27532947,  0.18462841,  0.94345431],
         [ 0.29916705,  0.1428042 ,  0.94345431],
         [ 0.28233717,  0.13319738,  0.95002325],
         [ 0.34001511,  0.15514004,  0.92753506],
         [ 0.35358851,  0.11967283,  0.92771417],
         [ 0.40573405,  0.1336354 ,  0.90416893],
         [ 0.41710657,  0.0913497 ,  0.90425513],
         [ 0.48033933,  0.10150229,  0.87118965],
         [ 0.48707009,  0.05997741,  0.871301  ],
         [ 0.4841862 ,  0.08268655,  0.87104917],
         [ 0.54534645,  0.09006864,  0.8333576 ],
         [ 0.55111516,  0.04227516,  0.8333576 ],
         [ 0.48995491,  0.03489307,  0.87104917],
         [ 0.48779298,  0.05394194,  0.87129115],
         [ 0.4246086 ,  0.04379716,  0.90431706]])
    polycenter = np.array(
        [ 0.16877166,  0.19684143,  0.96579997])

    testFoV = polygon.SphericalPolygon(testpoints, inside=testcenter)
    poly = polygon.SphericalPolygon(polypoints, inside=polycenter)

    _ = poly.overlap(testFoV)


@pytest.mark.xfail(reason="currently there is no solution to get this to pass")
def test_intersection_crash_similar_poly():
    p1 = polygon.SphericalPolygon(
        np.array([[-0.1094946215827374, -0.8592766830993238, -0.499654390280199 ],
                  [-0.1089683641318892, -0.8595220381654031, -0.4993473355555343],
                  [-0.108610535224965 , -0.8593183788298407, -0.4997756250993051],
                  [-0.1091500557209236, -0.8590667764452905, -0.5000905307482003],
                  [-0.1094946215827374, -0.8592766830993238, -0.499654390280199 ]]),
        np.array([-0.1090595793730483, -0.8592979843505629, -0.4997128998115153])
    )

    p2 = polygon.SphericalPolygon(
        np.array([[-0.1094946213367254, -0.8592766831114167, -0.4996543903133135],
                  [-0.1089683641834766, -0.859522038038747 , -0.4993473357622887],
                  [-0.1086105354789061, -0.8593183788183577, -0.4997756250638628],
                  [-0.109150055669766 , -0.8590667765760884, -0.5000905305346783],
                  [-0.1094946213367254, -0.8592766831114167, -0.4996543903133135]]),
        np.array([-0.1090595793730483, -0.8592979843505629, -0.4997128998115153])
    )

    p3 = p1.intersection(p2)

    pts1 = np.sort(list(p1.points)[0][:-1], axis=0)
    pts3 = np.sort(list(p3.points)[0][:-1], axis=0)
    assert_allclose(pts1, pts3, rtol=0, atol=1e-15)


# TODO make this test pass
@pytest.mark.xfail(reason="https://github.com/spacetelescope/spherical_geometry/issues/278")
def test_complement_regression():
    """https://github.com/spacetelescope/spherical_geometry/issues/278"""

    p1 = polygon.SingleSphericalPolygon.from_cone(90, 0, 100)
    p2 = polygon.SingleSphericalPolygon.from_cone(270, 0, 100)

    assert p1.contains_lonlat(0, 0)
    assert p2.contains_lonlat(0, 0)

    p12 = p1.intersection(p2)

    assert p12.contains_lonlat(0, 0)


@pytest.mark.parametrize(
    "polygons",
    [
        [
            polygon.SphericalPolygon.from_lonlat(lon=lon, lat=lat, degrees=False)
            for (lon, lat) in [
                ([3.0, 5.0, 5.0, 3.0], [7.0, 7.0, 10.0, 10.0]),
                ([3.0, 4.0, 5.0, 4.0], [5.0, 5.0, 8.0, 8.0]),
                ([1.0, 4.0, 4.0, 1.0], [6.0, 6.0, 9.0, 9.0]),
            ]
        ],
        pytest.param(
            [
                polygon.SphericalPolygon.from_cone(
                    lat=lat,
                    lon=lon,
                    radius=radius,
                    degrees=True,
                    steps=16,
                )
                for (lat, lon, radius) in [
                    (40.5785, -122.4005, 126.08093425137112),
                    (-33.5605, -70.5815, 90.64909777330483),
                    (38.9005, -77.0505, 92.53797630605003),
                    (45.0905, 7.6575, 108.34122674041656),
                    (-25.7795, -56.4515, 73.960984449381),
                ]
            ],
            # TODO make this test pass
            marks=pytest.mark.xfail(
                reason="https://github.com/spacetelescope/spherical_geometry/issues/292"
            ),
        ),
    ],
)
def test_commutative_intersection(polygons):
    """https://github.com/spacetelescope/spherical_geometry/issues/292"""

    # pair polygons with their areas for later sorting
    polygons = [(polygon, polygon.area()) for polygon in polygons]

    # Compute the iterative intersection of all polygons
    unsorted_intersection = polygons[0][0]
    for region in polygons[1:]:
        assert unsorted_intersection.intersects_poly(region[0])
        unsorted_intersection = unsorted_intersection.intersection(region[0])

    unsorted_intersection_area = unsorted_intersection.area()

    # Sort the polygons by their areas
    # Recompute their mutual intersection area
    polygons.sort(key=lambda x: x[1])
    sorted_intersection = polygons[0][0]
    for region in polygons[1:]:
        assert sorted_intersection.intersects_poly(region[0])
        sorted_intersection = sorted_intersection.intersection(region[0])

    sorted_intersection_area = sorted_intersection.area()

    assert_allclose(unsorted_intersection_area, sorted_intersection_area)


@pytest.mark.xfail(reason="currently there is no solution to get this to pass")
def test_intersection_order_independent_from_large_cones():
    """
    Intersections of several polygons should be independent of the order of the
    intersections, e.g., (A^B)^C should give the same result as (A^C)^B.

    Based on user report in https://github.com/spacetelescope/spherical_geometry/issues/292
    Possibly related to https://github.com/spacetelescope/spherical_geometry/issues/278

    """
    #TODO: It would be useful to check that intersection polygons have the same
    # number of vertices and same vertex coordinates, although this can be
    # tricky as theoretically these vertices could be partially different,
    # e.g., some polygons could have extra vertices lying on the arc connecting
    # two adjacent vertices that are present in both orders of intersection.
    # These polygons would be equivalent but we currently do not a way to
    # detect this.

    cone0 = {'lat': 40.5785, 'lon': -122.4005, 'radius': 126.08093425137112}
    cone1 = {'lat': -33.5605, 'lon': -70.5815, 'radius': 90.64909777330483}
    cone2 = {'lat': 38.9005, 'lon': -77.0505, 'radius': 92.53797630605003}
    cone3 = {'lat': 45.0905, 'lon': 7.6575, 'radius': 108.34122674041656}
    cone4 = {'lat': -25.7795, 'lon': -56.4515, 'radius': 73.960984449381}

    p0 = polygon.SphericalPolygon.from_cone(**cone0, degrees=True, steps=22)
    p1 = polygon.SphericalPolygon.from_cone(**cone1, degrees=True, steps=22)
    p2 = polygon.SphericalPolygon.from_cone(**cone2, degrees=True, steps=22)
    p3 = polygon.SphericalPolygon.from_cone(**cone3, degrees=True, steps=22)
    p4 = polygon.SphericalPolygon.from_cone(**cone4, degrees=True, steps=22)

    # case 1 (theoretical area: 3.925526452880896):
    theor_area = 3.925526452880896
    p01_2 = (p0.intersection(p1)).intersection(p2)
    p02_1 = (p0.intersection(p2)).intersection(p1)
    # The area of the intersection should be independent of the order of the
    # intersections.
    assert_allclose(p01_2.area(), p02_1.area(), rtol=0, atol=1e-10)
    # 3.925526452880896 is exact result for cones. When discretized,
    # using a finite number of steps/edges, the area is slightly different,
    # but should be close to this value.
    # TODO: reduce tolerance once the area calculation is more accurate.
    assert_allclose(p02_1.area(), theor_area, rtol=0, atol=0.01)
    assert_allclose(p01_2.area(), theor_area, rtol=0, atol=0.01)

    # case 2 (theoretical area: 3.772239102140351):
    theor_area = 3.772239102140351
    p04_1 = (p0.intersection(p4)).intersection(p1)
    p01_4 = (p0.intersection(p1)).intersection(p4)
    assert_allclose(p04_1.area(), p01_4.area(), rtol=0, atol=1e-10)
    assert_allclose(p04_1.area(), theor_area, rtol=0, atol=1e-10)
    assert_allclose(p01_4.area(), theor_area, rtol=0, atol=1e-10)

    # case 3 (theoretical area: 3.043038701225517):
    theor_area = 3.043038701225517
    p13_4 = (p1.intersection(p3)).intersection(p4)
    p14_3 = (p1.intersection(p4)).intersection(p3)
    assert_allclose(p13_4.area(), p14_3.area(), rtol=0, atol=1e-10)
    assert_allclose(p13_4.area(), theor_area, rtol=0, atol=1e-10)
    assert_allclose(p14_3.area(), theor_area, rtol=0, atol=1e-10)


@pytest.mark.xfail(reason="currently there is no solution to get this to pass")
def test_intersection_order_with_repeats_from_small_cones():
    """
    Intersections of several polygons with some repeats, e.g., A^B^C^C^B should
    give the same result as A^B^C and should be independent of the order of the
    intersections, e.g., (A^B)^C should give the same result as (A^C)^B.

    Inspired from https://github.com/spacetelescope/spherical_geometry/issues/292
    """
    #TODO: It would be useful to check that intersection polygons have the same
    # number of vertices and same vertex coordinates, although this can be
    # tricky as theoretically these vertices could be partially different,
    # e.g., some polygons could have extra vertices lying on the arc connecting
    # two adjacent vertices that are present in both orders of intersection.
    # These polygons would be equivalent but we currently do not a way to
    # detect this.

    cone0 = {'lat': 40.5785, 'lon': -102.4005, 'radius': 46.08093425137112}
    cone1 = {'lat': -33.5605, 'lon': -70.5815, 'radius': 79.64909777330483}
    cone2 = {'lat': 38.9005, 'lon': -77.0505, 'radius': 62.53797630605003}
    cone3 = {'lat': 45.0905, 'lon': 7.6575, 'radius': 58.34122674041656}
    cone4 = {'lat': -25.7795, 'lon': -56.4515, 'radius': 73.960984449381}

    p0 = polygon.SphericalPolygon.from_cone(**cone0, degrees=True, steps=24)
    p1 = polygon.SphericalPolygon.from_cone(**cone1, degrees=True, steps=24)
    p2 = polygon.SphericalPolygon.from_cone(**cone2, degrees=True, steps=24)
    p3 = polygon.SphericalPolygon.from_cone(**cone3, degrees=True, steps=24)
    p4 = polygon.SphericalPolygon.from_cone(**cone4, degrees=True, steps=24)
    polygons = [p0, p1, p2, p3, p4]

    # case 1 (theoretical area: 1.6677035846811172):
    theor_area = 1.6677035846811172
    u = (1, 2, 4, 4, 2)
    p = polygons[u[0]]
    for i in u[1::]:
        p = p.intersection(polygons[i])
    area1 = p.area()

    u = (1, 2, 4)  # same as above but with repeats removed
    pr = polygons[u[0]]
    for i in u[1::]:
        pr = pr.intersection(polygons[i])
    area2 = pr.area()

    assert_allclose(area1, area2, rtol=0, atol=1e-10)
    assert_allclose(area2, theor_area, rtol=0.05, atol=0)
    assert np.vstack(list(p.points)).shape == np.vstack(list(pr.points)).shape

    # case 2 (theoretical area: 0.1394454760460098):
    theor_area = 0.1394454760460098
    u = (0, 1, 2, 2, 3)
    p = polygons[u[0]]
    for i in u[1::]:
        p = p.intersection(polygons[i])
    area1 = p.area()

    u = (0, 1, 2, 3)  # same as above but with repeats removed
    pr = polygons[u[0]]
    for i in u[1::]:
        pr = pr.intersection(polygons[i])
    area2 = pr.area()

    assert_allclose(area1, area2, rtol=0, atol=1e-10)
    assert_allclose(area2, theor_area, rtol=0.05, atol=0)
    assert np.vstack(list(p.points)).shape == np.vstack(list(pr.points)).shape


@pytest.mark.xfail(reason="currently there is no solution to get this to pass")
def test_near_identical_polygons():
    from astropy.wcs import WCS
    from astropy.io import fits

    hdr = fits.Header.fromfile(
        os.path.join(ROOT_DIR, "malformed.hdr"),
        sep="\n",
        endcard=False,
        padding=False
    )

    w = WCS(header=hdr, fix=False)
    w.naxes = [1200, 1200]

    x = np.array([0, 1200, 1200.0, 0])
    y = np.array([0, 0, 1200.0, 1200])

    ra_a1, dec_a1 = w.pixel_to_world_values(x, y)
    ra_a2, dec_a2 = w.pixel_to_world_values(x + 1000, y)
    pa1 = polygon.SphericalPolygon.from_lonlat(ra_a1, dec_a1)
    pa2 = polygon.SphericalPolygon.from_lonlat(ra_a2, dec_a2)
    p_ref = pa1.union(pa2)

    # Add a small amount of noise to the pixel coordinates to create
    # a slightly different polygon:
    np.random.seed(0)
    sigma = 1e-10

    rng = np.random.default_rng(42)
    dxs = rng.normal(0.0, sigma, 4)
    dys = rng.normal(0.0, sigma, 4)

    r, d = w.pixel_to_world_values(x + dxs, y + dys)
    pa = polygon.SphericalPolygon.from_lonlat(r, d)
    p_ref.intersection(pa)
