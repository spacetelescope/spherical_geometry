"""
SkyLine tests.

:Author: Pey Lian Lim

:Organization: Space Telescope Science Institute

Examples
--------
>>> cd path/to/project
>>> nosetests

"""
from __future__ import absolute_import

from copy import copy
from numpy.testing import assert_almost_equal

from ..vector import vector_to_radec
from ..polygon import SphericalPolygon
from ..skyline import SkyLine

from .test_util import ROOT_DIR
from .test_shared import resolve_imagename


#---------------------------------------#
# Load footprints used by all the tests #
#---------------------------------------#
f_2chipA  = resolve_imagename(ROOT_DIR, '2chipA.fits') # ACS/WFC #1
im_2chipA = SkyLine(f_2chipA)
f_2chipB  = resolve_imagename(ROOT_DIR, '2chipB.fits') # ACS/WFC #2
im_2chipB = SkyLine(f_2chipB)
f_2chipC  = resolve_imagename(ROOT_DIR, '2chipC.fits') # WFC3/UVIS
im_2chipC = SkyLine(f_2chipC)
f_66_tan  = resolve_imagename(ROOT_DIR, '1904-66_TAN.fits')
im_66_tan = SkyLine(f_66_tan, extname='primary')


#----- SHARED FUNCTIONS -----

def same_members(mem1, mem2):
    assert len(mem1) == len(mem2)

    for m in mem1:
        assert m in mem2

    for m in mem2:
        assert m in mem1

def subset_members(mem_child, mem_parent):
    assert len(mem_parent) > len(mem_child)

    for m in mem_child:
        assert m in mem_parent

def subset_polygon(p_child, p_parent):
    """Overlap not working. Do this instead until fixed."""
    assert p_parent.area() >= p_child.area()
    assert p_parent.contains_point(p_child.inside)

def no_polygon(p_child, p_parent):
    """Overlap not working. Do this instead until fixed."""
    assert not p_parent.contains_point(p_child.inside)


#----- MEMBERSHIP -----

def do_member_overlap(im):
    for m in im.members:
        assert_almost_equal(m.polygon.overlap(im), 1.0)

def test_membership():
    do_member_overlap(im_2chipA)
    do_member_overlap(im_2chipB)
    do_member_overlap(im_66_tan)

    assert len(im_2chipA.members) == 1
    assert im_2chipA.members[0].fname == f_2chipA
    assert im_2chipA.members[0].ext == (1,4)


#----- COPY -----

def test_copy():
    a_copy = copy(im_2chipA)
    assert a_copy is not im_2chipA


#----- SPHERICAL POLYGON RELATED -----

def test_sphericalpolygon():
    assert im_2chipA.contains_point(im_2chipA.inside)
    
    assert im_2chipA.intersects_poly(im_2chipB.polygon)
    
    assert im_2chipA.intersects_arc(im_2chipA.inside, im_2chipB.inside)
    
    assert im_2chipA.overlap(im_2chipB) < im_2chipA.overlap(im_2chipA)
    
    assert_almost_equal(im_2chipA.area(), im_2chipB.area())

    ra_A, dec_A = im_2chipA.to_radec()
    for i in xrange(len(im_2chipA.points)):
        p = im_2chipA.points[i]
        ra, dec = vector_to_radec(p[0], p[1], p[2], degrees=True)
        assert_almost_equal(ra_A[i], ra)
        assert_almost_equal(dec_A[i], dec)


#----- WCS -----

def test_wcs():
    wcs = im_2chipA.to_wcs()
    new_p = SphericalPolygon.from_wcs(wcs)
    subset_polygon(im_2chipA, new_p)


#----- UNION -----

def do_add_image(im1, im2):
    u1 = im1.add_image(im2)
    u2 = im2.add_image(im1)

    assert u1.same_points_as(u2)
    same_members(u1.members, u2.members)

    all_mems = im1.members + im2.members
    same_members(u1.members, all_mems)

    subset_polygon(im1, u1)
    subset_polygon(im2, u1)

def test_add_image():
    # Dithered
    do_add_image(im_2chipA, im_2chipB)

    # Not related
    do_add_image(im_2chipA, im_66_tan)


#----- INTERSECTION -----

def do_intersect_image(im1, im2):
    i1 = im1.find_intersection(im2)
    i2 = im2.find_intersection(im1)

    assert i1.same_points_as(i2)
    same_members(i1.members, i2.members)

    if len(i1.points) > 0:
        subset_members(im1.members, i1.members)
        subset_members(im2.members, i1.members)
    
        subset_polygon(i1, im1)
        subset_polygon(i1, im2)

def test_find_intersection():   
    # Dithered
    do_intersect_image(im_2chipA, im_2chipB)

    # Not related
    do_intersect_image(im_2chipA, im_66_tan)


#----- SKYLINE OVERLAP -----

def test_max_overlap():
    max_s, max_a = im_2chipA.find_max_overlap([im_2chipB, im_2chipC, im_66_tan])
    assert max_s is im_2chipB
    assert_almost_equal(max_a, im_2chipA.intersection(im_2chipB).area())

    max_s, max_a = im_2chipA.find_max_overlap([im_2chipB, im_2chipA])
    assert max_s is im_2chipA
    assert_almost_equal(max_a, im_2chipA.area())

def test_max_overlap_pair():
    assert SkyLine.max_overlap_pair(
        [im_2chipB, im_2chipC, im_2chipA, im_66_tan]) == (im_2chipB, im_2chipA)

    assert SkyLine.max_overlap_pair([im_2chipC, im_2chipA, im_66_tan]) is None


#----- INTENDED USE CASE -----

def test_science_1():
    mos, inc, exc = SkyLine.mosaic([im_2chipA, im_2chipB, im_2chipC, im_66_tan])

    assert inc == [f_2chipA, f_2chipB]
    assert exc == [f_2chipC, f_66_tan]

    subset_polygon(im_2chipA, mos)
    subset_polygon(im_2chipB, mos)

    no_polygon(im_2chipC, mos)
    no_polygon(im_66_tan, mos)

def test_science_2():
    """Like `test_science_1` but different input order."""
    mos, inc, exc = SkyLine.mosaic([im_2chipB, im_66_tan, im_2chipC, im_2chipA])

    assert inc == [f_2chipB, f_2chipA]
    assert exc == [f_66_tan, f_2chipC]

    subset_polygon(im_2chipA, mos)
    subset_polygon(im_2chipB, mos)

    no_polygon(im_2chipC, mos)
    no_polygon(im_66_tan, mos)


#----- UNSTABLE -----

def DISABLED_unstable_overlap():
    i1 = im_2chipA.find_intersection(im_2chipB)
    i2 = im_2chipB.find_intersection(im_2chipA)
    
    u1 = im_2chipA.add_image(im_2chipB)
    u2 = im_2chipB.add_image(im_2chipA)

    # failed here before - known bug
    # failure not always the same due to hash mapping
    assert_almost_equal(i1.overlap(u1), 1.0)
    assert_almost_equal(i1.overlap(i2), 1.0)
    assert_almost_equal(u1.overlap(u2), 1.0)
