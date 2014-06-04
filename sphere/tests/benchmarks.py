import os
import sys
import time

import numpy as np
from sphere import *
from test_util import *
from test_shared import resolve_imagename

def point_in_poly_lots():
    image_name = resolve_imagename(ROOT_DIR,'1904-66_TAN.fits')
    
    poly1 = SphericalPolygon.from_wcs(image_name, 64, crval=[0, 87])
    poly2 = SphericalPolygon.from_wcs(image_name, 64, crval=[20, 89])
    poly3 = SphericalPolygon.from_wcs(image_name, 64, crval=[180, 89])

    points = get_point_set(density=25)

    count = 0
    for point in points:
        if poly1.contains_point(point) or poly2.contains_point(point) or \
               poly3.contains_point(point):
            count += 1

    assert count == 5
    assert poly1.intersects_poly(poly2)
    assert not poly1.intersects_poly(poly3)
    assert not poly2.intersects_poly(poly3)

if __name__ == '__main__':
    for benchmark in [point_in_poly_lots]:
        t = time.time()
        sys.stdout.write(benchmark.__name__)
        sys.stdout.write('...')
        sys.stdout.flush()
        benchmark()
        sys.stdout.write(' %.03fs\n' % (time.time() - t))
        sys.stdout.flush()

