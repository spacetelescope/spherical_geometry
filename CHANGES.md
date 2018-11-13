## Version 1.2.6 on 13 November 2018

Replaced _naxis with pixel_shape

Updated README file

Removed debugging code from graph.py

## Version 1.2.5 on 10 July 2018

Added a method to create a polygon from the convex hull of a list
of points.

## Version 1.2.4 on 28 June 2018

The public methods in SingleSphericalPolygon now match the methods in
SphericalPolygon so that objects of either type can be used
interchangably (for the most part.) SphericalPolygon now subclasses
SingleSphericalPolygon.

## Version 1.2.3 on 20 June 2018

Every method with lonlat in its name now has an alias with lonlat
replaced by radec.

The class _SingleSphericalPolygon has been renamed to
SingleSphericalPolygon. The former name has been retained as an alias.

The from_lonlat (and from_radec) method is now available in
SingleSphericalPolygon as well as SphericalPolygon.

The methods iter_polygons_flat have been renamed to to __iter__. The
former name has been retained as an alias.
