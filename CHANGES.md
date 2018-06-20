## Version 1.2.3 on 20 June 2018

Every method with lonlat in its name now has an alias with lonlat
replaced by radec.

The class _SingleSphericalPolygon has been renamed to
SingleSphericalPolygon. The former name has been retained as an alias.

The from_lonlat (and from_radec) method is now available in
SingleSphericalPolygon as well as SphericalPolygon.

The methods iter_polygons_flat have been renamed to to __iter__. The
former name has been retained as an alias.
