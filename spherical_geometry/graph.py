# -*- coding: utf-8 -*-

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This contains the code that does the actual unioning of regions.
"""
# TODO: Weak references for memory management problems?
from __future__ import absolute_import, division, unicode_literals, print_function

# STDLIB
import itertools
from .utils.compat import weakref

# THIRD-PARTY
import numpy as np
from astropy.extern.six.moves import xrange

# LOCAL
from . import great_circle_arc
from . import math_util
from . import vector

__all__ = ['Graph']

# Set to True to enable some sanity checks
DEBUG = True

# The following two functions are called by sorted to provide a consistent
# ordering of nodes and edges retrieved from the graph, since values are
# retrieved from sets in an order that varies from run to run

def node_order(node):
    return hash(tuple(node._point))

def edge_order(edge):
    return node_order(edge._nodes[0]) + node_order(edge._nodes[1])

class Graph:
    """
    A graph of nodes connected by edges.  The graph is used to build
    unions between polygons.

    .. note::
       This class is not meant to be used directly.  Instead, use
       `~spherical_geometry.polygon.SphericalPolygon.union` and
       `~spherical_geometry.polygon.SphericalPolygon.intersection`.
    """

    class Node:
        """
        A `~Graph.Node` represents a single point, connected by an arbitrary
        number of `~Graph.Edge` objects to other `~Graph.Node` objects.
        """
        def __init__(self, point, source_polygons=[]):
            """
            Parameters
            ----------
            point : 3-sequence (*x*, *y*, *z*) coordinate

            source_polygon : `~spherical_geometry.polygon.SphericalPolygon` instance, optional
                The polygon(s) this node came from.  Used for bookkeeping.
            """
            self._point = np.asanyarray(point)
            self._source_polygons = set(source_polygons)
            self._edges = weakref.WeakSet()

        def __repr__(self):
            return "Node(%s %d)" % (str(self._point), len(self._edges))

        def equals(self, other, thres=2e-8):
            """
            Returns `True` if the location of this and the *other*
            `~Graph.Node` are the same.

            Parameters
            ----------
            other : `~Graph.Node` instance
                The other node.

            thres : float
                If difference is smaller than this, points are equal.
                The default value of 2e-8 radians is set based on
                empirical test cases. Relative threshold based on
                the actual sizes of polygons is not implemented.
            """
            return np.array_equal(self._point, other._point)


    class Edge:
        """
        An `~Graph.Edge` represents a connection between exactly two
        `~Graph.Node` objects.  This `~Graph.Edge` class has no direction.
        """
        def __init__(self, A, B, source_polygons=[]):
            """
            Parameters
            ----------
            A, B : `~Graph.Node` instances

            source_polygon : `~spherical_geometry.polygon.SphericalPolygon` instance, optional
                The polygon this edge came from.  Used for bookkeeping.
            """
            self._nodes = [A, B]
            for node in self._nodes:
                node._edges.add(self)
            self._source_polygons = set(source_polygons)

        def __repr__(self):
            nodes = self._nodes
            return "Edge(%s -> %s)" % (nodes[0]._point, nodes[1]._point)

        def follow(self, node):
            """
            Follow along the edge from the given *node* to the other
            node.

            Parameters
            ----------
            node : `~Graph.Node` instance

            Returns
            -------
            other : `~Graph.Node` instance
            """
            nodes = self._nodes
            try:
                return nodes[not nodes.index(node)]
            except IndexError:
                raise ValueError("Following from disconnected node")

        def equals(self, other):
            """
            Returns `True` if the other edge is between the same two nodes.

            Parameters
            ----------
            other : `~Graph.Edge` instance

            Returns
            -------
            equals : bool
            """
            if (self._nodes[0].equals(other._nodes[0]) and
                self._nodes[1].equals(other._nodes[1])):
                return True
            if (self._nodes[1].equals(other._nodes[0]) and
                self._nodes[0].equals(other._nodes[1])):
                return True
            return False


    def __init__(self, polygons):
        """
        Parameters
        ----------
        polygons : sequence of `~spherical_geometry.polygon.SphericalPolygon` instances
            Build a graph from this initial set of polygons.
        """
        self._nodes = set()
        self._edges = set()
        self._source_polygons = set()

        self.add_polygons(polygons)

    def add_polygons(self, polygons):
        """
        Add more polygons to the graph.

        .. note::
            Must be called before `union` or `intersection`.

        Parameters
        ----------
        polygons : sequence of `~spherical_geometry.polygon.SphericalPolygon` instances
            Set of polygons to add to the graph
        """
        for polygon in polygons:
            self.add_polygon(polygon)

    def add_polygon(self, polygon):
        """
        Add a single polygon to the graph.

        .. note::
            Must be called before `union` or `intersection`.

        Parameters
        ----------
        polygon : `~spherical_geometry.polygon.SphericalPolygon` instance
            Polygon to add to the graph
        """
        points = polygon._points

        if len(points) < 3:
            return

        self._source_polygons.add(polygon)

        start_node = nodeA = self._add_node(points[0], [polygon])
        for i in range(1, len(points) - 1):
            nodeB = self._add_node(points[i], [polygon])
            # Don't create self-pointing edges
            if nodeB is not nodeA:
                self._add_edge(nodeA, nodeB, [polygon])
                nodeA = nodeB
        # Close the polygon
        self._add_edge(nodeA, start_node, [polygon])

    def _add_node(self, point, source_polygons=[]):
        """
        Add a node to the graph.  It will be disconnected until used
        in a call to `_add_edge`.

        Parameters
        ----------
        point : 3-sequence (*x*, *y*, *z*) coordinate

        source_polygon : `~spherical_geometry.polygon.SphericalPolygon` instance, optional
            The polygon this node came from.  Used for bookkeeping.

        Returns
        -------
        node : `~Graph.Node` instance
            The new node
        """
        # Any nodes whose Cartesian coordinates are closer together
        # than 2 ** -32 will cause numerical problems in the
        # intersection calculations, so we merge any nodes that
        # are closer together than that.

        # Don't add nodes that already exist.  Update the existing
        # node's source_polygons list to include the new polygon.

        point = vector.normalize_vector(point)

        if len(self._nodes):
            nodes = list(self._nodes)
            node_array = np.array([node._point for node in nodes])

            diff = np.all(np.abs(point - node_array) < 2 ** -32, axis=-1)

            indices = np.nonzero(diff)[0]
            if len(indices):
                node = nodes[indices[0]]
                node._source_polygons.update(source_polygons)
                return node

        new_node = self.Node(point, source_polygons)
        self._nodes.add(new_node)
        return new_node

    def _remove_node(self, node):
        """
        Removes a node and all of the edges that touch it.

        .. note::
            It is assumed that *Node* is already a part of the graph.

        Parameters
        ----------
        node : `~Graph.Node` instance
        """
        assert node in self._nodes

        for edge in list(node._edges):
            nodeB = edge.follow(node)
            nodeB._edges.remove(edge)
            if len(nodeB._edges) == 0:
                self._nodes.remove(nodeB)
            self._edges.remove(edge)
        if node in self._nodes:
            self._nodes.remove(node)

    def _add_edge(self, A, B, source_polygons=[]):
        """
        Add an edge between two nodes.

        .. note::
            It is assumed both nodes already belong to the graph.

        Parameters
        ----------
        A, B : `~Graph.Node` instances

        source_polygons : `~spherical_geometry.polygon.SphericalPolygon` instance, optional
            The polygon(s) this edge came from.  Used for bookkeeping.

        Returns
        -------
        edge : `~Graph.Edge` instance
            The new edge
        """
        assert A in self._nodes
        assert B in self._nodes

        # Don't add any edges that already exist.  Update the edge's
        # source polygons list to include the new polygon.  Care needs
        # to be taken here to not create an Edge until we know we need
        # one, otherwise the Edge will get hooked up to the nodes but
        # be orphaned.
        for edge in self._edges:
            if ((A is edge._nodes[0] and
                 B is edge._nodes[1]) or
                (A is edge._nodes[1] and
                 B is edge._nodes[0])):
                edge._source_polygons.update(source_polygons)
                return edge

        new_edge = self.Edge(A, B, source_polygons)
        self._edges.add(new_edge)
        return new_edge

    def _remove_edge(self, edge):
        """
        Remove an edge from the graph.  The nodes it points to remain intact.

        .. note::
            It is assumed that *edge* is already a part of the graph.

        Parameters
        ----------
        edge : `~Graph.Edge` instance
        """
        assert edge in self._edges

        A, B = edge._nodes
        A._edges.remove(edge)
        if len(A._edges) == 0:
            self._remove_node(A)
        if A is not B:
            B._edges.remove(edge)
            if len(B._edges) == 0:
                self._remove_node(B)
        self._edges.remove(edge)

    def _split_edge(self, edge, node):
        """
        Splits an `~Graph.Edge` *edge* at `~Graph.Node` *node*, removing
        *edge* and replacing it with two new `~Graph.Edge` instances.
        It is intended that *E* is along the original edge, but that is
        not enforced.

        Parameters
        ----------
        edge : `~Graph.Edge` instance
            The edge to split

        node : `~Graph.Node` instance
            The node to insert

        Returns
        -------
        edgeA, edgeB : `~Graph.Edge` instances
            The two new edges on either side of *node*.
        """
        assert edge in self._edges
        assert node in self._nodes

        A, B = edge._nodes
        edgeA = self._add_edge(A, node, edge._source_polygons)
        edgeB = self._add_edge(node, B, edge._source_polygons)
        if edge not in (edgeA, edgeB):
            self._remove_edge(edge)
        return [edgeA, edgeB]

    def _sanity_check(self, title, node_is_2=False):
        """
        For debugging purposes: assert that edges and nodes are
        connected to each other correctly and there are no orphaned
        edges or nodes.
        """
        if not DEBUG:
            return

        try:
            unique_edges = set()
            for edge in self._edges:
                for node in edge._nodes:
                    assert edge in node._edges
                    assert node in self._nodes
                edge_repr = [tuple(x._point) for x in edge._nodes]
                edge_repr.sort()
                edge_repr = tuple(edge_repr)
                # assert edge_repr not in unique_edges
                unique_edges.add(edge_repr)

            for node in self._nodes:
                if node_is_2:
                    assert len(node._edges) % 2 == 0
                else:
                    assert len(node._edges) >= 2
                for edge in node._edges:
                    assert node in edge._nodes
                    assert edge in self._edges
        except AssertionError as e:
            import traceback
            traceback.print_exc()
            self._dump_graph(title=title)
            raise

    def _dump_graph(self, title=None, lon_0=0, lat_0=90,
                    projection='vandg', func=lambda x: len(x._edges),
                    highlight_edge=None, intersections=[]):
        from mpl_toolkits.basemap import Basemap
        from matplotlib import pyplot as plt
        fig = plt.figure()
        m = Basemap()

        minx = np.inf
        miny = np.inf
        maxx = -np.inf
        maxy = -np.inf
        for polygon in self._source_polygons:
            polygon.draw(m, lw=10, alpha=0.1, color="black")
            v = polygon._points
            ra, dec = vector.vector_to_lonlat(v[:, 0], v[:, 1], v[:, 2])
            x, y = m(ra, dec)
            for x0 in x:
                minx = min(x0, minx)
                maxx = max(x0, maxx)
            for y0 in y:
                miny = min(y0, miny)
                maxy = max(y0, maxy)

        counts = {}
        for node in self._nodes:
            count = func(node)
            counts.setdefault(count, [])
            counts[count].append(list(node._point))

        for edge in list(self._edges):
            A, B = [x._point for x in edge._nodes]
            lon0, lat0 = vector.vector_to_lonlat(A[0], A[1], A[2])
            lon1, lat1 = vector.vector_to_lonlat(B[0], B[1], B[2])
            if edge is highlight_edge:
                color = 'red'
                lw = 2
            else:
                color = 'blue'
                lw = 0.5
            m.drawgreatcircle(lon0, lat0, lon1, lat1, color=color, lw=lw)

        for k, v in counts.items():
            v = np.array(v)
            ra, dec = vector.vector_to_lonlat(v[:, 0], v[:, 1], v[:, 2])
            x, y = m(ra, dec)
            m.plot(x, y, 'o', label=str(k))
            for x0 in x:
                minx = min(x0, minx)
                maxx = max(x0, maxx)
            for y0 in y:
                miny = min(y0, miny)
                maxy = max(y0, maxy)

        for v in intersections:
            if np.all(np.isfinite(v)):
                ra, dec = vector.vector_to_lonlat(v[0], v[1], v[2])
                x, y = m(ra, dec)
                m.plot(x, y, 'x', markersize=20)

        plt.xlim(minx, maxx)
        plt.ylim(miny, maxy)
        if title:
            plt.title("%s, %d v, %d e" % (
                title, len(self._nodes), len(self._edges)))
        plt.legend()
        plt.show()

    def union(self):
        """
        Once all of the polygons have been added to the graph,
        join the polygons together.

        Returns
        -------
        points : Nx3 array of (*x*, *y*, *z*) points
            This is a list of points outlining the union of the
            polygons that were given to the constructor.  If the
            original polygons are disjunct or contain holes, cut lines
            will be included in the output.
        """
        self._remove_cut_lines()
        self._sanity_check("union - remove cut lines")
        self._find_all_intersections()
        self._sanity_check("union - find all intersections")
        self._remove_interior_edges()
        self._sanity_check("union - remove interior edges")
        self._remove_degenerate_edges()
        self._sanity_check("union - remove degenerate edges")
        self._remove_3ary_edges()
        self._sanity_check("union - remove 3ary edges")
        self._remove_orphaned_nodes()
        self._sanity_check("union - remove orphan nodes", True)
        return self._trace()

    def intersection(self):
        """
        Once all of the polygons have been added to the graph,
        calculate the intersection.

        Returns
        -------
        points : Nx3 array of (*x*, *y*, *z*) points
            This is a list of points outlining the intersection of the
            polygons that were given to the constructor.  If the
            resulting polygons are disjunct or contain holes, cut lines
            will be included in the output.
        """
        self._remove_cut_lines()
        self._sanity_check("intersection - remove cut lines")
        self._find_all_intersections()
        self._sanity_check("intersection - find all intersections")
        self._remove_exterior_edges()
        self._sanity_check("intersection - remove exterior edges")
        self._remove_cut_lines()
        self._sanity_check("intersection - remove cut lines [2]")
        self._remove_orphaned_nodes()
        self._sanity_check("intersection - remove orphan nodes", True)
        return self._trace()

    def _remove_cut_lines(self):
        """
        Removes any cutlines that may already have existed in the
        input polygons.  This is so any cutlines in the final result
        will be optimized to be as short as possible and won't
        intersect each other.

        This works by finding coincident edges that are reverse to
        each other, and then splicing around them.
        """
        # As this proceeds, edges are removed from the graph.  It
        # iterates over a static list of all edges that exist at the
        # start, so each time one is selected, we need to ensure it
        # still exists as part of the graph.

        # This transforms the following (where = is the cut line)
        #
        #     \                    /
        #  A' +                    + B'
        #     |                    |
        #  A  +====================+ B
        #
        #  D  +====================+ C
        #     |                    |
        #  D' +                    + C'
        #     /                    \
        #
        # to this:
        #
        #     \                    /
        #  A' +                    + B'
        #     |                    |
        #  A  +                    + C
        #     |                    |
        #  D' +                    + C'
        #     /                    \
        #

        edges = list(self._edges)
        for i in xrange(len(edges)):
            AB = edges[i]
            if AB not in self._edges:
                continue
            A, B = AB._nodes
            if len(A._edges) == 3 and len(B._edges) == 3:
                self._remove_edge(AB)

    def _get_edge_points(self, edges):
        return (np.array([x._nodes[0]._point for x in edges]),
                np.array([x._nodes[1]._point for x in edges]))

    def _find_point_to_arc_intersections(self):
        # For speed, we want to vectorize all of the intersection
        # calculations.  Therefore, there is a list of edges, and an
        # array of points for all of the nodes.  Then calculating the
        # intersection between an edge and all other nodes becomes a
        # fast, vectorized operation.

        edges = sorted(self._edges, key=edge_order)
        starts, ends = self._get_edge_points(edges)

        nodes = sorted(self._nodes, key=node_order) ## DBG FIX
        nodes_array = np.array([x._point for x in nodes])

        # Split all edges by any nodes that intersect them
        while len(edges) > 1:
            AB = edges.pop(0)
            A, B = list(AB._nodes)

            intersects = great_circle_arc.intersects_point(
                A._point, B._point, nodes_array)
            intersection_indices = np.nonzero(intersects)[0]

            for index in intersection_indices:
                node = nodes[index]
                if node not in AB._nodes:
                    newA, newB = self._split_edge(AB, node)

                    new_edges = [
                        edge for edge in (newA, newB)
                        if edge not in edges]

                    for end_point in AB._nodes:
                        node._source_polygons.update(
                            end_point._source_polygons)
                    edges = edges + new_edges
                    break

    def _find_arc_to_arc_intersections(self):
        # For speed, we want to vectorize all of the intersection
        # calculations.  Therefore, there is a list of edges, and two
        # arrays containing the end points of those edges.  They all
        # need to have things added and removed from them at the same
        # time to keep them in sync, but of course the interface for
        # doing so is different between Python lists and numpy arrays.

        edges = sorted(self._edges, key=edge_order)
        starts, ends = self._get_edge_points(edges)

        # Calculate edge-to-edge intersections and break
        # edges on the intersection point.
        while len(edges) > 1:
            AB = edges.pop(0)
            A = starts[0]; starts = starts[1:]  # numpy equiv of "pop(0)"
            B = ends[0];   ends = ends[1:]      # numpy equiv of "pop(0)"

            # Calculate the intersection points between AB and all
            # other remaining edges
            with np.errstate(invalid='ignore'):
                intersections = great_circle_arc.intersection(
                    A, B, starts, ends)
            # intersects is `True` everywhere intersections has an
            # actual intersection
            intersects = np.isfinite(intersections[..., 0])

            intersection_indices = np.nonzero(intersects)[0]

            # Iterate through the candidate intersections, if any --
            # we want to eliminate intersections that only intersect
            # at the end points
            for j in intersection_indices:
                CD = edges[j]
                E = intersections[j]

                # This is a bona-fide intersection, and E is the
                # point at which the two lines intersect.  Make a
                # new node for it -- this must belong to the all
                # of the source polygons of both of the edges that
                # crossed.

                #                A
                #                |
                #             C--E--D
                #                |
                #                B

                E = self._add_node(
                    E, AB._source_polygons | CD._source_polygons)
                newA, newB = self._split_edge(AB, E)
                newC, newD = self._split_edge(CD, E)

                new_edges = [
                    edge for edge in (newA, newB, newC, newD)
                    if edge not in edges]

                # Delete CD, and push the new edges to the
                # front so they will be tested for intersection
                # against all remaining edges.
                edges = edges[:j] + edges[j+1:] + new_edges
                new_starts, new_ends = self._get_edge_points(new_edges)
                starts = np.vstack(
                    (starts[:j], starts[j+1:], new_starts))
                ends = np.vstack(
                    (ends[:j], ends[j+1:], new_ends))
                break

    def _find_all_intersections(self):
        """
        Find all the intersecting edges in the graph.  For each
        intersecting pair, four new edges are created around the
        intersection point.
        """
        self._find_point_to_arc_intersections()
        self._find_arc_to_arc_intersections()

    def _remove_interior_edges(self):
        """
        Removes any nodes that are contained inside other polygons.
        What's left is the (possibly disjunct) outline.
        """
        polygons = self._source_polygons

        for edge in self._edges:
            edge._count = 0
            A, B = edge._nodes
            for polygon in polygons:
                if (not polygon in edge._source_polygons and
                    ((polygon in A._source_polygons or
                      polygon.contains_point(A._point)) and
                     (polygon in B._source_polygons or
                      polygon.contains_point(B._point))) and
                    polygon.contains_point(
                        great_circle_arc.midpoint(A._point, B._point))):
                    edge._count += 1

        for edge in list(self._edges):
            if edge._count >= 1:
                self._remove_edge(edge)

        self._remove_orphaned_nodes()

    def _remove_exterior_edges(self):
        """
        Removes any edges that are not contained in all of the source
        polygons.  What's left is the (possibly disjunct) outline.
        """
        polygons = self._source_polygons

        for edge in self._edges:
            edge._count = 0
            A, B = edge._nodes
            for polygon in polygons:
                if polygon in edge._source_polygons:
                    edge._count += 1
                elif ((polygon in A._source_polygons or
                       polygon.contains_point(A._point)) and
                      (polygon in B._source_polygons or
                       polygon.contains_point(B._point)) and
                      polygon.contains_point(
                          great_circle_arc.midpoint(A._point, B._point))):
                    edge._count += 1

        for edge in list(self._edges):
            if edge._count < len(polygons):
                self._remove_edge(edge)

        self._remove_orphaned_nodes()

    def _remove_degenerate_edges(self):
        """
        Remove edges where both endpoints are the same point
        """
        removals = []
        for edge in self._edges:
            if edge._nodes[0].equals(edge._nodes[1]):
                removals.append(edge)
 
        for edge in removals:
            if edge in self._edges:
                self._remove_edge(edge)

    def _remove_3ary_edges(self):
        """
        Remove edges between pairs of nodes that have odd numbers of
        edges.  This removes triangles that can't be traced.
        """
        removals = []
        for edge in self._edges:
            nedges_a = len(edge._nodes[0]._edges)
            nedges_b = len(edge._nodes[1]._edges)
            if (nedges_a % 2 == 1 and nedges_a >= 3 and
                nedges_b % 2 == 1 and nedges_b >= 3):
                removals.append(edge)

        for edge in removals:
            if edge in self._edges:
                self._remove_edge(edge)

    def _remove_orphaned_nodes(self):
        """
        Remove nodes with fewer than 2 edges.
        """
        while True:
            removes = []
            for node in list(self._nodes):
                if len(node._edges) < 2:
                    removes.append(node)
            if len(removes):
                for node in removes:
                    if node in self._nodes:
                        self._remove_node(node)
            else:
                break

    def _trace(self):
        """
        Given a graph that has had cutlines removed and all
        intersections found, traces it to find a resulting single
        polygon.
        """
        from . import polygon as mpolygon

        def pick_next_edge(node, last_edge):
            # Pick the next edge when arriving at a node from
            # last_edge.  If there's only one other edge, the choice
            # is obvious.  If there's more than one, disfavor an edge
            # with the same normal as the previous edge, in order to
            # trace 4-connected nodes into separate distinct shapes
            # and avoid edge crossings.
            candidates = []
            for edge in node._edges:
                if not edge._followed:
                    candidates.append(edge)
            if len(candidates) == 0:
                raise ValueError("No more edges to follow")
            elif len(candidates) == 1:
                return candidates[0]

            for edge in candidates:
                last_edge_cross = math_util.cross(
                    last_edge._nodes[0]._point, last_edge._nodes[1]._point)
                if (not
                    np.all(math_util.cross(
                        edge._nodes[0]._point, edge._nodes[1]._point) ==
                           last_edge_cross) or
                    np.all(math_util.cross(
                        edge._nodes[0]._point, edge._nodes[1]._point) ==
                           last_edge_cross)):
                    return edge

            return candidates[0]

        polygons = []
        edges = set(self._edges)  # copy
        for edge in self._edges:
            edge._followed = False

        while len(edges):
            points = []
            edge = edges.pop()
            start_node = node = edge._nodes[0]
            while True:
                # TODO: Do we need this if clause any more?
                if len(points):
                    if not np.array_equal(points[-1], node._point):
                        points.append(node._point)
                else:
                    points.append(node._point)
                edge = pick_next_edge(node, edge)
                edge._followed = True
                edges.discard(edge)
                node = edge.follow(node)
                if node is start_node:
                    points.append(node._point)
                    break

            polygon = mpolygon._SingleSphericalPolygon(points, auto_orient=True)
            polygons.append(polygon)

        return mpolygon.SphericalPolygon(polygons)
