# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This contains the code that does the actual unioning of regions.
"""
# TODO: Weak references for memory management problems?

# STDLIB
import inspect
import math
import weakref

# THIRD-PARTY
import numpy as np

# LOCAL
from spherical_geometry import great_circle_arc as gca
from spherical_geometry import vector
from spherical_geometry.polygon import (
    MalformedPolygonError,
    SingleSphericalPolygon,
    SphericalPolygon,
)

__all__ = ["Graph"]

# Set to True to enable some sanity checks
DEBUG = True

# tolerance for comparing points
NODE_EPS = 2.0**(-32)
NRM_EPS = 1.0e-16 * NODE_EPS  # for projection normalization
ANG_EPS = 0.0  # skip near-zero angles to avoid backtracking

# '_node_order()' and '_edge_order()' functions are called by sorted to provide
# a consistent ordering of nodes and edges retrieved from the graph,
# since values are retrieved from sets in an order that varies from run to run:
def _point_key(point):
    return (float(point[0]), float(point[1]), float(point[2]))

# TODO: Consider using _point_key for consistent ordering instead of hash-based
# ordering. Needs to guard against None and NaN values (maybe in _add_node or
# Node initializer?)
#
# def _node_order(node):
#     return _point_key(node._point)

# def _edge_order(edge):
#     a, b = _node_order(edge._nodes[0]), _node_order(edge._nodes[1])
#     return (a, b) if a <= b else (b, a)   # commutative AND injective

def _node_order(node):
    return hash(tuple(node._point))


def _edge_order(edge):
    return _node_order(edge._nodes[0]) + _node_order(edge._nodes[1])


def _malformed_polygon_error(msg):
    frame = inspect.currentframe()
    try:
        if frame is not None and frame.f_back is not None:
            line = frame.f_back.f_lineno
        else:
            line = "unknown"
    finally:
        del frame

    raise MalformedPolygonError(
        f"{msg} in module: \"{__name__}\" at line: {line}"
    )


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
        def __init__(self, point, source_polygons=None):
            """
            Parameters
            ----------
            point : 3-sequence (*x*, *y*, *z*) coordinate

            source_polygon : `~spherical_geometry.polygon.SphericalPolygon` instance, optional
                The polygon(s) this node came from.  Used for bookkeeping.
            """
            self._point = np.asanyarray(point)
            self._source_polygons = set() if source_polygons is None else set(source_polygons)
            self._edges = weakref.WeakSet()

        def __repr__(self):
            return f"Node({self._point!s} {len(self._edges)})"

        def equals(self, other, thresh=NODE_EPS):
            """
            Returns `True` if the location of this and the *other*
            `~Graph.Node` are the same.

            Parameters
            ----------
            other : `~Graph.Node` instance
                The other node.

            thresh : float
                If difference is smaller than this, points are equal.
                The default value of 1e-10 radians is set based on
                empirical test cases. Relative threshold based on
                the actual sizes of polygons is not implemented.
            """
            # return np.array_equal(self._point, other._point)
            return np.linalg.norm(self._point - other._point) < thresh
            # Possibly more accurate but much slower.
            # TODO: do more testing once the main stability issues are dealt with.
            # return math_util.length(self._point, other._point) < thresh

    class Edge:
        """
        An `~Graph.Edge` represents a connection between exactly two
        `~Graph.Node` objects.  This `~Graph.Edge` class has no direction.
        """
        def __init__(self, A, B, source_polygons=None):
            """
            Parameters
            ----------
            A, B : `~Graph.Node` instances

            source_polygons : sequence of `~spherical_geometry.polygon.SphericalPolygon` instances, optional
                The polygons this edge came from.  Used for bookkeeping.
            """
            self._nodes = [A, B]
            for node in self._nodes:
                node._edges.add(self)
            self._source_polygons = set() if source_polygons is None else set(source_polygons)

        def __repr__(self):
            nodes = self._nodes
            return f"Edge({nodes[0]._point} -> {nodes[1]._point})"

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
                raise RuntimeError("Following from disconnected node")

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

        if len(points) < 4 or polygon._degenerate:
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

    def _add_node(self, point, source_polygons=None):
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
        # than NODE_EPS will cause numerical problems in the
        # intersection calculations, so we merge any nodes that
        # are closer together than that.

        # Don't add nodes that already exist.  Update the existing
        # node's source_polygons list to include the new polygon.

        point = vector.normalize_vector(point)

        if len(self._nodes):
            nodes = list(self._nodes)
            node_array = np.array([node._point for node in nodes])

            diff = np.linalg.norm(node_array - point, axis=-1) < NODE_EPS
            # diff = np.all(np.abs(point - node_array) < NODE_EPS, axis=-1)

            # TODO: consider alternatives such as using the arccos:
            # v1 . v2 = cos(angle), so angle = arccos(v1 . v2)
            # however, arccos can be numerically unstable for angles very close
            # to 0 or pi.
            # diff = np.arccos(np.clip(node_array @ point, -1, 1)) < 100 * NODE_EPS

            # TODO: even more accurate but slower (makes one xfailed test to
            # pass and another test to fail that was expected to pass -
            # different number of vertices - that's OK):
            # diff = gca.length(node_array, point) < NODE_EPS

            indices = np.nonzero(diff)[0]
            if len(indices):
                node = nodes[indices[0]]
                if source_polygons is not None:
                    node._source_polygons.update(source_polygons)
                return node

        if source_polygons is not None:
            source_polygons = set(source_polygons)
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
        for edge in list(node._edges):
            nodeB = edge.follow(node)
            nodeB._edges.remove(edge)
            if len(nodeB._edges) == 0:
                self._nodes.remove(nodeB)
            self._edges.remove(edge)
        if node in self._nodes:
            self._nodes.remove(node)

    def _add_edge(self, A, B, source_polygons=None):
        """
        Add an edge between two nodes.

        .. note::
            It is assumed both nodes already belong to the graph.

        Parameters
        ----------
        A, B : `~Graph.Node` instances

        source_polygons : sequence of `~spherical_geometry.polygon.SphericalPolygon` instances, optional
            The polygons this edge came from.  Used for bookkeeping.

        Returns
        -------
        edge : `~Graph.Edge` instance
            The new edge
        """
        if A not in self._nodes or B not in self._nodes:
            raise ValueError("Nodes not in the graph.")

        # Don't add any edges that already exist.  Update the edge's
        # source polygons list to include the new polygon.  Care needs
        # to be taken here to not create an Edge until we know we need
        # one, otherwise the Edge will get hooked up to the nodes but
        # be orphaned.
        for edge in self._edges:
            # Q: what happens when A and B are the same node?
            # Should we allow self-pointing edges?
            # if A is B and A.equals(B):
            #     # TODO: clarify what to do with self-pointing edges.
            #     # For now, don't add self-pointing edges.
            #     return edge
            if ((A is edge._nodes[0] and
                 B is edge._nodes[1]) or
                (A is edge._nodes[1] and
                 B is edge._nodes[0])):
                # Q: is it possible for an edge to be between the same two
                # nodes but not be the same edge?
                # Q: what happens when A and B are the same node?
                # Should we allow self-pointing edges?
                if source_polygons is not None:
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
        if edge not in self._edges:
            raise ValueError("Edge not in the graph.")

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
        A, B = edge._nodes

        # If node coincides with A or B, do not split
        if node is A or node is B or node.equals(A) or node.equals(B):
            # Return the original edge twice so callers can unpack
            return [edge, edge]

        # Normal split
        edgeA = self._add_edge(A, node, edge._source_polygons)
        edgeB = self._add_edge(node, B, edge._source_polygons)

        valid_edges = []
        for e in (edgeA, edgeB):
            # Zero-length/self-edge -> remove immediately
            if e._nodes[0] is e._nodes[1] or e._nodes[0].equals(e._nodes[1]):
                if e in self._edges:
                    self._remove_edge(e)
            else:
                valid_edges.append(e)

        # If both collapsed, return original edge twice
        if not valid_edges:
            return [edge, edge]

        # If only one valid edge remains, return it twice
        if len(valid_edges) == 1:
            if edge not in valid_edges and edge in self._edges:
                self._remove_edge(edge)
            return [valid_edges[0], valid_edges[0]]

        # Normal case: two valid edges
        if edge not in valid_edges and edge in self._edges:
            self._remove_edge(edge)

        return valid_edges


    def _sanity_check(self, msg, require_even_node_degree=False):
        """
        For debugging purposes: assert that edges and nodes are
        connected to each other correctly and there are no orphaned
        edges or nodes.
        """
        if not DEBUG:
            return

        unique_edges = set()
        for edge in self._edges:
            for node in edge._nodes:
                if edge not in node._edges or node not in self._nodes:
                    _malformed_polygon_error(msg)
            edge_repr = [tuple(x._point) for x in edge._nodes]
            edge_repr.sort()
            edge_repr = tuple(edge_repr)
            assert edge_repr not in unique_edges
            unique_edges.add(edge_repr)

        for node in self._nodes:
            if require_even_node_degree:
                if len(node._edges) % 2 != 0:
                    _malformed_polygon_error(msg)

            else:
                if not len(node._edges) >= 2:
                    _malformed_polygon_error(msg)

            for edge in node._edges:
                if node not in edge._nodes or edge not in self._edges:
                    _malformed_polygon_error(msg)

    def union(self):
        """
        Once all of the polygons have been added to the graph,
        join the polygons together.

        Returns
        -------
        points : Nx3 array of (*x*, *y*, *z*) points
            This is a list of points outlining the union of the
            polygons that were given to the constructor.
        """
        self._find_all_intersections()
        self._sanity_check("union: find all intersections")
        self._remove_interior_edges()
        self._sanity_check("union: remove interior edges")
        self._cleanup_graph(remove_cut_lines=False)
        self._sanity_check("union: cleanup", require_even_node_degree=True)

        return SphericalPolygon((self._trace(),))

    def intersection(self):
        """
        Once all of the polygons have been added to the graph,
        calculate the intersection.

        Returns
        -------
        points : Nx3 array of (*x*, *y*, *z*) points
            This is a list of points outlining the intersection of the
            polygons that were given to the constructor.
        """
        self._find_all_intersections()
        self._sanity_check("intersection: find all intersections")
        self._remove_exterior_edges()
        self._sanity_check("intersection: remove exterior edges")
        self._cleanup_graph(remove_cut_lines=True)
        self._sanity_check("intersection: cleanup", True)

        poly = self._trace()

        if len(poly._polygons) == 1 and not self._contains_inside_point(poly):
            poly = poly.invert_polygon()
        return poly

    def disjoint_polygons(self):
        """
        Convert a graph containing cut lines and self intersections
        into a list of disjoint polygons

        Returns
        -------
        polygons : list of SphericalPolygon
            A list of disjoint polygons obtained from the graph.

        """
        changed = self._remove_cut_lines()
        self._sanity_check("disjoint: remove cut lines")
        changed = self._find_all_intersections() or changed
        self._sanity_check("disjoint: find all intersections")
        changed = self._cleanup_graph() or changed
        self._sanity_check("disjoint: cleanup", require_even_node_degree=True)

        if changed:
            polygons = self._trace_polygons()
        else:
            polygons = list(self._source_polygons)
        return polygons

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

        cut_lines = []
        changed = False
        for edge in self._edges:
            A, B = edge._nodes
            if len(A._edges) == 3 and len(B._edges) == 3:
                cut_lines.append(edge)

        for edge in cut_lines:
            if edge in self._edges:
                A, B = edge._nodes
                if len(A._edges) == 3 and len(B._edges) == 3:
                    self._remove_edge(edge)
                    changed = True

        changed = self._remove_orphaned_nodes() or changed
        return changed

    def _get_edge_points(self, edges):
        if not edges:
            # Return proper 2-D empty arrays so vstack works
            return (np.empty((0, 3), dtype=float),
                    np.empty((0, 3), dtype=float))

        return (np.array([x._nodes[0]._point for x in edges]),
                np.array([x._nodes[1]._point for x in edges]))

    def _find_point_to_arc_intersections(self):
        # For speed, we want to vectorize all of the intersection
        # calculations.  Therefore, there is a list of edges, and an
        # array of points for all of the nodes.  Then calculating the
        # intersection between an edge and all other nodes becomes a
        # fast, vectorized operation.

        edges = sorted(self._edges, key=_edge_order)
        _starts, _ends = self._get_edge_points(edges)

        nodes = sorted(self._nodes, key=_node_order)
        nodes_array = np.array([x._point for x in nodes])

        # Split all edges by any nodes that intersect them
        changed = False
        while len(edges) > 1:
            AB = edges.pop(0)
            A, B = list(AB._nodes)

            intersects = gca.intersects_point(
                A._point, B._point, nodes_array)
            intersection_indices = np.nonzero(intersects)[0]

            for index in intersection_indices:
                node = nodes[index]
                if node not in AB._nodes:
                    changed = True
                    newA, newB = self._split_edge(AB, node)

                    new_edges = [
                        edge for edge in (newA, newB)
                        if edge not in edges]

                    for end_point in AB._nodes:
                        # Q: doesn't 'node' already have its own source polygons?
                        # Why do we need to update it with the end point's source polygons?
                        node._source_polygons.update(
                            end_point._source_polygons)
                    edges = edges + new_edges
                    break
        return changed

    def _find_arc_to_arc_intersections(self):
        # For speed, we want to vectorize all of the intersection
        # calculations.  Therefore, there is a list of edges, and two
        # arrays containing the end points of those edges.  They all
        # need to have things added and removed from them at the same
        # time to keep them in sync, but of course the interface for
        # doing so is different between Python lists and numpy arrays.

        edges = sorted(self._edges, key=_edge_order)
        starts, ends = self._get_edge_points(edges)

        # Calculate edge-to-edge intersections and break
        # edges on the intersection point.
        changed = False
        while len(edges) > 1:
            AB = edges.pop(0)
            A = starts[0]
            starts = starts[1:]  # numpy equiv of "pop(0)"
            B = ends[0]
            ends = ends[1:]      # numpy equiv of "pop(0)"

            # Calculate the intersection points between AB and all
            # other remaining edges
            with np.errstate(invalid="ignore"):
                intersections = gca.intersection(
                    A, B, starts, ends)
            # intersects is `True` everywhere intersections has an
            # actual intersection
            intersects = np.isfinite(intersections[..., 0])

            intersection_indices = np.nonzero(intersects)[0]

            # Iterate through the candidate intersections, if any --
            # we want to eliminate intersections that only intersect
            # at the end points
            for j in intersection_indices:
                changed = True
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
        return changed

    def _find_all_intersections(self):
        """
        Find all the intersecting edges in the graph.  For each
        intersecting pair, four new edges are created around the
        intersection point.
        """
        changed = self._find_arc_to_arc_intersections()
        changed = self._find_point_to_arc_intersections() or changed
        return changed

    def _remove_interior_edges(self):
        """
        Removes any nodes that are contained inside other polygons.
        What's left is the (possibly disjunct) outline.
        """
        changed = False
        polygons = self._source_polygons

        for edge in self._edges:
            edge._count = 0
            A, B = edge._nodes
            for polygon in polygons:
                if (polygon not in edge._source_polygons and
                    ((polygon in A._source_polygons or
                      polygon.contains_point(A._point)) and
                     (polygon in B._source_polygons or
                      polygon.contains_point(B._point))) and
                    polygon.contains_point(
                        gca.midpoint(A._point, B._point))):
                    edge._count += 1

        for edge in list(self._edges):
            if edge._count >= 1:
                self._remove_edge(edge)
                changed = True

        changed = self._remove_orphaned_nodes() or changed
        return changed

    def _remove_exterior_edges(self):
        """
        Removes any edges that are not contained in all of the source
        polygons.  What's left is the (possibly disjunct) outline.
        """
        changed = False
        polygons = self._source_polygons

        for edge in self._edges:
            edge._count = 0
            A, B = edge._nodes
            for polygon in polygons:
                if polygon in edge._source_polygons:
                # TODO: This seems unreliable, especially the midpoint check:
                #       it may fail for concave polygons and also for large
                #       polygons spanning a significant portion of the sphere.
                    edge._count += 1
                elif ((polygon in A._source_polygons or
                       polygon.contains_point(A._point)) and
                      (polygon in B._source_polygons or
                       polygon.contains_point(B._point)) and
                      polygon.contains_point(
                          gca.midpoint(A._point, B._point))):
                    edge._count += 1

        for edge in list(self._edges):
            if edge._count < len(polygons):
                self._remove_edge(edge)
                changed = True

        changed = self._remove_orphaned_nodes() or changed
        return changed

    def _remove_degenerate_edges(self):
        """
        Remove edges where both endpoints are the same point
        """
        changed = False
        removals = []
        for edge in self._edges:
            if edge._nodes[0].equals(edge._nodes[1]):
                removals.append(edge)
                changed = True

        for edge in removals:
            if edge in self._edges:
                self._remove_edge(edge)

        if changed:
            self._remove_orphaned_nodes()

        return changed

    def _remove_3ary_edges(self):
        """
        Remove edges between pairs of nodes that have odd numbers of
        edges.  This removes triangles that can't be traced.
        """
        changed = False
        removals = []
        for edge in self._edges:
            nedges_a = len(edge._nodes[0]._edges)
            nedges_b = len(edge._nodes[1]._edges)

            # Q: Why 3? Why not >= 2 ? When can two nodes be connected by more
            # than one edge? I thought this code no longer uses cut lines,
            # so how can this happen or when 2 lines are allowed?
            if (nedges_a % 2 == 1 and nedges_a >= 3 and
                    nedges_b % 2 == 1 and nedges_b >= 3):
                removals.append(edge)
                changed = True

        for edge in removals:
            if edge in self._edges:
                self._remove_edge(edge)

        if changed:
            self._remove_orphaned_nodes()

        return changed

    def _remove_orphaned_nodes(self):
        """
        Remove nodes with fewer than 2 edges.
        """
        changed = False
        while True:
            removes = []
            for node in list(self._nodes):
                if len(node._edges) < 2:
                    removes.append(node)
                    changed = True
            if len(removes):
                for node in removes:
                    if node in self._nodes:
                        self._remove_node(node)
            else:
                break
        return changed

    def _cleanup_graph(self, remove_cut_lines=False):
        """
        Run all graph-simplification passes to a fixed point.
        """
        changed = True
        any_change = False
        while changed:
            changed = self._remove_degenerate_edges()
            if remove_cut_lines:
                changed = self._remove_cut_lines() or changed
            changed = self._remove_3ary_edges() or changed
            changed = self._remove_orphaned_nodes() or changed
            any_change = any_change or changed
        return any_change

    def _contains_inside_point(self, poly):
        """
        Check if the polygons in the graph all contain
        the interior point of a polygon
        """
        for point in poly.inside:
            for source_poly in self._source_polygons:
                if not source_poly.contains_point(point):
                    return False
        return True

    def _trace_polygons(self):
        """
        Given a graph that has had cutlines removed and all
        intersections found, traces it to find a list of
        disjoint polygons.

        Assumes:
        - graph represents simple (non-self-intersecting) spherical polygons
        - edges are great-circle arcs between unit 3D points
        - polygons are not larger than a hemisphere (recommended)

        """

        def edge_dir(node, edge):
            """
            Direction of edge when leaving `node`, projected to the tangent
            plane at `node._point` and normalized.
            """
            p = node._point
            q = edge.follow(node)._point  # other endpoint

            # project q onto tangent plane at p
            v = q - np.dot(q, p) * p
            nrm = np.linalg.norm(v)
            if nrm < NRM_EPS:
                return v
            return v / nrm

        def signed_turn_angle(node, last_edge, next_edge):
            """
            Signed angle from last_edge to next_edge around the normal `node._point`.
            Positive = left turn when looking along node._point.
            """
            p = node._point
            d0 = edge_dir(node, last_edge)
            d1 = edge_dir(node, next_edge)

            # atan2( (d0 x d1)·p , d0·d1 )
            cross = np.cross(d0, d1)
            sin_theta = np.dot(cross, p)
            cos_theta = np.dot(d0, d1)
            angle = math.atan2(sin_theta, cos_theta)

            # normalize to [0, 2π)
            if angle < 0.0:
                angle += 2.0 * math.pi

            return angle

        def pick_next_edge(node, last_edge):
            """
            Pick the next edge when arriving at `node` from `last_edge`.
            Deterministic: independent of set / WeakSet iteration order.
            """
            candidates = [e for e in node._edges if not e._followed]
            if not candidates:
                raise ValueError("No more edges to follow at node")

            # Primary determinism: a fixed order on the candidate edges,
            # keyed by the coordinate of the far endpoint.
            candidates.sort(key=lambda e: _point_key(e.follow(node)._point))

            if last_edge is None or len(candidates) == 1:
                return candidates[0]

            def turn_key(e):
                ang = signed_turn_angle(node, last_edge, e)
                # Push backtracking / collinear edges to the back so a reversal
                # is only taken when it is the sole remaining option.
                if ang <= ANG_EPS:
                    ang += 2.0 * math.pi
                # Secondary key breaks genuine ties (duplicate / overlapping
                # edges produced by intersection splitting).
                return (ang, _point_key(e.follow(node)._point))

            return min(candidates, key=turn_key)

        polygons = []
        edges = sorted(self._edges, key=_edge_order)
        for edge in self._edges:
            edge._followed = False

        while edges:
            points = []

            # start from the first unused edge
            edge = edges.pop(0)
            if edge._followed:
                continue
            edge._followed = True

            # Deterministic traversal direction: enter the ring from the
            # endpoint with the smaller point key, so a given cycle is always
            # walked the same way regardless of which Edge object we seeded
            # from or how `_nodes` happened to be ordered.
            n0, n1 = edge._nodes
            if _point_key(n0._point) <= _point_key(n1._point):
                start_node, node = n0, n1
            else:
                start_node, node = n1, n0

            points.append(start_node._point)
            points.append(node._point)

            while True:
                if not np.array_equal(points[-1], node._point):
                    points.append(node._point)

                next_edge = pick_next_edge(node, edge)
                next_edge._followed = True
                try:
                    edges.remove(next_edge)
                except ValueError:
                    pass

                edge = next_edge
                node = edge.follow(node)

                if node is start_node:
                    points.append(node._point)
                    break

            polygon = SingleSphericalPolygon(points)
            # TODO: consider ensuring consistent winding order for polygons
            # if polygon.is_clockwise():
            #     polygon = SingleSphericalPolygon(points[::-1])
            polygons.append(polygon)

        return polygons

    def _trace(self):
        """
        Given a graph that has had cutlines removed and all
        intersections found, traces it to find a resulting single
        polygon.
        """
        return SphericalPolygon(self._trace_polygons())
