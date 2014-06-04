# -*- coding: utf-8 -*-

# Copyright (C) 2011 Association of Universities for Research in
# Astronomy (AURA)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     1. Redistributions of source code must retain the above
#       copyright notice, this list of conditions and the following
#       disclaimer.
#
#     2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials
#       provided with the distribution.
#
#     3. The name of AURA and its representatives may not be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This contains the code that does the actual unioning of regions.
"""
# TODO: Weak references for memory management problems?
from __future__ import absolute_import, division, unicode_literals, print_function

# STDLIB
import itertools
import weakref

# THIRD-PARTY
import numpy as np

# LOCAL
from . import great_circle_arc
from . import vector

# Set to True to enable some sanity checks
DEBUG = True


class Graph:
    """
    A graph of nodes connected by edges.  The graph is used to build
    unions between polygons.

    .. note::
       This class is not meant to be used directly.  Instead, use
       `~sphere.polygon.SphericalPolygon.union` and
       `~sphere.polygon.SphericalPolygon.intersection`.
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

            source_polygon : `~sphere.polygon.SphericalPolygon` instance, optional
                The polygon(s) this node came from.  Used for bookkeeping.
            """
            point = vector.normalize_vector(*point)

            self._point = np.asanyarray(point)
            self._source_polygons = set(source_polygons)
            self._edges = weakref.WeakSet()

        def __repr__(self):
            return "Node(%s %d)" % (str(self._point), len(self._edges))

        def follow(self, edge):
            """
            Follows from one edge to another across this node.

            Parameters
            ----------
            edge : `~Graph.Edge` instance
                The edge to follow away from.

            Returns
            -------
            other : `~Graph.Edge` instance
                The other edge.
            """
            edges = list(self._edges)
            assert len(edges) == 2
            try:
                return edges[not edges.index(edge)]
            except IndexError:
                raise ValueError("Following from disconnected edge")

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
                emphirical test cases. Relative threshold based on
                the actual sizes of polygons is not implemented.
            """
            # return np.array_equal(self._point, other._point)
            return great_circle_arc.length(self._point, other._point,
                                           degrees=False) < thres


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

            source_polygon : `~sphere.polygon.SphericalPolygon` instance, optional
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
        polygons : sequence of `~sphere.polygon.SphericalPolygon` instances
            Build a graph from this initial set of polygons.
        """
        self._nodes = set()
        self._edges = set()
        self._source_polygons = set()
        self._start_node = None

        self.add_polygons(polygons)

    def add_polygons(self, polygons):
        """
        Add more polygons to the graph.

        .. note::
            Must be called before `union` or `intersection`.

        Parameters
        ----------
        polygons : sequence of `~sphere.polygon.SphericalPolygon` instances
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
        polygon : `~sphere.polygon.SphericalPolygon` instance
            Polygon to add to the graph
        """
        points = polygon._points

        if len(points) < 3:
            raise ValueError("Too few points in polygon")

        self._source_polygons.add(polygon)

        start_node = nodeA = self.add_node(points[0], [polygon])
        if self._start_node is None:
            self._start_node = start_node
        for i in range(1, len(points) - 1):
            nodeB = self.add_node(points[i], [polygon])
            # Don't create self-pointing edges
            if nodeB is not nodeA:
                self.add_edge(nodeA, nodeB, [polygon])
                nodeA = nodeB
        # Close the polygon
        self.add_edge(nodeA, start_node, [polygon])

    def add_node(self, point, source_polygons=[]):
        """
        Add a node to the graph.  It will be disconnected until used
        in a call to `add_edge`.

        Parameters
        ----------
        point : 3-sequence (*x*, *y*, *z*) coordinate

        source_polygon : `~sphere.polygon.SphericalPolygon` instance, optional
            The polygon this node came from.  Used for bookkeeping.

        Returns
        -------
        node : `~Graph.Node` instance
            The new node
        """
        new_node = self.Node(point, source_polygons)

        # Don't add nodes that already exist.  Update the existing
        # node's source_polygons list to include the new polygon.
        for node in self._nodes:
            if node.equals(new_node):
                node._source_polygons.update(source_polygons)
                return node

        self._nodes.add(new_node)
        return new_node

    def remove_node(self, node):
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
            self._edges.remove(edge)
        self._nodes.remove(node)

    def add_edge(self, A, B, source_polygons=[]):
        """
        Add an edge between two nodes.

        .. note::
            It is assumed both nodes already belong to the graph.

        Parameters
        ----------
        A, B : `~Graph.Node` instances

        source_polygons : `~sphere.polygon.SphericalPolygon` instance, optional
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
            if ((A.equals(edge._nodes[0]) and
                 B.equals(edge._nodes[1])) or
                (B.equals(edge._nodes[0]) and
                 A.equals(edge._nodes[1]))):
                edge._source_polygons.update(source_polygons)
                return edge

        new_edge = self.Edge(A, B, source_polygons)
        self._edges.add(new_edge)
        return new_edge

    def remove_edge(self, edge):
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
            self.remove_node(A)
        if A is not B:
            B._edges.remove(edge)
            if len(B._edges) == 0:
                self.remove_node(B)
        self._edges.remove(edge)

    def split_edge(self, edge, node):
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
        edgeA = self.add_edge(A, node, edge._source_polygons)
        edgeB = self.add_edge(node, B, edge._source_polygons)
        if edge not in (edgeA, edgeB):
            self.remove_edge(edge)
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
                    assert len(node._edges) == 2
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
                    projection='vandg', func=lambda x: len(x._edges)):
        from mpl_toolkits.basemap import Basemap
        from matplotlib import pyplot as plt
        fig = plt.figure()
        m = Basemap()

        counts = {}
        for node in self._nodes:
            count = func(node)
            counts.setdefault(count, [])
            counts[count].append(list(node._point))

        minx = np.inf
        miny = np.inf
        maxx = -np.inf
        maxy = -np.inf
        for k, v in counts.items():
            v = np.array(v)
            ra, dec = vector.vector_to_radec(v[:, 0], v[:, 1], v[:, 2])
            x, y = m(ra, dec)
            m.plot(x, y, 'o', label=str(k))
            for x0 in x:
                minx = min(x0, minx)
                maxx = max(x0, maxx)
            for y0 in y:
                miny = min(y0, miny)
                maxy = max(y0, maxy)

        for edge in list(self._edges):
            A, B = [x._point for x in edge._nodes]
            r0, d0 = vector.vector_to_radec(A[0], A[1], A[2])
            r1, d1 = vector.vector_to_radec(B[0], B[1], B[2])
            m.drawgreatcircle(r0, d0, r1, d1, color='blue')

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
        self._remove_3ary_edges(large_first=True)
        self._sanity_check("intersection - remove 3ary edges")
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
            for j in xrange(i + 1, len(edges)):
                CD = edges[j]
                if CD not in self._edges:
                    continue
                C, D = CD._nodes
                # To be a cutline, the candidate edges need to run in
                # the opposite direction, hence A == D and B == C, not
                # A == C and B == D.
                if (A.equals(D) and B.equals(C)):
                    # Create new edges A -> D' and C -> B'
                    self.add_edge(
                        A, D.follow(CD).follow(D),
                        AB._source_polygons | CD._source_polygons)
                    self.add_edge(
                        C, B.follow(AB).follow(B),
                        AB._source_polygons | CD._source_polygons)

                    # Remove B and D which are identical to C and A
                    # respectively.  We do not need to remove AB and
                    # CD because this will remove it for us.
                    self.remove_node(D)
                    self.remove_node(B)

                    break

    def _find_all_intersections(self):
        """
        Find all the intersecting edges in the graph.  For each
        intersecting pair, four new edges are created around the
        intersection point.
        """
        def get_edge_points(edges):
            return (np.array([x._nodes[0]._point for x in edges]),
                    np.array([x._nodes[1]._point for x in edges]))

        # For speed, we want to vectorize all of the intersection
        # calculations.  Therefore, there is a list of edges, and two
        # arrays containing the end points of those edges.  They all
        # need to have things added and removed from them at the same
        # time to keep them in sync, but of course the interface for
        # doing so is different between Python lists and numpy arrays.

        edges = list(self._edges)
        starts, ends = get_edge_points(edges)

        # First, split all edges by any nodes that intersect them
        while len(edges) > 1:
            AB = edges.pop(0)
            A = starts[0]; starts = starts[1:]  # numpy equiv of "pop(0)"
            B = ends[0];   ends = ends[1:]      # numpy equiv of "pop(0)"

            distance = great_circle_arc.length(A, B)
            for node in self._nodes:
                if node not in AB._nodes:
                    distanceA = great_circle_arc.length(node._point, A)
                    distanceB = great_circle_arc.length(node._point, B)
                    if np.abs((distanceA + distanceB) - distance) < 1e-8:
                        newA, newB = self.split_edge(AB, node)

                        new_edges = [
                            edge for edge in (newA, newB)
                            if edge not in edges]

                        edges = new_edges + edges
                        new_starts, new_ends = get_edge_points(new_edges)
                        starts = np.vstack(
                            (new_starts, starts))
                        ends = np.vstack(
                            (new_ends, ends))
                        break

        edges = list(self._edges)
        starts, ends = get_edge_points(edges)

        # Next, calculate edge-to-edge intersections and break
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
                C = starts[j]
                D = ends[j]
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

                E = self.add_node(
                    E, AB._source_polygons | CD._source_polygons)
                newA, newB = self.split_edge(AB, E)
                newC, newD = self.split_edge(CD, E)

                new_edges = [
                    edge for edge in (newA, newB, newC, newD)
                    if edge not in edges]

                # Delete CD, and push the new edges to the
                # front so they will be tested for intersection
                # against all remaining edges.
                del edges[j]  # CD
                edges = new_edges + edges
                new_starts, new_ends = get_edge_points(new_edges)
                starts = np.vstack(
                    (new_starts, starts[:j], starts[j+1:]))
                ends = np.vstack(
                    (new_ends, ends[:j], ends[j+1:]))
                break

    def _remove_interior_edges(self):
        """
        Removes any nodes that are contained inside other polygons.
        What's left is the (possibly disjunct) outline.
        """
        polygons = self._source_polygons

        for edge in self._edges:
            edge._count = 0
            for polygon in polygons:
                if (not polygon in edge._source_polygons and
                    polygon.intersects_arc(
                        edge._nodes[0]._point, edge._nodes[1]._point)):
                    edge._count += 1

        for edge in list(self._edges):
            if edge._count >= 1:
                self.remove_edge(edge)

    def _remove_exterior_edges(self):
        """
        Removes any edges that are not contained in all of the source
        polygons.  What's left is the (possibly disjunct) outline.
        """
        polygons = self._source_polygons

        for edge in self._edges:
            edge._count = 0
            for polygon in polygons:
                if (polygon in edge._source_polygons or
                    polygon.intersects_arc(
                        edge._nodes[0]._point, edge._nodes[1]._point)):
                    edge._count += 1

        for edge in list(self._edges):
            if edge._count < len(polygons):
                self.remove_edge(edge)

    def _remove_3ary_edges(self, large_first=False):
        """
        Remove edges between pairs of nodes that have 3 or more edges.
        This removes triangles that can't be traced.
        """
        if large_first:
            max_ary = 0
            for node in self._nodes:
                max_ary = max(len(node._edges), max_ary)
            order = range(max_ary + 1, 2, -1)
        else:
            order = [3]

        for i in order:
            removals = []
            for edge in list(self._edges):
                if (len(edge._nodes[0]._edges) >= i and
                    len(edge._nodes[1]._edges) >= i):
                    removals.append(edge)

            for edge in removals:
                if edge in self._edges:
                    self.remove_edge(edge)

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
                    self.remove_node(node)
            else:
                break

    def _trace(self):
        """
        Given a graph that has had cutlines removed and all
        intersections found, traces it to find a resulting single
        polygon.
        """
        polygons = []
        edges = set(self._edges)  # copy
        seen_nodes = set()
        while len(edges):
            polygon = []
            # Carefully pick out an "original" edge first.  Synthetic
            # edges may not be pointing in the right direction to
            # properly calculate the area.
            for edge in edges:
                if len(edge._source_polygons) == 1:
                    break
            edges.remove(edge)
            start_node = node = edge._nodes[0]
            while True:
                # TODO: Do we need this if clause any more?
                if len(polygon):
                    if not np.array_equal(polygon[-1], node._point):
                        polygon.append(node._point)
                else:
                    polygon.append(node._point)
                edge = node.follow(edge)
                edges.discard(edge)
                node = edge.follow(node)
                if node is start_node:
                    polygon.append(node._point)
                    break

            polygons.append(np.asarray(polygon))

        if len(polygons) == 1:
            return polygons[0]
        elif len(polygons) == 0:
            return []
        else:
            return self._join(polygons)

    def _join(self, polygons):
        """
        If the graph is disjunct, joins the parts with cutlines.

        The closest nodes between each pair that don't intersect
        any other edges are used as cutlines.

        TODO: This is not optimal, because the closest distance
        between two polygons may not in fact be between two vertices,
        but may be somewhere along an edge.
        """
        def do_join(polygons):
            all_polygons = polygons[:]

            skipped = 0

            polyA = polygons.pop(0)
            while len(polygons):
                polyB = polygons.pop(0)

                # If fewer than 3 edges, it's not a polygon,
                # just throw it out
                if len(polyB) < 4:
                    continue

                # Find the closest set of vertices between polyA and
                # polyB that don't cross any of the edges in *any* of
                # the polygons
                closest = np.inf
                closest_pair_idx = (None, None)
                for a in xrange(len(polyA) - 1):
                    A = polyA[a]
                    distances = great_circle_arc.length(A, polyB[:-1])
                    b = np.argmin(distances)
                    distance = distances[b]
                    if distance < closest:
                        B = polyB[b]
                        # Does this candidate line cross other edges?
                        crosses = False
                        for poly in all_polygons:
                            if np.any(
                                great_circle_arc.intersects(
                                    A, B, poly[:-1], poly[1:])):
                                crosses = True
                                break
                        if not crosses:
                            closest = distance
                            closest_pair_idx = (a, b)

                if not np.isfinite(closest):
                    # We didn't find a pair of points that don't cross
                    # something else, so we want to try to join another
                    # polygon.  Defer the current polygon until later.
                    if len(polygons) in (0, skipped):
                        return None
                    polygons.append(polyB)
                    skipped += 1
                else:
                    # Splice the two polygons together using a cut
                    # line
                    a, b = closest_pair_idx
                    new_poly = np.vstack((
                        # polyA up to and including the cut point
                        polyA[:a+1],

                        # polyB starting with the cut point and
                        # wrapping around back to the cut point.
                        # Ignore the last point in polyB, because it
                        # is the same as the first
                        np.roll(polyB[:-1], -b, 0),

                        # The cut point on polyB
                        polyB[b:b+1],

                        # the rest of polyA, starting with the cut
                        # point
                        polyA[a:]
                        ))

                    skipped = 0
                    polyA = new_poly

            return polyA

        for permutation in itertools.permutations(polygons):
            poly = do_join(list(permutation))
            if poly is not None:
                return poly

        raise RuntimeError("Could not find cut points")
