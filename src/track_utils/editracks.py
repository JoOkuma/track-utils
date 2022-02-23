
from typing import Callable, Container, Optional, List

import numpy as np
from numpy.typing import ArrayLike
import napari

from napari.layers import Tracks, Layer
from napari.layers.tracks._managers._interactive_track_manager import Node, InteractiveTrackManager
from napari.layers.utils.plane import ClippingPlane
from napari.utils.colormaps.standardize_color import _handle_str
from napari.utils.events import EventedModel

from magicgui.widgets import (
    Container, create_widget, SpinBox, Slider, PushButton, CheckBox,
    FileEdit, Textarea
)

from vispy.gloo import VertexBuffer
from vispy.scene.visuals import Markers
from vispy.visuals.filters import Filter

import warnings

from pathlib import Path

from ._writer import tracks_to_dataframe


class StatsManager(EventedModel):
    n_tracks: int = 0
    n_nodes: int = 0
    n_add: int = 0
    n_deleted: int = 0
    n_verified: int = 0

    _stats_tlpt = '# Tracks {n_tracks}\n' \
                  '# Nodes {n_nodes}\n' \
                  '# Additions {n_add}\n' \
                  '# Deleted {n_deleted}\n' \
                  '# Verified {n_verified}'
    
    def __str__(self) -> str:
        return self._stats_tlpt.format(
            n_tracks=self.n_tracks,
            n_nodes=self.n_nodes,
            n_add=self.n_add,
            n_deleted=self.n_deleted,
            n_verified=self.n_verified,
        )


class SpatialFilter(Filter):

    VERT_SHADER = """
        void apply_track_spatial_shading()
        {
            float relative_dif = abs($a_vertex_stack - $current_stack) / $stack_range;
            if ($stack_range != 0)
            {
                v_track_color.a *= clamp(1.0 - relative_dif, 0.0, 1.0);
            }
        }
    """

    FRAG_SHADER = """
        void apply_track_spatial_shading()
        {
            // if the alpha is below the threshold, discard the fragment
            if( v_track_color.a <= 0.0 ) {
                discard;
            }
            // interpolate
            gl_FragColor.a = clamp(v_track_color.a * gl_FragColor.a, 0.0, 1.0);
        }
    """

    def __init__(self, stack_range: float, current_stack: int = 0, vertex_stack: Optional[ArrayLike] = None):
        super().__init__(vcode=self.VERT_SHADER, vpos=4, fcode=self.FRAG_SHADER, fpos=10)

        self.stack_range = stack_range
        self.current_stack = current_stack
        if vertex_stack is None:
            vertex_stack = np.empty((0,), dtype=np.float32)
        self.vertex_stack = vertex_stack
        # TODO, filter according to space
    
    @property
    def stack_range(self) -> float:
        return self._stack_range
    
    @stack_range.setter
    def stack_range(self, value: float) -> None:
        self._stack_range = value
        self.vshader['stack_range'] = float(value)

    @property
    def current_stack(self) -> int:
        return self._current_stack
    
    @current_stack.setter
    def current_stack(self, value: int) -> None:
        self._current_stack = value
        self.vshader['current_stack'] = value

    @property
    def vertex_stack(self) -> ArrayLike:
        return self._vertex_stack
    
    @vertex_stack.setter
    def vertex_stack(self, value: ArrayLike) -> None:
        self._vertex_stack = np.asarray(value).astype(np.float32)
        self.vshader['a_vertex_stack'] = VertexBuffer(self.vertex_stack)


class EditTracks(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer
        self._viewer.dims.events.current_step.connect(self._update_current_slice)
        self._viewer.dims.events.ndisplay.connect(self._set_filtering_status)
        self._viewer.dims.events.order.connect(self._on_tracks_change)

        self._tracks_layer = create_widget(annotation=Tracks, name='Tracks')
        self.append(self._tracks_layer)

        self._pending_additions = []

        self._hover_visual = Markers(scaling=True)
        self._eval_visual = Markers(scaling=True)
        self._neighbor_visual = Markers(scaling=True)
        self._vertices_spatial_filter = SpatialFilter(10.0)
        self._graph_spatial_filter = SpatialFilter(10.0)

        self._node_radius = 10
        self._node_t_trail_length = 10
        self._node_spatial_trail_length = 10

        self._reset_eval_tracking()
        self._reset_neighborhood()
        self._is_adding = False

        self._setup_clipping_planes()

        self._load_button = PushButton(text='Load', enabled=False)
        self._load_button.changed.connect(self._on_load)
        self.append(self._load_button)

        self._node_radius = SpinBox(max=100, value=10, label='Node Radius')
        self.append(self._node_radius)

        self._spatial_range = Slider(min=0, max=500, value=self._vertices_spatial_filter.stack_range,
                                     label='Spatial Range')
        self._spatial_range.changed.connect(lambda e: setattr(self._vertices_spatial_filter, 'stack_range', e.value))
        self._spatial_range.changed.connect(lambda e: setattr(self._graph_spatial_filter, 'stack_range', e.value))
        self.append(self._spatial_range)

        self._neighbor_show = CheckBox(text='Display Neighbors', value=False, enabled=True)
        self._neighbor_show.changed.connect(lambda _: self._load_neighborhood_visual())
        self.append(self._neighbor_show)

        self._neighbor_radius = SpinBox(min=1, max=25, value=10, label='Neigh. Radius')
        self.append(self._neighbor_radius)

        self._bounding_box = CheckBox(text='Bounding Box', value=False, enabled=False)
        self._bounding_box.changed.connect(self._on_bounding_box_change)
        self.append(self._bounding_box)

        self._bbox_size = SpinBox(min=5, max=500, value=25, label='BBox Size')
        self._bbox_size.changed.connect(lambda _: self._update_bbox_position())
        self.append(self._bbox_size)

        self._backward_incr = CheckBox(text='Backward', value=False, enabled=True)
        self.append(self._backward_incr)

        self._addition_mode = PushButton(text='Add New Track', enabled=True)
        self._addition_mode.changed.connect(lambda _: setattr(self, '_is_adding', True))
        self._addition_mode.changed.connect(lambda _: setattr(self, '_inspecting', False))
        self._addition_mode.changed.connect(lambda _: setattr(self, '_eval_node', None))

        self.append(self._addition_mode)
        self._add_confirm = PushButton(text='Confirm')
        self._add_confirm.changed.connect(lambda _: self._confirm_addition())
        self._add_cancel = PushButton(text='Cancel')
        self._add_cancel.changed.connect(lambda _: self._cancel_addition())
 
        container = Container(layout='horizontal', widgets=[self._add_confirm, self._add_cancel], label='Addition: ')
        self.append(container)

        self._save_dialog = FileEdit(mode='w', filter='*.csv', label='Save Verified', enabled=False)
        self._save_dialog.changed.connect(self._save_verified)
        self.append(self._save_dialog)

        self._stats = Textarea(label='Stats', enabled=False)
        self.append(self._stats)

        self._stats_manager = StatsManager()       
        self._stats_manager.events.n_tracks.connect(self._update_stats)
        self._stats_manager.events.n_nodes.connect(self._update_stats)
        self._stats_manager.events.n_add.connect(self._update_stats)
        self._stats_manager.events.n_deleted.connect(self._update_stats)
        self._stats_manager.events.n_verified.connect(self._update_stats)

        self._update_stats()

        shortcuts_txt = "M: Next node\n" +\
                        "N: Previous node\n" +\
                        "C: Confirm node\n" +\
                        "X: Unconfirm node\n" +\
                        "L: Next neighbor\n" +\
                        "K: Previous neighbor\n" +\
                        "A: Add link to neighbor\n" +\
                        "S: Switch link to neighbor\n" +\
                        "F: Focus to current track node\n" +\
                        "D: Delete node\n" +\
                        "Esc: Escape selected track\n"

        self._shortcuts_dialog = Textarea(label='Shortcuts', enabled=False, value=shortcuts_txt)

        self._tracks_layer.changed.connect(lambda v: setattr(self._load_button, 'enabled', v is not None))
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

    def _non_visible_spatial_axis(self) -> int:
        return self._viewer.dims.order[-3]

    def _is_2d(self) -> bool:
        return self._viewer.dims.ndisplay == 2
    
    def _to_display(self, track: ArrayLike) -> ArrayLike:
        track = np.atleast_2d(track)
        track = track[:, np.asarray(self._viewer.dims.order[-3:]) - 1]  # reordering according to napari
        return track[:, ::-1]  # reordering to vispy
    
    @staticmethod
    def _get_node_predecessors(node: Node, max_depth: int = 500) -> List[Node]:
        current_time = node.time
        predecessors = []
        queue = [node]

        while len(queue) > 0:
            node = queue.pop()
            predecessors.append(node)

            for parent in node.parents:
                if current_time - parent.time < max_depth:
                    queue.append(parent)

        return predecessors

    def _on_hover(self, layer: Tracks, event) -> None:
        if not self._inspecting or not self._is_2d():
            return

        coord = layer.world_to_data(np.asarray(event.position))
        index = self._get_index(layer, coord)
        if index is None:
            self._make_empty(self._hover_visual)
            return
        
        node = layer._manager._get_node(index)
        preds = self._get_node_predecessors(node, self._node_t_trail_length)
        track = np.stack([n.vertex for n in preds])

        relative_t = ((track[:, 0] - coord[0]) + self._node_t_trail_length) / self._node_t_trail_length
        selected = np.logical_and(relative_t > 0, relative_t <= 1.0)

        s_idx = self._non_visible_spatial_axis()
        relative_s = (self._node_spatial_trail_length - np.abs(track[:, s_idx] - coord[s_idx])) / self._node_spatial_trail_length

        track = track[selected][:, 1:]  # picking z, y, x
        size = np.clip(self._node_radius.value * relative_s[selected], 1, None)

        edge_color = np.empty((len(size), 4), dtype=np.float32)
        edge_color[...] = _handle_str('white')[np.newaxis, ...]
        edge_color[:, -1] = relative_t[selected]

        self._hover_visual.set_data(
            self._to_display(track),
            size=size,
            edge_width=self._node_edge_width,
            face_color='transparent',
            edge_color=edge_color,
        )
        self._node.update()
    
    def _on_double_click(self, layer: Tracks, event) -> None:
        if not self._is_2d():
            return

        coord = layer.world_to_data(np.asarray(event.position))
        if self._is_adding:
            index = layer._manager.add(coord, {'verified': True})
            self._pending_additions.append(index)
            if self._eval_node is not None:
                if self._eval_node.vertex[0] < coord[0]:
                    parent, child = self._eval_node.index, index
                else:
                    parent, child = index, self._eval_node.index
                self._manager.link(parent, child)
            self._eval_node = self._get_node(index)
            self._increment_time(1 - 2 * self._backward_incr.value)
            self._load_eval_visual()
            return

        index = self._get_index(layer, coord)
        if index is None:
            return
 
        if self._inspecting:
            self._select_node(index)
    
    @staticmethod
    def _is_modified(event, modifier: str) -> bool:
        return len(event.modifiers) == 1 and event.modifiers[0] == modifier
    
    def _on_drag(self, layer: Tracks, event) -> None:
        if not self._is_2d():
            return
        
        if self._eval_node is not None and self._is_modified(event, 'Shift'):
            coord = layer.world_to_data(np.asarray(event.position))

            # FIXME: THIS WORKS, napari's transforms needs to be fixed
            # if np.linalg.norm(coord[1:] - self._eval_node.coord[1:]) > 5:
            #     # forcing drag starts from selected data
            #     return
            
            displayed = np.asarray(self._viewer.dims.displayed, dtype=int)
            self._is_moving = True
            yield

            while event.type == 'mouse_move':
                coord = np.asarray(layer.world_to_data(np.asarray(event.position)))
                self._eval_node.vertex[displayed] = coord[displayed]
                self._manager._is_serialized = False
                self._manager._view_time_slice = (0, -1)  # invalidate slicing
                self._load_eval_visual()
                yield
            
            self._is_moving = False

    def _select_node(self, index: int) -> None:
        self._inspecting = False
        self._make_empty(self._hover_visual)

        self._eval_node = self._get_node(index)

        self._load_eval_visual()

    def _load_eval_visual(self) -> bool:
        node = self._eval_node
        time = self._viewer.dims.point[0]
        if node is None or abs(node.time - time) > 1:
            self._make_empty(self._eval_visual)
            return False

        elif node.features.get('verified', False):
            edge_color = 'green'
        else:
            edge_color = 'white'

        self._center_to(node.vertex)

        size = self._node_radius.value

        self._eval_visual.set_data(
            self._to_display(node.vertex[1:]),
            face_color='transparent',
            edge_color=edge_color,
            size=size,
            edge_width=self._node_edge_width,
        )
        self._node.update()

        next_coord = node.vertex + np.asarray([1, 0, 0, 0])
        self._load_neighborhood(next_coord, self._neighbor_radius.value)
        return True
    
    @property
    def _node_edge_width(self) -> float:
        return self._node_radius.value / 20

    def _center_to(self, coord: ArrayLike) -> None:
        layer = self._tracks_layer.value
        data_to_world = layer._transforms[1:3].simplified
        self._viewer.dims.set_point(range(4), data_to_world(coord))
        self._update_bbox_position()
        
    def _make_empty(self, visual: Markers) -> None:
        visual.set_data(np.empty((0, 3)))
    
    @property
    def current_t(self) -> int:
        if self._tracks_layer.value is None:
            return 0
        step = self._viewer.dims.current_step
        assert len(step) == 4
        return int(self._tracks_layer.value.world_to_data(step)[0])

    def _escape_eval_mode(self) -> None:
        self._reset_eval_tracking()
        self._reset_neighborhood()

    def _reset_eval_tracking(self) -> None:
        self._inspecting = True
        self._eval_node = None
        self._is_adding = False
        self._pending_additions.clear()
        self._make_empty(self._eval_visual)

    def _load_neighborhood(self, coord: ArrayLike, radius: int) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            return

        coord = np.asarray(coord)

        nodes = layer._manager._time_to_nodes.get(int(round(coord[0])), [])

        if len(nodes) == 0:
            self._reset_neighborhood()
            return

        track_vertices = np.stack([node.vertex for node in nodes])
        distances = np.linalg.norm(track_vertices[:, 1:] - coord[np.newaxis, 1:], axis=1)
        
        neighbors_mask = distances < radius
        if neighbors_mask.sum() == 0:
            self._reset_neighborhood()
            return

        distances = distances[neighbors_mask]
        track_vertices = track_vertices[neighbors_mask]

        indices = np.argsort(distances)
        ordered_indices = np.nonzero(neighbors_mask)[0][indices]

        self._neighbor_original_ids = [nodes[i].index for i in ordered_indices]
        self._neighbor_selected_id = 0
        self._neighbor_vertices = track_vertices[indices]

        self._load_neighborhood_visual()
    
    def _load_neighborhood_visual(self) -> None:
        if self._neighbor_show.value == False or self._neighbor_vertices is None:
            self._make_empty(self._neighbor_visual)
            return

        edge_color = np.empty((len(self._neighbor_vertices), 4), dtype=np.float32)
        edge_color[...] = _handle_str('yellow')[np.newaxis, ...]
        edge_color[self._neighbor_selected_id] = _handle_str('magenta')

        self._neighbor_visual.set_data(
            self._to_display(self._neighbor_vertices[:, 1:]),
            face_color='transparent',
            edge_color=edge_color,
            size=self._node_radius.value,
            edge_width=self._node_edge_width,
        )
        self._node.update()

    def _increment_neighbor_selected_index(self, step: int) -> None:
        if self._neighbor_selected_id is None:
            return
        
        self._neighbor_selected_id = (self._neighbor_selected_id + step) % len(self._neighbor_vertices)
        self._load_neighborhood_visual()
 
    def _reset_neighborhood(self) -> None:
        self._neighbor_original_ids = None
        self._neighbor_selected_id = None
        self._neighbor_vertices = None
        self._make_empty(self._neighbor_visual)
    
    def _get_index(self, layer: Tracks, coord: ArrayLike, radius: int = 25) -> Optional[int]:
        coord = np.asarray(coord)

        index = layer.get_value(coord)
        if index is None:
            return

        distance = np.linalg.norm(
            layer._manager._id_to_nodes[index].vertex - coord
        )

        if distance > radius:
            return None
        
        return index
    
    def _get_node(self, index: int) -> Node:
        layer: Tracks = self._tracks_layer.value
        return layer._manager._get_node(index)
    
    @staticmethod
    def _search_time_point(node: Node, time: int, neigh_fun: Callable) -> Node:

        queue = [node]

        min_dist = 1e6
        nearest = node

        while len(queue) > 0:
            node = queue.pop(0)

            dist = abs(node.time - time)
            if dist < 1:
                return node

            elif dist < min_dist:
                nearest = node
            
            for neigh in neigh_fun(node):
                queue.append(neigh)

        return nearest
    
    def _increment_time(self, step: int) -> None:
        self._viewer.dims.set_current_step(0, self._viewer.dims.current_step[0] + step)
        time = self._viewer.dims.point[0]
        if self._eval_node:
            if step > 0:
                self._eval_node = self._search_time_point(self._eval_node, time, lambda x: x.children)
            else:
                self._eval_node = self._search_time_point(self._eval_node, time, lambda x: x.parents)

        self._load_eval_visual()

    def _on_load(self) -> None:
        layer: Tracks = self._tracks_layer.value
        layer.editable = True

        if layer is None:
            self._save_dialog.enabled = False
            return
        
        features = layer.features
        if 'verified' in features.columns:
            self._stats_manager.n_verified = features['verified'].sum()
        else:
            self._stats_manager.n_verified = 0

        self._on_tracks_change()

        aux_visuals = [self._hover_visual, self._eval_visual, self._neighbor_visual]
        visual = self._viewer.window.qt_viewer.layer_to_visual[layer]
        self._node = visual.node
        for visual in aux_visuals:
            try:
                self._node.remove_subvisual(visual)
            except ValueError:
                pass

        # visuals must be removed before attaching spatial filter
        self._node._subvisuals[0].attach(self._vertices_spatial_filter)
        self._node._subvisuals[2].attach(self._graph_spatial_filter)
        for visual in aux_visuals:
            self._node.add_subvisual(visual)

        # FIXME: disabled this to avoid serialization
        # layer.events.rebuild_graph.connect(self._update_stats_n_tracks)
        self._update_stats_n_tracks()

        layer.events.rebuild_graph.connect(self._on_tracks_change)
        layer.events.rebuild_tracks.connect(self._on_tracks_change)
        self._spatial_range.changed.connect(lambda _: layer.refresh())

        layer.mouse_double_click_callbacks.append(self._on_double_click)
        layer.mouse_move_callbacks.append(self._on_hover)
        layer.mouse_drag_callbacks.append(self._on_drag)

        layer.bind_key('N', lambda _: self._increment_time(-1), overwrite=True)
        layer.bind_key('M', lambda _: self._increment_time(+1), overwrite=True)

        layer.bind_key('C', lambda _: self._set_verified_status(True), overwrite=True)
        layer.bind_key('X', lambda _: self._set_verified_status(False), overwrite=True)

        layer.bind_key('K', lambda _: self._increment_neighbor_selected_index(-1), overwrite=True)
        layer.bind_key('L', lambda _: self._increment_neighbor_selected_index(+1), overwrite=True)
        layer.bind_key('A', lambda _: self._add_link_to_neigh(), overwrite=True)
        layer.bind_key('S', lambda _: self._switch_link_to_neigh(), overwrite=True)

        layer.bind_key('F', lambda _: self._focus_to_current_track(), overwrite=True)
        layer.bind_key('D', lambda _: self._delete_eval_node(), overwrite=True)
        layer.bind_key('Escape', lambda _: self._escape_eval_mode(), overwrite=True)

        self._save_dialog.enabled = True
         
    def _on_layer_removed(self, layer: Layer) -> None:
        if layer == self._tracks_layer.value:
            visual = self._viewer.window.qt_viewer.layer_to_visual[layer]
            visual.node.remove_subvisual(self._hover_visual)
            visual.node.remove_subvisual(self._eval_visual)
            visual.node.remove_subvisual(self._neighbor_visual)
            visual.node._subvisuals[0].detach(self._vertices_spatial_filter)
            visual.node._subvisuals[2].detach(self._graph_spatial_filter)
     
    def _on_tracks_change(self) -> None:
        self._set_filter_stack()
        self._update_current_slice()
        
    def _set_filter_stack(self) -> None:
        layer: Tracks =  self._tracks_layer.value
        g_vertices = layer._view_graph_vertices
        self._vertices_spatial_filter.vertex_stack = layer._view_track_vertices[:, self._non_visible_spatial_axis()]
        if g_vertices is not None and len(g_vertices) > 0:
            self._graph_spatial_filter.vertex_stack = g_vertices[:, self._non_visible_spatial_axis()]
    
    def _update_current_slice(self) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            return

        coord = layer.world_to_data(self._viewer.dims.point)
        self._vertices_spatial_filter.current_stack = coord[self._non_visible_spatial_axis()]
        self._graph_spatial_filter.current_stack = coord[self._non_visible_spatial_axis()]
    
    def _set_filtering_status(self, event) -> None:
        if event.value == 3:
            self._vertices_spatial_filter.stack_range = 0
            self._graph_spatial_filter.stack_range = 0
            self._hover_visual.visible = False
            if self._eval_node is None:
                self._eval_visual.visible = False
            self._bounding_box.enabled = True
            if self._neighbor_original_ids is None:
                self._neighbor_visual.visible = False

            if self._node._subvisuals[2].bounds(0) is None:
                self._node._subvisuals[2].visible = False
        else:
            self._vertices_spatial_filter.stack_range = self._spatial_range.value
            self._graph_spatial_filter.stack_range = self._spatial_range.value
            self._hover_visual.visible = True
            self._eval_visual.visible = True
            self._neighbor_visual.visible = True
            self._bounding_box.value = False
            self._bounding_box.enabled = False
            self._node._subvisuals[2].visible = True
         
        self._load_eval_visual()
        self._load_neighborhood_visual()
   
    def _delete_eval_node(self) -> None:
        """Delete node and splits the path"""
        layer: Tracks = self._tracks_layer.value
        if layer is None or self._eval_node is None:
            return
        
        node = self._eval_node
        if node.features.get('verified', False):
            warnings.warn('Deleting `verified` node is not allowed.')
            return
        
        layer._manager.remove(node.index)

        self._stats_manager.n_deleted += 1
        self._escape_eval_mode()

    def _on_bounding_box_change(self, status: bool):
        for layer in self._viewer.layers:
            for plane in layer.experimental_clipping_planes:
                plane.enabled = status
            layer.refresh()
    
    def _update_bbox_position(self) -> None:
        node = self._eval_node
        if node is None:
            return
        
        coord = node.vertex
        for layer in self._viewer.layers:
            for plane in layer.experimental_clipping_planes:
                position = np.flip(coord[1:]) - np.asarray(plane.normal) * self._bbox_size.value / 2
                plane.position = position
            layer.refresh()
        
    def _setup_clipping_planes(self) -> None:
        default_params = {
            'position': np.zeros(3),
            'normal': np.zeros(3),
            'enabled': False,
        }

        bbox_planes = []
        for axis in range(3):
            for direction in (-1, 1):
                params = default_params.copy()
                params['normal'] = params['normal'].copy()
                params['normal'][axis] = direction
                bbox_planes.append(ClippingPlane(**params))
        
        for layer in self._viewer.layers:
            layer.experimental_clipping_planes = bbox_planes

    def _set_verified_status(self, status: bool) -> None:
        node = self._eval_node
        if node is None:
            return

        node.features['verified'] = status
        if status:
            self._stats_manager.n_verified += 1
            self._increment_time(1 - 2 * self._backward_incr.value)
        else:
            self._stats_manager.n_verified -= 1

    def _save_verified(self, path: Path) -> None:
        if not path or not str(path).endswith('.csv'):
            return
        
        layer: Tracks = self._tracks_layer.value

        features = layer.features
        verified = features['verified']

        features = features.drop('verified', axis='columns', inplace=False)
        data = layer.data[verified]

        df = tracks_to_dataframe(data, features, layer.graph)
        df.to_csv(path, index=False, header=True)
    
    def _focus_to_current_track(self) -> None:
        if self._eval_node is None:
            return
        
        time = self._viewer.dims.point[0]
        if time < self._eval_node.time:
            self._eval_node = self._search_time_point(self._eval_node, time, lambda x: x.parents)
        else:
            self._eval_node = self._search_time_point(self._eval_node, time, lambda x: x.children)

        self._center_to(self._eval_node.vertex)
        self._load_eval_visual()
    
    def _confirm_addition(self) -> None:
        self._stats_manager.n_add += len(self._pending_additions)
        self._reset_eval_tracking()

    def _cancel_addition(self) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            return

        for index in self._pending_additions:
            layer._manager.remove(index)

        self._reset_eval_tracking()

    def _validate_link_to_neigh(self) -> bool:
        return self._neighbor_show.value and len(self._neighbor_original_ids) > 0
    
    @property
    def _manager(self) -> InteractiveTrackManager:
        layer: Tracks = self._tracks_layer.value
        return layer._manager

    def _add_link_to_neigh(self):
        if not self._validate_link_to_neigh():
            return
       
        neigh_id = self._neighbor_original_ids[self._neighbor_selected_id]
        self._manager.link(neigh_id, self._eval_node.index)

    def _switch_link_to_neigh(self):
        if not self._validate_link_to_neigh():
            return
        
        neigh_id = self._neighbor_original_ids[self._neighbor_selected_id]

        for child in self._eval_node.children:
            child.parents.remove(self._eval_node)

        self._eval_node.children.clear()
        self._manager.link(neigh_id, self._eval_node.index)
    
    def _update_stats(self, event=None) -> None:
        self._stats.value = str(self._stats_manager)
        # TODO implement auto backup
    
    def _update_stats_n_tracks(self, event=None) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            return
        
        self._stats_manager.n_tracks = len(layer._manager)
        self._stats_manager.n_nodes = len(layer.data)
