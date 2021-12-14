
from typing import Container, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
import napari

from napari.layers import Tracks, Layer
from napari.layers.utils.plane import ClippingPlane
from napari.utils.colormaps.standardize_color import _handle_str

from magicgui.widgets import (
    Container, create_widget, SpinBox, Slider, PushButton, CheckBox,
    FileEdit,
)

from vispy.gloo import VertexBuffer
from vispy.scene.visuals import Markers
from vispy.visuals.filters import Filter

import warnings

from pathlib import Path

from ._writer import tracks_to_dataframe


@dataclass
class Node:
    coord: ArrayLike
    verified: bool = False
    original_id: Optional[int] = None


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
        self._vertex_stack = np.array(value).astype(np.float32)
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

        self._tracks_layer.changed.connect(lambda v: setattr(self._load_button, 'enabled', v is not None))
        self._viewer.layers.events.removed.connect(self._on_layer_removed)
    
    def _non_visible_spatial_axis(self) -> int:
        return self._viewer.dims.order[-3] - 1

    def _is_2d(self) -> bool:
        return self._viewer.dims.ndisplay == 2
    
    def _to_display(self, track: ArrayLike) -> ArrayLike:
        track = np.atleast_2d(track)
        track = track[:, np.array(self._viewer.dims.order[-3:]) - 1]  # reordering according to napari
        return track[:, ::-1]  # reordering to vispy

    def _on_hover(self, layer: Tracks, event) -> None:
        if not self._inspecting or not self._is_2d():
            return

        coord = layer.world_to_data(np.array(event.position))
        index = self._get_index(layer, coord)
        if index is None:
            self._make_empty(self._hover_visual)
            return

        track_ids = layer._manager.track_ids
        track = layer._manager.track_vertices[track_ids == track_ids[index]]

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

        coord = layer.world_to_data(np.array(event.position))
        if self._is_adding:
            self._add_node(coord)
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
        
        if self._eval_track_id is not None and self._is_modified(event, 'Shift'):
            coord = layer.world_to_data(np.array(event.position))

            # FIXME: THIS WORKS, napari's transforms needs to be fixed
            # if np.linalg.norm(coord[1:] - self._eval_node.coord[1:]) > 5:
            #     # forcing drag starts from selected data
            #     return
            
            displayed = np.array(self._viewer.dims.displayed, dtype=int)
            self._is_moving = True
            yield

            index = self._eval_node.original_id
            while event.type == 'mouse_move':
                coord = np.array(layer.world_to_data(np.array(event.position)))
                self._eval_nodes[int(coord[0])].coord = coord
                if index is not None:
                    self._tracks_layer.value.data[index, displayed + 1] = coord[displayed]

                self._load_eval_visual()
                yield
            
            layer._manager.build_tracks()
            self._is_moving = False

    def _select_node(self, index: int) -> None:
        self._inspecting = False
        self._make_empty(self._hover_visual)

        layer = self._tracks_layer.value
        track_ids = layer._manager.track_ids
        self._eval_track_id = track_ids[index]

        selected, = np.nonzero(track_ids == self._eval_track_id)
        vertices = self.track_vertices[selected]
        verified = self._tracks_layer.value.properties['verified'][selected]

        self._eval_nodes = {
            int(c[0]): Node(coord=c, verified=v, original_id=i)
            for i, v, c in zip(selected, verified, vertices)
        }

        self._load_eval_visual()
    
    @property
    def track_vertices(self) -> ArrayLike:
        return self._tracks_layer.value._manager.track_vertices

    @property
    def track_ids(self) -> ArrayLike:
        return self._tracks_layer.value._manager.track_ids

    def _load_eval_visual(self) -> bool:
        node = self._eval_node
        if node is None:
            self._make_empty(self._eval_visual)
            return False

        if node.original_id is None:
            edge_color = 'cyan'
        elif node.verified:
            edge_color = 'green'
        else:
            edge_color = 'white'

        self._center_to(node.coord)

        size = self._node_radius.value

        self._eval_visual.set_data(
            self._to_display(node.coord[1:]),
            face_color='transparent',
            edge_color=edge_color,
            size=size,
            edge_width=self._node_edge_width,
        )
        self._node.update()

        next_coord = node.coord + np.array([1, 0, 0, 0])
        self._load_neighborhood(next_coord, self._neighbor_radius.value)
        return True
    
    @property
    def _node_edge_width(self) -> float:
        return self._node_radius.value / 20

    def _center_to(self, coord: ArrayLike) -> None:
        layer = self._tracks_layer.value
        data_to_world = layer._transforms[1:3].simplified
        self._viewer.dims.set_current_step(range(4), data_to_world(coord))
        self._update_bbox_position()
        
    def _make_empty(self, visual: Markers) -> None:
        visual.set_data(np.empty((0, 3)))
    
    def _is_valid_index(self, index: int) -> bool:
        return (0 <= index < len(self.track_ids) and
            self.track_ids[index] == self._eval_track_id)
    
    @property
    def current_t(self) -> int:
        if self._tracks_layer.value is None:
            return 0
        step = self._viewer.dims.current_step
        assert len(step) == 4
        return int(self._tracks_layer.value.world_to_data(step)[0])

    @property
    def _eval_node(self) -> Optional[Node]:
        return self._eval_nodes.get(self.current_t)
      
    def _escape_eval_mode(self) -> None:
        self._reset_eval_tracking()
        self._reset_neighborhood()

    def _reset_eval_tracking(self) -> None:
        self._inspecting = True
        self._eval_track_id = None
        self._eval_nodes = {}
        self._is_adding = False
        self._make_empty(self._eval_visual)

    def _load_neighborhood(self, coord: ArrayLike, radius: int) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            return

        coord = np.array(coord)

        selected, = np.nonzero(np.abs(layer._manager.track_times - coord[0]) < 1)

        if len(selected) == 0:
            self._reset_neighborhood()
            return

        track_vertices = layer._manager.track_vertices[selected]
        distances = np.linalg.norm(track_vertices[:, 1:] - coord[np.newaxis, 1:], axis=1)
        
        neighbors_mask = distances < radius
        distances = distances[neighbors_mask]
        selected = selected[neighbors_mask]
        if len(selected) == 0:
            self._reset_neighborhood()
            return

        indices = np.argsort(distances)
        selected = selected[indices]

        self._neighbor_original_ids = selected
        self._neighbor_selected_id = 0
        self._neighbor_vertices = layer._manager.track_vertices[selected]

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
    
    def _get_index(self, layer: Tracks, coord: ArrayLike, radius: int = 5) -> Optional[int]:
        coord = np.array(coord)

        track_vertices = layer._manager.track_vertices
        non_vis_axis = self._non_visible_spatial_axis()

        selected, = np.nonzero(np.logical_and(
            np.abs(layer._manager.track_times - coord[0]) < 1,
            np.abs(track_vertices[:, non_vis_axis] - coord[non_vis_axis]) < self._spatial_range.value
        ))

        if len(selected) == 0:
            return None

        visible_axis = np.array(self._viewer.dims.displayed, dtype=int)
        
        closest = np.argmin(
            np.abs(
                track_vertices[selected][:, visible_axis] - coord[visible_axis]
            ).sum(axis=1)
        )
        index = selected[closest]
        distance = np.linalg.norm(
            layer._manager.track_vertices[index, visible_axis] - coord[visible_axis]
        )
        if distance > radius:
            return None
        
        return index
    
    def _increment_time(self, step: int) -> None:
        self._viewer.dims.set_current_step(0, self._viewer.dims.current_step[0] + step)
        self._load_eval_visual()

    def _on_load(self) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            self._save_dialog.enabled = False
            return
        
        props = layer.properties
        if 'verified' not in props:
            props['verified'] = np.zeros(len(layer.data), dtype=bool)
        layer.properties = props
        
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
        vertices = self._tracks_layer.value._manager.track_vertices
        g_vertices = self._tracks_layer.value._manager.graph_vertices
        self._vertices_spatial_filter.vertex_stack = vertices[:, self._non_visible_spatial_axis()]
        if g_vertices is not None:
            self._graph_spatial_filter.vertex_stack = g_vertices[:, self._non_visible_spatial_axis()]
    
    def _update_current_slice(self) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            return

        coord = layer.world_to_data(self._viewer.dims.current_step)
        self._vertices_spatial_filter.current_stack = coord[self._non_visible_spatial_axis()]
        self._graph_spatial_filter.current_stack = coord[self._non_visible_spatial_axis()]
    
    def _set_filtering_status(self, event) -> None:
        if event.value == 3:
            self._vertices_spatial_filter.stack_range = 0
            self._graph_spatial_filter.stack_range = 0
            self._hover_visual.visible = False
            if self._eval_track_id is None:
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
        if layer is None or self._eval_track_id is None:
            return
        
        node = self._eval_node
        if node.verified:
            warnings.warn('Deleting `verified` node is not allowed.')
            return
        
        if node.original_id is None:
            return self._eval_nodes.pop(int(node.coord[0]))
        
        graph = layer.graph
        data = layer.data

        first_half = data[:node.original_id]
        last_half = data[node.original_id + 1:]
        last_mask = last_half[:, 0] == self._eval_track_id

        if last_mask.sum() > 0:
            new_track_id = layer._manager.track_ids.max() + 1
            last_half[last_mask, 0] = new_track_id  # splitting rest of the track
            self._update_graph_parent(graph, self._eval_track_id, new_track_id)
        else:
            graph = {k: v for k, v in graph.items() if v[0] != self._eval_track_id}

        data = np.concatenate((first_half, last_half), axis=0)

        props = layer.properties
        layer.data = data
        for k, v in props.items():
            props[k] = np.delete(v, node.original_id, axis=0)
        layer.properties = props
        layer.graph = graph

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
        
        coord = node.coord
        for layer in self._viewer.layers:
            for plane in layer.experimental_clipping_planes:
                position = np.flip(coord[1:]) - np.array(plane.normal) * self._bbox_size.value / 2
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

    def _update_node_props(self, index: int, new_props: Dict) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            raise RuntimeError('Tried to update properties when the layer was not available')

        props = layer.properties
        for k, v in new_props.items():
            props[k][index] = v
        
        # hack: to avoid mixing up the tracks ordering
        layer._manager._order = np.arange(len(layer.data))
        layer.properties = props
    
    def _add_nodes_to_tracks(self, new_tracks: ArrayLike, new_props: Dict) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            raise RuntimeError('Tried to append nodes to an empty layer')

        graph = layer.graph
        props = layer.properties
        data = layer.data
        layer.data = np.append(data, new_tracks, axis=0)

        merged_props = {}
        for k, v in props.items():
            merged_props[k] = np.append(v, new_props[k], axis=0)

        layer.properties = merged_props
        layer.graph = graph

    def _get_property(self, key: str, indices: Union[ArrayLike, slice, int]) -> Any:
        layer: Tracks = self._tracks_layer.value
        return layer.properties[key][indices]

    def _set_property(self, key: str, indices: Union[ArrayLike, slice, int], value: Any) -> None:
        layer: Tracks = self._tracks_layer.value
        layer.properties[key][indices] = value
     
    def _set_verified_status(self, status: bool) -> None:
        node = self._eval_node
        node.verified = status
        if node.original_id is not None:
            self._set_property('verified', node.original_id, status)

        if status:
            self._increment_time(1 - 2 * self._backward_incr.value)

    def _save_verified(self, path: Path) -> None:
        if not path or not str(path).endswith('.csv'):
            return
        
        layer: Tracks = self._tracks_layer.value

        props = layer.properties
        verified = props.pop('verified')

        for k, v in props.items():
            props[k] = v[verified]

        data = layer.data[verified]

        df = tracks_to_dataframe(data, props, layer.graph)
        df.to_csv(path, index=False, header=True)
    
    def _focus_to_current_track(self) -> None:
        if self._eval_track_id is None:
            return

        vertices = np.array([n.coord for n in self._eval_nodes.values()])
        argmin = np.argmin(np.abs(vertices[:, 0] - self.current_t))

        self._center_to(vertices[argmin])
        self._load_eval_visual()
    
    def _new_track_id(self) -> int:
        return self._tracks_layer.value._manager.track_ids.max() + 1
 
    def _add_node(self, coord: ArrayLike) -> ArrayLike:
        coord = np.array(coord)
        if self._eval_track_id is None:
            self._eval_track_id = self._new_track_id()

        if int(coord[0]) in self._eval_nodes:
            warnings.warn('Adding multiple nodes to a unique time point is not allowed')
            return
        
        self._eval_nodes[int(coord[0])] = Node(coord, verified=False)
        self._increment_time(1 - 2 * self._backward_incr.value)
    
    def _eval_to_new_track(self) -> Tuple[ArrayLike, Dict]:

        vertices = np.array([n.coord for n in self._eval_nodes.values()])
        vertices = vertices[np.argsort(vertices[:, 0])]
        size = len(vertices)
        tracks = np.insert(vertices, 0, np.full(size, self._eval_track_id), axis=1)

        keys = self._tracks_layer.value.properties.keys()
        props = {k: np.zeros(size) for k in keys}
        props['verified'] = np.ones(size, dtype=bool)

        return tracks, props

    def _confirm_addition(self) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None or self._eval_track_id is None:
            return

        # updated new track
        new_track, new_props = self._eval_to_new_track()

        # removing old track and added new track
        props = layer.properties
        data = layer.data
        selected = layer._manager.track_ids != self._eval_track_id

        for k, v in props.items():
            props[k] = np.append(v[selected], new_props[k], axis=0)
        
        graph = layer.graph

        layer.data = np.append(data[selected], new_track, axis=0)
        layer.properties = props
        layer.graph = graph

        self._reset_eval_tracking()

    def _cancel_addition(self) -> None:
        self._reset_eval_tracking()

    def _has_parent(self, index: int) -> bool:
        track_id = self.track_ids[index]
        if index > 0 and track_id == self.track_ids[index - 1]:
            return True

        layer: Tracks = self._tracks_layer.value
        return track_id in layer.graph

    def _has_child(self, index: int) -> bool:
        track_id = self.track_ids[index]
        if index < (len(self.track_ids) - 1) and track_id == self.track_ids[index + 1]:
            return True

        graph = self._tracks_layer.value.graph
        return any(parent[0] == track_id for parent in graph.values())
      
    def _link_nodes(self, current: int, neighbor: int) -> None:
        if self._is_adding:
            warnings.warn('Cannot link nodes during `add` mode.')
            return
        
        track_ids = self.track_ids
        cur_track_id = track_ids[current]
        neigh_track_id = track_ids[neighbor]
        if cur_track_id == neigh_track_id:
            return
        
        if self._has_parent(neighbor):
            warnings.warn('Neighbor is already connected to a node.')
            return
        
        layer: Tracks = self._tracks_layer.value
        graph = layer.graph
        data = layer.data

        if self._has_child(current):
            # updating subsequent vertices of track if necessary
            selected, = np.nonzero(data[:, 0] == cur_track_id and data[:, 1] > data[current, 1])
            # this will not ocurr when the childs are mapped in the graph and not the tracks data
            if len(selected) > 0:
                new_track_id = self._new_track_id()

                # updating old track id parents
                self._update_graph_parent(graph, cur_track_id, new_track_id)

                data[selected, 0] = new_track_id
                graph[new_track_id] = [cur_track_id]

            # setting new split
            graph[neigh_track_id] = [cur_track_id]

        else:
            self._update_graph_parent(graph, neigh_track_id, cur_track_id)
            data[data[:, 0] == neigh_track_id, 0] = cur_track_id

        props = layer.properties
        layer.data = data
        layer.graph = graph
        layer.properties = props

    @staticmethod
    def _update_graph_parent(graph: Dict, old_parent: int, new_parent: int) -> None:
        for node in list(graph.keys()):
            if graph[node][0] == old_parent:
                graph[node] = [new_parent]
