
from typing import Container, Optional

import numpy as np
from numpy.typing import ArrayLike
import napari

from napari.layers import Tracks, Layer
from napari.utils.colormaps.standardize_color import _handle_str

from magicgui.widgets import Container, create_widget, SpinBox, PushButton

from vispy.gloo import VertexBuffer
from vispy.scene.visuals import Markers
from vispy.visuals.filters import Filter


class SpatialFilter(Filter):

    VERT_SHADER = """
        // varying vec4 v_spatial_color;
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
        // varying vec4 v_spatial_color;
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

        self._tracks_layer = create_widget(annotation=Tracks, name='Tracks')
        self.append(self._tracks_layer)

        self._hover_visual = Markers(scaling=True)
        self._eval_visual = Markers(scaling=True)
        self._spatial_filter = SpatialFilter(10.0)

        self._node_radius = 10
        self._node_t_trail_length = 10
        self._node_spatial_trail_length = 10

        self._inspecting = True

        self._load_button = PushButton(text='Load', enabled=False)
        self._load_button.changed.connect(self._on_load)
        self.append(self._load_button)

        self._node_radius = SpinBox(max=100, value=10, label='Radius')
        self.append(self._node_radius)

        self._spatial_range = SpinBox(max=500, value=self._spatial_filter.stack_range,
                                      label='Spatial Range')
        self._spatial_range.changed.connect(lambda e: setattr(self._spatial_filter, 'stack_range', e.value))
        self.append(self._spatial_range)

        self._tracks_layer.changed.connect(lambda v: setattr(self._load_button, 'enabled', v is not None))
        self._viewer.layers.events.removed.connect(self._on_layer_removed)
    
    def _non_visible_spatial_axis(self) -> int:
        return self._viewer.dims.order[-3]

    def _is_2d(self) -> bool:
        return self._viewer.dims.ndisplay == 2
    
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
            track[:, ::-1],
            size=size,
            edge_width=self._node_edge_width,
            face_color='transparent',
            edge_color=edge_color,
        )
        self._node.update()
    
    def _on_drag(self, layer: Tracks, event) -> None:
        if not self._is_2d():
            return

        coord = layer.world_to_data(np.array(event.position))
        index = self._get_index(layer, coord)
        if index is None:
           return
        
        self._inspecting = False
        self._make_empty(self._hover_visual)

        track_ids = layer._manager.track_ids
        self._eval_track_id = track_ids[index]
        self._eval_track = layer._manager.track_vertices[track_ids == self._eval_track_id]
        self._eval_relative_id = np.where(self._eval_track[:, 0] == int(coord[0]))[0][0]
        self._set_eval_markers()

    def _set_eval_markers(self, t_radius: int = 0) -> None:
        # TODO add color to indicate forward and backward nodes
        index = self._eval_relative_id
        self._center_to(self._eval_track[index])

        if t_radius == 0:
            track = self._eval_track[index, 1:]
            size = self._node_radius.value
        else:
            slicing = slice(max(0, index - t_radius), min(index + t_radius + 1, len(self._eval_track)))
            track = self._eval_track[slicing, 1:]
            scale = 1 - np.abs(np.arange(slicing.start, slicing.stop) - index) / t_radius
            size = (self._node_radius.value - 1) * scale + 1

        self._eval_visual.set_data(
            np.atleast_2d(track)[:, ::-1],
            face_color='transparent',
            edge_color='white',
            size=size,
            edge_width=self._node_edge_width,
        )
        self._node.update()
    
    @property
    def _node_edge_width(self) -> float:
        return self._node_radius.value / 20

    def _center_to(self, coord: ArrayLike) -> None:
        layer = self._tracks_layer.value
        data_to_world = layer._transforms[1:3].simplified
        self._viewer.dims.set_current_step(range(4), data_to_world(coord))
        
    def _make_empty(self, visual: Markers) -> None:
        visual.set_data(np.empty((0, 3)))
      
    def _escape_eval_mode(self) -> None:
        self._make_empty(self._eval_visual)
        self._inspecting = True
        self._eval_relative_id = None
    
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
    
    def _increment_eval_relative_index(self, step: int) -> None:
        if self._eval_relative_id is None:
            return

        self._eval_relative_id += step
        self._eval_relative_id = min(max(self._eval_relative_id, 0), len(self._eval_track) - 1)
        self._set_eval_markers()
    
    def _on_load(self) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            return
        
        self._on_tracks_change()

        visual = self._viewer.window.qt_viewer.layer_to_visual[layer]
        self._node = visual.node
        for visual in (self._hover_visual, self._eval_visual):
            try:
                self._node.remove_subvisual(visual)
            except ValueError:
                pass

        self._node.attach(self._spatial_filter)
        self._node.add_subvisual(self._hover_visual)
        self._node.add_subvisual(self._eval_visual)

        layer.events.rebuild_tracks.connect(self._on_tracks_change)

        layer.mouse_move_callbacks.append(self._on_hover)
        layer.mouse_drag_callbacks.append(self._on_drag)

        layer.bind_key('N', lambda _: self._increment_eval_relative_index(-1), overwrite=True)
        layer.bind_key('M', lambda _: self._increment_eval_relative_index(+1), overwrite=True)
        layer.bind_key('Escape', lambda _: self._escape_eval_mode(), overwrite=True)
         
    def _on_layer_removed(self, layer: Layer) -> None:
        if layer == self._tracks_layer.value:
            visual = self._viewer.window.qt_viewer.layer_to_visual[layer]
            visual.node.remove_subvisual(self._hover_visual)
            visual.node.remove_subvisual(self._eval_visual)
            visual.detach(self._spatial_filter)
    
    def _on_tracks_change(self) -> None:
        self._set_filter_stack()
        self._update_current_slice()
        
    def _set_filter_stack(self) -> None:
        vertices = self._tracks_layer.value._manager.track_vertices
        self._spatial_filter.vertex_stack = vertices[:, self._non_visible_spatial_axis()]
    
    def _update_current_slice(self) -> None:
        layer: Tracks = self._tracks_layer.value
        if layer is None:
            return

        coord = layer.world_to_data(self._viewer.dims.current_step)
        self._spatial_filter.current_stack = coord[self._non_visible_spatial_axis()]
    
    def _set_filtering_status(self, event) -> None:
        if event.value == 3:
            self._spatial_filter.stack_range = 0
            self._hover_visual.visible = False
        else:
            self._spatial_filter.stack_range = self._spatial_range.value
            self._hover_visual.visible = True
   