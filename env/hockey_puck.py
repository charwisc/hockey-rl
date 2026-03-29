"""HockeyPuck: puck entity with cylinder geom and constrained slide+hinge joints."""

from dm_control import composer
from dm_control import mjcf


class HockeyPuck(composer.Entity):
    """Hockey puck as a dm_control Composer Entity.

    Geometry: cylinder, radius=0.05m, half-height=0.01m, mass=0.170kg
    Motion: constrained to 2D plane via slide joints (x, y) + hinge joint (z rotation)
    NOT a freejoint — prevents 3D flipping on puck-board contact.

    contype=5 (bits 0+2):
    - Bit 2: collides with board walls (contype=4, conaffinity=4)
    - Bit 0: will collide with stick geoms (contype=1) added in Plan 03
    """

    def _build(self, name="puck"):
        self._mjcf_root = mjcf.RootElement(model=name)

        body = self._mjcf_root.worldbody.add('body', name='puck')

        # Cylinder geom: radius=0.05m (5cm), half-height=0.01m (1cm), mass=0.170kg
        body.add('geom', name='puck_geom', type='cylinder',
                 size=[0.05, 0.01],
                 mass=0.170,
                 friction=[0.05, 0.005, 0.0001],
                 solref=[0.02, 0.4],
                 solimp=[0.9, 0.95, 0.001],
                 condim=3,
                 contype=5,       # bits 0+2: boards (bit 2) + sticks (bit 0)
                 conaffinity=5,
                 rgba=[0.1, 0.1, 0.1, 1.0])

        # Slide joints for x and y translation (constrained to 2D plane)
        # Damping simulates ice friction drag on the puck
        body.add('joint', name='puck_x', type='slide', axis=[1, 0, 0],
                 damping=0.1)
        body.add('joint', name='puck_y', type='slide', axis=[0, 1, 0],
                 damping=0.1)

        # Hinge joint for z-rotation (puck can spin flat on ice)
        body.add('joint', name='puck_rot', type='hinge', axis=[0, 0, 1],
                 damping=0.05)

    @property
    def mjcf_model(self):
        return self._mjcf_root
