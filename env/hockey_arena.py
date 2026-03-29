"""HockeyArena: ice rink MJCF geometry using dm_control Composer."""

from dm_control import composer
from dm_control import mjcf


class HockeyArena(composer.Arena):
    """Ice rink with boards, goals, and physics options.

    Geometry:
    - Ice ground plane (visual + friction surface, no collision filtering)
    - 4 board walls as infinite half-space plane geoms (contype=4)
    - 2 goal detection sites (AABB zones, not collision geoms)

    Physics options:
    - timestep=0.005s, integrator=implicitfast, cone=elliptic
    """

    def _build(self, rink_length=30.0, rink_width=15.0, name="hockey_arena"):
        super()._build(name=name)
        self._rink_length = rink_length
        self._rink_width = rink_width

        # Global physics options
        self._mjcf_root.option.timestep = 0.005
        self._mjcf_root.option.integrator = "implicitfast"
        self._mjcf_root.option.cone = "elliptic"
        self._mjcf_root.option.gravity = [0, 0, -9.81]

        # Default contact parameters (applied to all geoms without overrides)
        self._mjcf_root.default.geom.solref = [0.02, 0.4]
        self._mjcf_root.default.geom.solimp = [0.9, 0.95, 0.001]

        # Ice ground plane — visual + friction surface
        # contype=0, conaffinity=0: no explicit collision filtering; MuJoCo ground
        # contact model applies ice friction via the plane geom automatically.
        self._mjcf_root.worldbody.add(
            'geom', name='ice', type='plane',
            size=[rink_length / 2, rink_width / 2, 0.1],
            friction=[0.05, 0.005, 0.0001],
            rgba=[0.8, 0.9, 1.0, 1.0],
            contype=0, conaffinity=0)

        # Board walls as infinite half-space plane geoms.
        # contype=4 (bit 2) so they collide with puck (conaffinity must include bit 2).
        #
        # xyaxes convention: first 3 floats = local x-axis direction,
        # next 3 floats = local y-axis direction. The plane normal is x cross y.

        # Right wall (+x boundary): normal points in -x direction
        self._mjcf_root.worldbody.add(
            'geom', name='board_right', type='plane',
            pos=[rink_length / 2, 0, 0],
            xyaxes=[0, 1, 0, 0, 0, 1],
            size=[1e-7, 1e-7, 1e-7],
            friction=[0.7, 0.005, 0.0001],
            rgba=[0.3, 0.3, 0.4, 1.0],
            contype=4, conaffinity=4)

        # Left wall (-x boundary): normal points in +x direction
        self._mjcf_root.worldbody.add(
            'geom', name='board_left', type='plane',
            pos=[-rink_length / 2, 0, 0],
            xyaxes=[0, -1, 0, 0, 0, 1],
            size=[1e-7, 1e-7, 1e-7],
            friction=[0.7, 0.005, 0.0001],
            rgba=[0.3, 0.3, 0.4, 1.0],
            contype=4, conaffinity=4)

        # Top wall (+y boundary): normal points in -y direction
        self._mjcf_root.worldbody.add(
            'geom', name='board_top', type='plane',
            pos=[0, rink_width / 2, 0],
            xyaxes=[-1, 0, 0, 0, 0, 1],
            size=[1e-7, 1e-7, 1e-7],
            friction=[0.7, 0.005, 0.0001],
            rgba=[0.3, 0.3, 0.4, 1.0],
            contype=4, conaffinity=4)

        # Bottom wall (-y boundary): normal points in +y direction
        self._mjcf_root.worldbody.add(
            'geom', name='board_bottom', type='plane',
            pos=[0, -rink_width / 2, 0],
            xyaxes=[1, 0, 0, 0, 0, 1],
            size=[1e-7, 1e-7, 1e-7],
            friction=[0.7, 0.005, 0.0001],
            rgba=[0.3, 0.3, 0.4, 1.0],
            contype=4, conaffinity=4)

        # Goal detection sites (AABB zones, not collision geoms)
        self._home_goal = self._mjcf_root.worldbody.add(
            'site', name='home_goal', type='box',
            size=[0.5, 2.0, 1.0],
            pos=[-rink_length / 2 + 0.5, 0, 0.5],
            rgba=[1, 0, 0, 0.3])

        self._away_goal = self._mjcf_root.worldbody.add(
            'site', name='away_goal', type='box',
            size=[0.5, 2.0, 1.0],
            pos=[rink_length / 2 - 0.5, 0, 0.5],
            rgba=[0, 0, 1, 0.3])

    @property
    def rink_length(self):
        return self._rink_length

    @property
    def rink_width(self):
        return self._rink_width

    @property
    def home_goal(self):
        return self._home_goal

    @property
    def away_goal(self):
        return self._away_goal
