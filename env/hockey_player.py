"""HockeyPlayer: capsule agent entity with stick hitbox and velocity actuators."""

from dm_control import composer
from dm_control import mjcf


class HockeyPlayer(composer.Entity):
    """Hockey player as a dm_control Composer Entity.

    Geometry:
    - Capsule body: radius=0.3m, half-height=0.4m, mass=75kg (contype=2)
    - Stick hitbox: box extending forward, mass=1kg (contype=1)

    Collision design:
    - Capsule contype=2 (bit 1): agents collide with each other
    - Stick contype=1 (bit 0): stick collides with puck (puck conaffinity=5 has bit 0)
    - capsule.contype & board.conaffinity = 2 & 4 = 0 -> NO board collision (joint limits enforce boundaries)
    - stick.contype & puck.conaffinity = 1 & 5 = 1 -> stick-puck contact generated
    - puck.contype & board.conaffinity = 5 & 4 = 4 -> puck-board contact generated

    Motion: 3 velocity-controlled actuators (vx, vy, vrot)
    Action mapping: 4-float [move_x, move_y, speed, stick_angle] -> 3 actuators is done
    in HockeyTask (not here).
    """

    def _build(self, team: int, player_idx: int, name=None):
        self._team = team
        self._player_idx = player_idx
        model_name = name or f"player_{team}_{player_idx}"
        self._mjcf_root = mjcf.RootElement(model=model_name)

        # Agent body
        body = self._mjcf_root.worldbody.add('body', name='body')

        # Capsule body: radius=0.3m, half-height=0.4m, mass=75kg
        # contype=2 (bit 1): agents can collide with each other
        # conaffinity=2 (bit 1): agents accept collisions from other agents
        body.add('geom', name='capsule', type='capsule',
                 size=[0.3, 0.4],
                 pos=[0, 0, 0.7],
                 mass=75.0,
                 contype=2, conaffinity=2,
                 rgba=[1.0, 0.2, 0.2, 1.0] if team == 0 else [0.2, 0.2, 1.0, 1.0])

        # Stick hitbox extending forward from player
        # contype=1 (bit 0): stick collides with puck (puck conaffinity=5 has bit 0)
        # conaffinity=1: stick accepts collisions from puck
        body.add('geom', name='stick', type='box',
                 size=[0.1, 0.5, 0.05],
                 pos=[0.4, 0, 0.3],
                 mass=1.0,
                 contype=1, conaffinity=1,
                 rgba=[0.6, 0.4, 0.2, 1.0])

        # Slide joints for x/y translation with limits to keep agents inside rink
        # Range slightly inside rink boundaries (rink_length/2 - 0.5 = 14.5, rink_width/2 - 0.5 = 7.0)
        body.add('joint', name='x', type='slide', axis=[1, 0, 0],
                 limited=True, range=[-14.5, 14.5],
                 damping=10.0)
        body.add('joint', name='y', type='slide', axis=[0, 1, 0],
                 limited=True, range=[-7.0, 7.0],
                 damping=10.0)

        # Hinge joint for z-rotation (facing direction)
        body.add('joint', name='rot', type='hinge', axis=[0, 0, 1],
                 damping=5.0)

        # Velocity-controlled actuators
        self._mjcf_root.actuator.add('velocity', name='vx',
                                     joint=body.find('joint', 'x'), kv=200.0)
        self._mjcf_root.actuator.add('velocity', name='vy',
                                     joint=body.find('joint', 'y'), kv=200.0)
        self._mjcf_root.actuator.add('velocity', name='vrot',
                                     joint=body.find('joint', 'rot'), kv=50.0)

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def team(self):
        return self._team

    @property
    def player_idx(self):
        return self._player_idx
