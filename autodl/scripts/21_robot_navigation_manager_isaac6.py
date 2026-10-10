class RobotNavigationManager:
    """Isaac 6 shim used while dynamic people avoidance is disabled."""

    def __init__(self, robot_name, robot, navmesh_enabled, dynamic_avoidance_enabled=False):
        self.robot_name = robot_name
        self.robot = robot
        self.navmesh_enabled = navmesh_enabled
        self.dynamic_avoidance_enabled = dynamic_avoidance_enabled

    def publish_robot_position(self, delta_time, radius):
        return None

    def destroy(self):
        self.robot = None
