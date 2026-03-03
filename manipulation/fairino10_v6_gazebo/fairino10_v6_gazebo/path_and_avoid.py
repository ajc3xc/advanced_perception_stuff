import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy , PlanRequestParameters
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import CollisionObject #helps define the collision object
from shape_msgs.msg import SolidPrimitive # helps define object shape

class Planner(Node):
    def __init__(self):
        super().__init__('path_and_avoidance_node')

    def collision_objects(self, planning_scene_monitor):
        self.get_logger().info('Adding collision objects to the planning scene.')
        with planning_scene_monitor.read_write() as scene:
            # Ground plane collision object
            # Positioned below the robot base to prevent planning into the floor
            ground = CollisionObject()
            ground.id = 'ground'
            ground.header.frame_id = 'world'

            ground_shape = SolidPrimitive()
            ground_shape.type = SolidPrimitive.BOX
            ground_shape.dimensions = [10.0, 10.0, 0.005]

            ground_pose = Pose()
            ground_pose.position.x = 0.0
            ground_pose.position.y = 0.0
            ground_pose.position.z = -0.005
            ground_pose.orientation.w = 1.0

            ground.primitives = [ground_shape]
            ground.primitive_poses = [ground_pose]
            ground.operation = CollisionObject.ADD

            # C1: Left obstacle culm
            culm_1 = CollisionObject()
            culm_1.id = 'culm_1'
            culm_1.header.frame_id = 'world'
            
            culm_1_shape = SolidPrimitive()
            culm_1_shape.type = SolidPrimitive.CYLINDER
            culm_1_shape.dimensions = [3.0, 0.05] 
            
            culm_1_pose = Pose()
            culm_1_pose.position.x = 1.0
            culm_1_pose.position.y = 0.25
            culm_1_pose.position.z = 1.5 
            culm_1_pose.orientation.w = 1.0
            
            culm_1.primitives = [culm_1_shape]
            culm_1.primitive_poses = [culm_1_pose]
            culm_1.operation = CollisionObject.ADD

            # C2: Right obstacle culm
            culm_2 = CollisionObject()
            culm_2.id = 'culm_2'
            culm_2.header.frame_id = 'world'
            
            culm_2_shape = SolidPrimitive()
            culm_2_shape.type = SolidPrimitive.CYLINDER
            culm_2_shape.dimensions = [3.0, 0.05]
            
            culm_2_pose = Pose()
            culm_2_pose.position.x = 1.0
            culm_2_pose.position.y = -0.25
            culm_2_pose.position.z = 1.5
            culm_2_pose.orientation.w = 1.0
            
            culm_2.primitives = [culm_2_shape]
            culm_2.primitive_poses = [culm_2_pose]
            culm_2.operation = CollisionObject.ADD

            # Apply object to the planning scene
            scene.apply_collision_object(ground)
            scene.apply_collision_object(culm_1)
            scene.apply_collision_object(culm_2)

            scene.current_state.update() #Update the planning scene state after adding the object

    def plan_and_execute(self,robot,planning_component, planner_id):
        plan_params = PlanRequestParameters(robot, planner_id) # loads the specified planner and finds its configuration profile

        planning_component.set_start_state_to_current_state() # sets the start state to the current state of the robot

        # Define the target pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'world'
        target_pose.pose.position.x = 1.2
        target_pose.pose.position.y = 0.0
        target_pose.pose.position.z = 0.25

        target_pose.pose.orientation.x = 0.5
        target_pose.pose.orientation.y = 0.5
        target_pose.pose.orientation.z = 0.5
        target_pose.pose.orientation.w = 0.5

        planning_component.set_goal_state(pose_stamped_msg = target_pose,pose_link='wrist3_link')# sets the goal pose for the end effector
        self.get_logger().info(f'Planning to the target pose with planner: {planner_id}')

        plan_result = planning_component.plan(plan_params) # generates a plan using the specified planner

        if plan_result:
            self.get_logger().info('Plan found, executing...')
            robot.execute(plan_result.trajectory,controllers = ['fairino10_controller']) # executes the planned trajectory using the specified controller
            self.get_logger().info('Execution complete.')
        else:
            self.get_logger().error('No plan found.')

def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    node.get_logger().info('Path and Avoidance Node has started.')

    # MoveItPy setup
    fairino10_v6_robot = MoveItPy(node_name='moveit_planning_node')
    arm = fairino10_v6_robot.get_planning_component('fairino10_v6_group')
    planning_scene_monitor = fairino10_v6_robot.get_planning_scene_monitor()

    # Populate the planning scene with collision objects
    node.collision_objects(planning_scene_monitor)

    # Set Planner ID and plan and execute a path
    node.plan_and_execute(fairino10_v6_robot, arm, planner_id="ompl_rrtc")


    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()