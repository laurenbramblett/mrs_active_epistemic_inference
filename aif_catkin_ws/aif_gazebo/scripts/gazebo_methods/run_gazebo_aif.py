#!/usr/bin/env python
import aif_functions_isobeliefs_convergent_gazebo as aif
import aif_functions_iterative_sampling_gazebo as aif_iterative
import numpy as np
import rospy, copy, yaml
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
import tf, itertools
from visualization_msgs.msg import Marker, MarkerArray

class robot_controller:
    """Class to control a single robot in the multi-robot system and make decisions based on our AIF algorithm."""
    def __init__(self, id, params_file, planning_service_topic):
        # Load parameters from the yaml file
        system_dict = self.get_yaml_params(params_file)
        # Planning boolean
        self.planning_sub = rospy.Subscriber(planning_service_topic, Bool, self.start_planning)
        self.planning = False
        # Initialize ROS publishers and subscribers for robot
        namespace = system_dict['agents'][id]['namespace']
        move_topic = system_dict['agents'][id]['move_topic']
        self.pose_topic = namespace + '/' + system_dict['agents'][id]['pose_topic']
        self.move_cmd = rospy.Publisher(namespace + '/' + move_topic, Twist, queue_size=1)  if system_dict['agents'][id]['agent_type'] == 'A' else rospy.Publisher(namespace + '/' + move_topic, PoseStamped, queue_size=1)
        print("Move topic: ", namespace + '/' + move_topic, " Agent type: ", system_dict['agents'][id]['agent_type'])
        self.num_agents = len(system_dict['agents'])
        self.num_goals = len(system_dict['goals'])
        self.goals = np.array(system_dict['goals'])
        pose_topic_lists = [system_dict['agents'][i]['namespace'] + '/' + system_dict['agents'][i]['pose_topic'] for i in range(self.num_agents)]
        self.pose_sub = [rospy.Subscriber(pose_topic_lists[i], PoseWithCovarianceStamped, self.pose_callback, i) for i in range(self.num_agents)]
        self.dt = system_dict['dt']
        # Initialize variables for robot
        self.vel = Twist()
        self.velocity_options = np.linspace(system_dict['agents'][id]['velocity_min'], system_dict['agents'][id]['velocity_max'], system_dict['agents'][id]['velocity_arg_num'], endpoint=True)
        self.heading_options = np.linspace(system_dict['agents'][id]['heading_min'], system_dict['agents'][id]['heading_max'], system_dict['agents'][id]['heading_arg_num'], endpoint=True)
        # Initialize variables for decision-making 
        self.agent_id = id
        self.agent_positions = np.zeros((self.num_agents, 3))
        self.received_observations = [False for i in range(self.num_agents)]
        self.agent_types = [system_dict['agents'][i]['agent_type'] for i in range(self.num_agents)]
        system_dict['agent_types'] = self.agent_types
        self.convergence_threshold = system_dict['convergence_threshold']
        # Initialize environmental variables
        self.system_dict = system_dict
        self.convergence_type = system_dict['convergence_type']
        self.max_reward_configs = np.min((system_dict['max_reward_configs'], np.math.factorial(self.num_agents)))
        self.path = Path()
        self.goal_pub = rospy.Publisher(f"goal_markers_{self.agent_id}", MarkerArray, queue_size=1, latch=True)
        print("Max reward configs: ", self.max_reward_configs)
        if self.convergence_type == 'exclusive':
            tuple_elements = [i for i in range(self.agent_positions.shape[0])]
            configurations = list(itertools.permutations(tuple_elements))
            self.reward_configs = configurations # Reward configurations if different goals
        elif self.convergence_type == 'convergent':
            self.reward_configs = [tuple(np.repeat(i, self.num_agents)) for i in range(self.num_goals)]
        else:
            self.reward_configs = [tuple()]*self.max_reward_configs

        

    # Configure agent variables (this is so I can use scripts from simulation)
    def parse_agent_vars(self):
        agent_vars = {
            'agent_id': self.agent_id,
            'home_base': self.system_dict['agents'][self.agent_id]['home_base'],
            'agent_type': self.system_dict['agents'][self.agent_id]['agent_type'],
            'agent_positions': self.agent_positions,
            'agent_types': self.agent_types,
            'num_agents': self.num_agents,
            'goals': np.array(self.system_dict['goals']),
            'velocity_options': self.velocity_options,
            'heading_options': self.heading_options,
            'num_actions': len(self.velocity_options) * len(self.heading_options),
            'max_distance_measure': self.system_dict['max_distance_measure'],
            'observation_error_std': self.system_dict['observation_error_std'],
            'use_ep': self.system_dict['use_ep'],
            'use_mcts': self.system_dict['use_mcts'],
            'reward_configs': self.reward_configs,
            'mcts_iterations': self.system_dict['mcts_iterations'],
            'horizon': self.system_dict['horizon'],
            'use_threading': self.system_dict['use_threading'],
            'use_rhc': self.system_dict['use_rhc'],
            'dt': self.system_dict['agents'][self.agent_id]['dt'],
            'convergence_threshold': self.convergence_threshold,
            'convergence_type': self.convergence_type,
            'greedy': self.system_dict['greedy'],
            'use_iterative': self.system_dict['use_iterative'],
            }
        self.agent_vars = agent_vars
        self.agent_vars['prior'] = aif_iterative.set_initial_prior(self.agent_vars)


    def get_yaml_params(self, file_path):
        with open(file_path, 'r') as file:
            yaml_params = yaml.safe_load(file)
        return yaml_params

    def pose_callback(self,msg,id):
        """Callback function to update all robot's poses."""
        self.agent_positions[id] = np.zeros((1, 3))
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        # Convert orientation to yaw angle
        orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(orientation)
        self.agent_positions[id] = np.concatenate((position, [yaw]))
        # print("Got pose: ", self.agent_positions[id])
        self.agent_vars['agent_positions'] = self.agent_positions
        self.received_observations[id] = True
    
    def start_planning(self, msg):
        """Callback function to start planning."""
        self.planning = msg.data

    def move(self, linear, angular):
        if self.agent_vars['agent_type'] == 'A':
            self.move_ground(linear, angular)
        else:
            self.move_aerial(linear, angular)

    def move_ground(self, linear, angular):
        self.vel.linear.x = linear
        self.vel.angular.z = angular
        self.move_cmd.publish(self.vel)
    
    def move_aerial(self, linear, angular):
        # print("Moving aerial agent, linear: ", linear, " angular: ", angular)
        agent_x = self.agent_positions[self.agent_id][0]
        agent_y = self.agent_positions[self.agent_id][1]
        agent_yaw = self.agent_positions[self.agent_id][2]
        heading = aif_iterative.wrapToPi(angular + agent_yaw)
        pose_x = agent_x + linear*np.cos(heading)
        pose_y = agent_y + linear*np.sin(heading)
        # print("Moving to: ", pose_x, pose_y, heading)
        pose = PoseStamped()
        pose.header.frame_id = self.system_dict['world_frame_id']
        pose.pose.position.x = pose_x
        pose.pose.position.y = pose_y
        pose.pose.position.z = 1.0
        quat = tf.transformations.quaternion_from_euler(0, 0, heading)
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        self.move_cmd.publish(pose)
    
    def publish_goal_markers(self,goals_completed):
        markers = MarkerArray()
        # Delete existing markers
        for i in range(len(self.goals)):
            delete_marker = Marker()
            delete_marker.header.frame_id = self.system_dict['world_frame_id']
            delete_marker.header.stamp = rospy.Time.now()
            delete_marker.ns = 'goal_markers_' + str(i)
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            markers.markers.append(delete_marker)
        
        # Create new markers for not completed goals
        for i in range(len(self.goals)):
            marker = Marker()
            marker.header.frame_id = self.system_dict['world_frame_id']
            marker.header.stamp = rospy.Time.now()
            marker.ns = 'goal_markers_' + str(i)
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.1
            if goals_completed[i]:
                marker.color.a = 1.0
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
            else:
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
            marker.pose.position.x = self.goals[i][0]
            marker.pose.position.y = self.goals[i][1]
            marker.pose.position.z = 0.01
            marker.pose.orientation.w = 1.0
            markers.markers.append(marker)
        # Publish markers
        self.goal_pub.publish(markers)

   
    def make_decision(self):
        # Initialize variables for decision-making
        use_ep = self.agent_vars['use_ep']
        agent_id = self.agent_vars['agent_id']
        prior = self.agent_vars['prior']
        observation_error_std = self.agent_vars['observation_error_std']
        agent_positions = self.agent_vars['agent_positions']
        agent_type = self.agent_vars['agent_type']

        # Store observations
        observations = []
        for idx in range(agent_positions.shape[0]):
            if idx == agent_id:
                observations.append(aif.simulate_observation(agent_positions[idx], 0, 's',idx)) # No noise for self-observation
            else:
                observations.append(aif.simulate_observation(agent_positions[idx], observation_error_std, agent_type,idx))


        observed_positions = aif.parse_observations(observations)
        # Calculate the likelihood of each goal being the target based on agent measurements -> May be able to turn into NN
        consensus_prior = prior
        if use_ep:
            consensus_prior = aif.softmax(aif.compute_consensus(prior, self.agent_vars['agent_types']))
            
        # Choose the action (for the current goal) that minimizes the free energy
        best_velocity, best_heading, best_value = aif_iterative.choice_heuristic(observed_positions, observations, consensus_prior, self.agent_vars, use_ep=use_ep, consensus=use_ep)
        best_action = (best_velocity, best_heading)
        # If distance to goal is less than a threshold, then the agent has reached the goal
        distances_to_goals = np.linalg.norm(agent_positions[agent_id][:2] - self.agent_vars['goals'], axis=1)
        # print("Distances to goals: ", distances_to_goals)
        if np.linalg.norm(distances_to_goals<self.convergence_threshold).any():
            print("Agent {} has reached the goal!".format(agent_id))
            self.planning = False
        
        return best_action, observations, best_value
    
    def make_decision_iterative(self):
        # Initialize variables for decision-making
        use_ep = self.agent_vars['use_ep']
        agent_id = self.agent_vars['agent_id']
        prior = self.agent_vars['prior']
        observation_error_std = self.agent_vars['observation_error_std']
        agent_positions = self.agent_vars['agent_positions']
        agent_type = self.agent_vars['agent_type']

        # Store observations
        observations = []
        for idx in range(agent_positions.shape[0]):
            if idx == agent_id:
                observations.append(aif_iterative.simulate_observation(agent_positions[idx], 0, 's',idx)) # No noise for self-observation
            else:
                observations.append(aif_iterative.simulate_observation(agent_positions[idx], observation_error_std, agent_type,idx))


        observed_positions = aif_iterative.parse_observations(observations)
        # Calculate the likelihood of each goal being the target based on agent measurements -> May be able to turn into NN
        consensus_prior = prior
        if use_ep:
            consensus_prior = aif_iterative.softmax(aif_iterative.compute_consensus(prior, self.agent_vars['agent_types']))
            
        # Choose the action (for the current goal) that minimizes the free energy
        best_velocity, best_heading, best_value = aif_iterative.choice_heuristic(observed_positions, observations, consensus_prior, self.agent_vars, use_ep=use_ep, consensus=use_ep)
        best_action = (best_velocity, best_heading)
        # If distance to goal is less than a threshold, then the agent has reached the goal
        distances_to_goals = np.linalg.norm(agent_positions[agent_id][:2] - self.agent_vars['goals'], axis=1)
        # print("Distances to goals: ", distances_to_goals)
        if np.linalg.norm(distances_to_goals<self.convergence_threshold).any():
            print("Agent {} has reached the goal!".format(agent_id))
            # self.planning = False
        
        return best_action, observations, best_value
    
    
    def main(self):
        #Mark completed goals
        goals_completed = np.zeros(self.num_goals, dtype=bool)
        goals_not_completed = self.goals[~goals_completed]
        rate = rospy.Rate(1./self.dt)
        agent_id = self.agent_vars['agent_id']
        # print("Goals are: ", self.goals)
        while not rospy.is_shutdown():
            if not self.planning or not all(self.received_observations):
                rate.sleep()
                continue
            if self.agent_vars['convergence_type'] in ['exclusive','convergent'] and not self.agent_vars['use_iterative']:
                best_action, observations, goal_scores = self.make_decision()
                self.move(best_action[0], best_action[1])
                self.agent_vars['prior'] = aif.get_likelihood(self.agent_id,observations, self.agent_vars['goals'], 
                                                            self.agent_vars, self.agent_vars['prior'], self.agent_vars['use_ep'])
                convergence_check, selected_goal = aif.check_convergence(self.agent_positions, self.agent_vars['goals'], self.convergence_type, self.convergence_threshold)
                self.publish_goal_markers(goals_completed)
                if convergence_check:
                    print("Agent {} has reached the goal!".format(agent_id))
                    self.planning = False
            else:
                goals_not_completed = self.goals[~goals_completed]
                # print("Goals not completed: ", goals_not_completed)
                self.reward_configs = aif_iterative.identify_reward_configs(self.agent_positions, self.agent_vars)
                self.agent_vars['reward_configs'] = self.reward_configs
                # print("Reward configs: ", self.reward_configs)
               
                # Make decision
                self.agent_vars['goals'] = aif_iterative.choose_subset_goals(goals_not_completed, self.agent_positions[:,:2], self.agent_types, self.agent_vars)
                best_action, observations, goal_scores = self.make_decision_iterative()
                self.move(best_action[0], best_action[1])
                self.agent_vars['prior'] = aif_iterative.get_likelihood(self.agent_id, observations, self.agent_vars['goals'], 
                                                            self.agent_vars, self.agent_vars['prior'], self.agent_vars['use_ep'])
                # print("Prior: ", self.agent_vars['prior'])
                # Check if agents have converged to the same goal
                new_prior, goals_completed = aif_iterative.delete_goals(self.goals, self.agent_positions[:,:2], goals_completed, self.convergence_threshold)
                if new_prior:
                    for agent_id in range(self.num_agents):
                        self.agent_vars['prior'] = aif_iterative.set_initial_prior(self.agent_vars)

                convergence_check, selected_goal = aif_iterative.check_convergence(self.agent_positions, self.agent_vars['goals'], self.convergence_type, goals_completed, self.convergence_threshold)
                self.publish_goal_markers(goals_completed)
                print("Agent positions: ", self.agent_positions[self.agent_id])
                if (goals_completed.all()) and np.linalg.norm(self.agent_positions[self.agent_id][:2] - self.agent_vars['home_base']) < self.convergence_threshold:
                    print("All goals have been completed!")
                    print("Agent {} has reached the goal!".format(agent_id))
                    self.planning = False
            rate.sleep()

    
if __name__ == '__main__':
    rospy.init_node('aif_run_gazebo_{}'.format(str(np.random.randint(1000))), anonymous=True)
    id = rospy.get_param('~agent_id',0)
    print("Agent ID: ", id)
    params_file = rospy.get_param('~params_file')
    planning_service_topic = rospy.get_param('~planning_service_topic')
    robot = robot_controller(id, params_file, planning_service_topic)
    robot.parse_agent_vars()
    robot.main()

