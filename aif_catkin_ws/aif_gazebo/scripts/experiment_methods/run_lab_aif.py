#!/usr/bin/env python
# import aif_functions_isobeliefs_convergent_lab as aif
import aif_functions_isobeliefs_convergent_lab as aif
import numpy as np
import rospy, copy, yaml
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Bool
import tf, itertools

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
        cmd_vel_topic = system_dict['agents'][id]['cmd_vel_topic']
        # pose_topic = system_dict['agents'][id]['pose_topic']
        self.cmd_vel_pub = rospy.Publisher(namespace + '/' + cmd_vel_topic, Twist, queue_size=1)
        self.num_agents = len(system_dict['agents'])
        self.num_goals = len(system_dict['goals'])
        pose_topic_lists = [system_dict['agents'][i]['pose_topic'] for i in range(self.num_agents)]
        self.pose_sub = [rospy.Subscriber(pose_topic_lists[i], TransformStamped, self.pose_callback, i) for i in range(self.num_agents)]
        print("Subscribed to pose topics: ", pose_topic_lists)
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
        if self.convergence_type == 'exclusive':
            tuple_elements = [i for i in range(self.agent_positions.shape[0])]
            configurations = list(itertools.permutations(tuple_elements))
            self.reward_configs = configurations # Reward configurations if different goals
        else:
            self.reward_configs = [tuple(np.repeat(i, self.num_agents)) for i in range(self.num_goals)]
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
            'dt': self.system_dict['dt'],
            'convergence_threshold': self.system_dict['convergence_threshold'],
            }
        if self.system_dict['use_ep']:
            agent_vars['prior'] = np.tile(1/len(agent_vars['reward_configs']), (len(agent_vars['agent_positions']), len(agent_vars['reward_configs'])))      
        else:
            agent_vars['prior'] = np.ones(len(agent_vars['reward_configs']), dtype=float) / len(agent_vars['reward_configs'])
        self.agent_vars = agent_vars


    def get_yaml_params(self, file_path):
        with open(file_path, 'r') as file:
            yaml_params = yaml.safe_load(file)
        return yaml_params

    def pose_callback(self,msg,id):
        """Callback function to update all robot's poses."""
        # print("Got pose: ", msg)
        self.agent_positions[id] = np.zeros((1, 3))
        position = np.array([msg.transform.translation.x, msg.transform.translation.y])
        # Convert orientation to yaw angle
        orientation = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(orientation)
        self.agent_positions[id] = np.concatenate((position, [yaw]))
        # print("Got pose: ", self.agent_positions[id])
        self.received_observations[id] = True
    
    def start_planning(self, msg):
        """Callback function to start planning."""
        self.planning = msg.data

    def move(self, linear, angular):
        self.vel.linear.x = linear
        self.vel.angular.z = angular
        self.cmd_vel_pub.publish(self.vel)

    def get_observations(self):
        return self.agent_positions
    
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
        best_velocity, best_heading, best_value = aif.choice_heuristic(observed_positions, observations, consensus_prior, self.agent_vars, use_ep=use_ep, consensus=use_ep)
        best_action = (best_velocity, best_heading)
        # If distance to goal is less than a threshold, then the agent has reached the goal
        distances_to_goals = np.linalg.norm(agent_positions[agent_id][:2] - self.agent_vars['goals'], axis=1)
        print("Distances to goals: ", distances_to_goals)
        if np.linalg.norm(distances_to_goals<self.convergence_threshold).any():
            print("Agent {} has reached the goal!".format(agent_id))
            self.planning = False
        
        return best_action, observations, best_value
    def main(self):
        rate = rospy.Rate(1./self.system_dict['dt'])
        while not rospy.is_shutdown():
            if not self.planning or not all(self.received_observations):
                rate.sleep()
                continue
            # self.agent_positions = np.array([self.agent_positions[i] for i in range(self.num_agents)])
            best_action, observations, goal_scores = self.make_decision()
            self.move(best_action[0], best_action[1])
            self.agent_vars['prior'] = aif.get_likelihood(self.agent_id,observations, self.agent_vars['goals'], 
                                                          self.agent_vars, self.agent_vars['prior'], self.agent_vars['use_ep'])
            rate.sleep()

    
if __name__ == '__main__':
    rospy.init_node('aif_run_lab_{}'.format(str(np.random.randint(1000))), anonymous=True)
    id = rospy.get_param('~agent_id',0)
    params_file = rospy.get_param('~params_file')
    planning_service_topic = rospy.get_param('~planning_service_topic')
    robot = robot_controller(id, params_file, planning_service_topic)
    robot.parse_agent_vars()
    robot.main()

