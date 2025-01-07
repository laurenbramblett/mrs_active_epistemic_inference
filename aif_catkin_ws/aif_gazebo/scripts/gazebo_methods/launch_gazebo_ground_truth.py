#!/usr/bin/env python

import yaml
import subprocess
import rospy
import time
import rospkg

def get_yaml_params(file_path):
    with open(file_path, 'r') as file:
        yaml_params = yaml.safe_load(file)
    return yaml_params


def launch_ground_truth(params, params_file):
    # Get the paths to the rotors_gazebo and rotors_description packages
    rospack = rospkg.RosPack()
    rotors_gazebo_path = rospack.get_path('rotors_gazebo')
    rotors_description_path = rospack.get_path('rotors_description')
    # Create a new tmux session
    session_name = 'jackal_nodes'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name, '-n', 'main'])
    time.sleep(3)

    for i, agent in enumerate(params['agents']):
        namespace = agent['namespace']
        config = agent['sensors']
        config_id = agent['id']
        scan_topic = agent['transformed_scan_topic']
        print(agent)
        init_x = agent['init_x']
        init_y = agent['init_y']
        init_z = agent['init_z']
        init_yaw = agent['init_yaw']
        service_topic = str('/' + params['aif_service_topic'])

        if agent['agent_type'] == 'A':
            # Define the launch command
            spawn_command = (
                f"roslaunch multi_jackal_base jackal_base.launch "
                f"ns:={namespace} "
                f"config:={config} "
                f"config_id:={config_id} "
                f"use_move_base:=false "
                f"x:={init_x} "
                f"y:={init_y} "
                f"yaw:={init_yaw}; read"
            )
        else:
            # Define the launch command
            spawn_command = (
                f"roslaunch rotors_gazebo spawn_aif_firefly.launch "
                f"mav_name:=firefly "
                f"namespace:={namespace} "
                f"init_x:={init_x} "
                f"init_y:={init_y} "
                f"init_z:={init_z} "
                f"init_yaw:={init_yaw}; read"
            )

        aif_command = (
            f"rosrun multi_jackal_aif run_gazebo_aif.py "
            f"_agent_id:={config_id} _params_file:={params_file} "
            f"_planning_service_topic:={service_topic}; read"
        )

        # Create a new window for the agent
        window_name = f'{namespace}'
        subprocess.run(['tmux', 'new-window', '-t', session_name, '-n', window_name])
        time.sleep(1)  # Add a short delay to ensure the window is created

        # Send the commands to the new tmux window
        subprocess.run(['tmux', 'split-window', '-t', f'{session_name}:{window_name}.0', spawn_command])
        subprocess.run(['tmux', 'split-window', '-t', f'{session_name}:{window_name}.1', aif_command])

        # Rename the panes
        subprocess.run(['tmux', 'select-pane', '-t', f'{session_name}:{window_name}.0', '-T', f'spawn_{i}'])
        subprocess.run(['tmux', 'select-pane', '-t', f'{session_name}:{window_name}.1', '-T', f'aif_{i}'])

        # Make the tiling pretty
        subprocess.run(['tmux', 'select-layout', '-t', f'{session_name}:{window_name}', 'tiled'])
        time.sleep(3)  # Add a short delay to ensure the window is created

    print('Nodes launched successfully')
    # Attach to the tmux session
    subprocess.run(['tmux', 'attach-session', '-t', session_name])

if __name__ == '__main__':
    rospy.init_node('launch_ground_truth')
    params_file = rospy.get_param('params_file')
    params = get_yaml_params(params_file)
    launch_ground_truth(params, params_file)
