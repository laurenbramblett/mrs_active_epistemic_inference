#!/usr/bin/env python
import yaml, subprocess, rospy, time

# Load the YAML file
def get_yaml_params(file_path):
    with open(file_path, 'r') as file:
        yaml_params = yaml.safe_load(file)
    return yaml_params

# Loop through each agent and launch the node
def launch_ground_truth(params, params_file):
    # Create a new tmux session
    session_name = 'jackal_nodes'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name, '-n', 'main'])
    time.sleep(3)

    for i, agent in enumerate(params['agents']):
        namespace = agent['namespace']
        config = agent['sensors']
        config_id = agent['id']
        scan_topic = agent['transformed_scan_topic']
        init_x = agent['init_x']
        init_y = agent['init_y']
        init_yaw = agent['init_yaw']    
        service_topic = str('/' + params['aif_service_topic'])


        # Define the launch command           
        aif_command = f"rosrun multi_jackal_aif run_lab_aif.py " + \
                       f"_agent_id:={config_id} _params_file:={params_file} " + \
                       f"_planning_service_topic:={service_topic}; read"
        
        # Create a new window for the agent
        window_name = f'{namespace}'
        subprocess.run(['tmux', 'new-window', '-t', session_name, '-n', window_name])
        time.sleep(1)  # Add a short delay to ensure the window is created
        
        # Split the window for spawning the robot and launching the AIF
        subprocess.run(['tmux', 'split-window', '-t', f'{session_name}:{window_name}.0', aif_command])
        
        # Rename the panes
        subprocess.run(['tmux', 'select-pane', '-t', f'{session_name}:{window_name}.1', '-T', f'aif_{i}'])     
        
        # Make the tiling pretty
        subprocess.run(['tmux', 'select-layout', '-t', f'{session_name}:{window_name}', 'tiled'])
        # Add a short delay to avoid race conditions
        time.sleep(5)

    print('Nodes launched successfully')


    # Attach to the tmux session
    subprocess.run(['tmux', 'attach-session', '-t', session_name])

if __name__ == '__main__':
    rospy.init_node('launch_lab_robots')
    params_file = rospy.get_param('params_file')
    params = get_yaml_params(params_file)
    launch_ground_truth(params, params_file)



