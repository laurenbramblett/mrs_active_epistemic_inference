#!/usr/bin/env python
## ROS service server that starts active inference when called

# import the service
from std_srvs.srv import SetBool, SetBoolResponse
from std_msgs.msg import Bool
import rospy 

NAME = 'start_active_inference_service'

class ActiveInferenceServer:
    def __init__(self, pub):
        self.pub = pub

    def start_active_inference(self):
        call_name = rospy.get_param('aif_service_call')
        s = rospy.Service(call_name, SetBool, self.call_active_inference)

        # spin() keeps Python from exiting until node is shutdown
        rospy.spin()

    def call_active_inference(self, req):
        if req.data:
            print("Starting active inference")
            self.pub.publish(True)
            return SetBoolResponse(success=True, message="Node started.")
        else:
            print("User entered value incompatible with start plan")
            self.pub.publish(False)
            return SetBoolResponse(success=False, message="Nothing happened.")

if __name__ == "__main__":
    rospy.init_node(NAME)
    pub_name = rospy.get_param('aif_service_topic')
    pub = rospy.Publisher(pub_name, Bool, queue_size=1, latch=True)
    pub.publish(False)
    server = ActiveInferenceServer(pub)
    server.start_active_inference()
