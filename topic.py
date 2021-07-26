import numpy as np

class ObjectState:
    def __init__(
            self, qpos=None, qvel=None, spool_qpos=None, spool_qvel=None, spool_curr=None):
        self.qpos = qpos
        self.qvel = qvel
        self.spool_qpos = spool_qpos
        self.spool_qvel = spool_qvel
        self.spool_curr = spool_curr

class ObjectListener:
    def __init__(self):
        import rospy
        from rospy_tutorials.msg import Floats
        from geometry_msgs.msg import Pose
        # NOTE: The object state will add on these offsets after reading
        # the OptiTrack data in order to align with the Sawyer axes

        """
        ### Calibration

        1. align the object such that the purple end at the middle of the reset wall
        2. Take a picture of optitrack terminal
        3. move the sawyer arm such that (palm as flat as possible) and (fingers touch the sides of reset wall)
        4. Take a picture of sawyer terminal
        5. enter the numbers below
        """


        sawyer_cali = np.array([0.75, 0.07, 0.01])
        object_cali = np.array([0.248, 0.059, 0.297])

        # swap the axis for object coord
        object_cali = np.array([-object_cali[0],
                                object_cali[2],
                                object_cali[1]
                                ])

        cali_offset = sawyer_cali - object_cali

        self.OFFSETS = np.array([
            # Sawyer coords - OptiTrack coords
            # i.e.: OptiTrack read coords + offset = Sawyer read coords
            # ROD OFFSETS
            # 0.539 - (0.128),
            # -0.05 - (-0.028),
            # 0.161 - (0.247 - 0.05),  # Z offsets
            # VALVE OFFSETS
            cali_offset[0],
            cali_offset[1],
            cali_offset[2],
            # TODO: Need to fix the OptiTrack axes?
            0, 0, 0, 0,
        ])
        self._object_state = ObjectState(
            qpos=np.zeros(7),
            qvel=np.zeros(6),
        )
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("get_object_pose", Pose, self.update_qpos)
        # rospy.sleep(1) #could be causing bug for object xyz tracking

    def update_qpos(self, pose):
        # notice to change both xyz and quat
        self._object_state.qpos = np.array([
            -pose.position.x,
            pose.position.z,
            pose.position.y, # Note: this is for correcting the different axis permutation between optitrack and sawyer
            pose.orientation.w,
            -pose.orientation.x,
            pose.orientation.z,
            pose.orientation.y,
        ]) + self.OFFSETS

    def update_qvel(self, qvel):
        qvel_data = np.array(qvel.data)
        self._object_state.qvel = qvel_data

    def get_state(self):
        return self._object_state

