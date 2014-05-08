import numpy
import rospy
import geometry_msgs.msg as GM
import tf.transformations as TT
from tf_lookup import TfStreamClient
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class TfStreamFilter:
    def __init__(self):
        self.transforms = {}
        self.tfs_client = TfStreamClient()
        self.frame_id = "odom"
        self.goal = None
        self.cb = None

    def tf_and_call(self, goal, cb):
        self.goal = goal
        self.cb = cb
        self._worker()

    def _worker(self):
        if self.goal is None:
            return
        trg = self._transform(self.goal)
        if trg is not None:
            self.goal = None
            self.cb(trg)

    def _mat44_from_transform(self, tr):
        translation = [tr.translation.x, tr.translation.y, tr.translation.z]
        rotation = [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]
        return numpy.dot(TT.translation_matrix(translation),
                         TT.quaternion_matrix(rotation))

    def _mat44_from_pose(self, ps):
        pos = ps.pose.position
        ori = ps.pose.orientation
        pos44 = TT.translation_matrix([pos.x, pos.y, pos.z])
        ori44 = TT.quaternion_matrix([ori.x, ori.y, ori.z, ori.w])
        return numpy.dot(pos44, ori44)

    def _transform(self, goal):
        frame_id = goal.header.frame_id
        if frame_id not in self.transforms:
            self.tfs_client.add_transform(self.frame_id,
                                          goal.header.frame_id,
                                          self._tfs_cb)
            return None
        mat44 = self._mat44_from_transform(self.transforms[frame_id])
        pose44 = self._mat44_from_pose(goal)
        trg44 = numpy.dot(mat44, pose44)
        xyz = tuple(TT.translation_from_matrix(trg44))[:3]
        quat = tuple(TT.quaternion_from_matrix(trg44))
        trg = goal
        trg.pose.position = GM.Vector3(*xyz)
        trg.pose.orientation = GM.Quaternion(*quat)
        trg.header.frame_id = self.frame_id
        return trg

    def _tfs_cb(self, data):
        self.transforms[data.child_frame_id] = data.transform
        self._worker()


class FollowMe:
    """
    This class is a convenience python interface to the follow_planner
    """
    def __init__(self):
        self.al_client = SimpleActionClient('move_base', MoveBaseAction)
        self.goal_pub_ = rospy.Publisher('move_base/follow_goal', GM.PoseStamped)
        self.started = False
        self.tfsf = TfStreamFilter()

    def start(self):
        if self.started:
            return True
        if not self.al_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("move_base action server could not be reached")
            return False
        self.started = True
        self._send_dummy_goal()

    def stop(self):
        self.al_client.cancel_all_goals()
        self.started = False
        return True

    def update(self, goal):
        self.tfsf.tf_and_call(goal, self._tfsf_cb)

    def _tfsf_cb(self, trgoal):
        self.goal_pub_.publish(trgoal)

    def _send_dummy_goal(self):
        pose = GM.PoseStamped()
        pose.header.frame_id = "odom"
        pose.pose.orientation.w = 1.0
        self.al_client.send_goal(MoveBaseGoal(pose), done_cb=self._move_base_cb)

    def _move_base_cb(self):
        if self.started:
            self._send_dummy_goal()
