#!/usr/bin/env python

import rospy
import sensor_msgs.msg as SM
import geometry_msgs.msg as GM
import tf_lookup
import numpy
import math
import tf.transformations as TT

class SpeedLimitException(Exception):
    pass

class Transformer:
    """
    A helper class to perform transformations on speed vectors, twists.
    This is not generic.
    """
    def __init__(self, base_link="/base_link", fixed=True):
        self.transforms = {}
        self.fixed = fixed
        self.base_link = base_link

        if self.fixed:
            self.tfl_client = tf_lookup.TfLookupClient()
        else:
            self.tfs_client = tf_lookup.TfStreamClient()
            self.tfs_handles = {}

    def quat(self, q):
        return numpy.array([q.x, q.y, q.z, q.w])

    def vect(self, v):
        return numpy.array([v.x, v.y, v.z])

    def no_z(self, v):
        v[2] = 0
        return v

    def add_frame(self, frame_name):
        """ Get the transformation from /base_link to the sensor frame. This
        will get it only once if the sensor is fixed, or get a stream from
        tf_lookup otherwise. """
        if frame_name in self.transforms:
            return
        if self.fixed:
            self.tfl_client.query_transform(self.base_link, frame_name, self._tf_callback)
        else:
            self.tfs_handles[frame_name] = self.tfs_client.add_transform(
                self.base_link, frame_name, self._tf_callback)

    def _tf_callback(self, data):
        self.transforms[data.child_frame_id] = data.transform

    def transform_vel(self, vel, frame_name):
        """ Returns the effective velocity at the given frame, but expressed
        in the /base_link coordinate frame. All calculations are done in
        /base_link frame.  """
        if not frame_name in self.transforms:
            raise SpeedLimitException("no transform exists for frame '{}'".format(frame_name))
        trp = self.vect(self.transforms[frame_name].translation)
        out_vel = GM.Twist()
        out_vel.angular = vel.angular
        out_vel.linear = GM.Vector3(*(self.vect(vel.linear)
                                      + numpy.cross(self.vect(vel.angular), trp)))
        return out_vel

    def x_axis(self, frame_name):
        return self.transform_axis(frame_name, (1, 0, 0))

    def y_axis(self, frame_name):
        return self.transform_axis(frame_name, (0, 1, 0))

    def z_axis(self, frame_name):
        return self.transform_axis(frame_name, (0, 0, 1))

    def transform_axis(self, frame_name, axis):
        if not frame_name in self.transforms:
            raise SpeedLimitException("no transform exists for frame '{}'".format(frame_name))
        rotM = TT.quaternion_matrix(self.quat(self.transforms[frame_name].rotation))
        posH = numpy.resize(axis, 4)
        posH[3] = 1
        return numpy.dot(rotM, posH)[:3]

    def clamp(self, val, b1, b2):
        a = min(b1, b2)
        b = max(b1, b2)
        if val < a:
            return a
        if val > b:
            return b
        return val

    def add_vel_to_base(self, frame_name, twist_base, vel):
        """ This method merges a velocity applied to one of the sensors into
        the mobile base velocity. It uses tricks to provide an acceptable
        behavior.  """
        if not frame_name in self.transforms:
            raise SpeedLimitException("no transform exists for frame '{}'".format(frame_name))
        ang_base = self.vect(twist_base.angular)
        lin_base = self.vect(twist_base.linear)
        trp = self.no_z(self.vect(self.transforms[frame_name].translation))
        rot_part = numpy.cross(trp, vel) / numpy.linalg.norm(trp) / numpy.linalg.norm(trp)
        lin_part = vel - numpy.cross(rot_part, trp)
        ang_base[2] = self.clamp(ang_base[2] + rot_part[2], 0.0, ang_base[2])
        lin_base[0] += lin_part[0]
        out_twist = GM.Twist()
        out_twist.linear = GM.Vector3(*lin_base)
        out_twist.angular = GM.Vector3(*ang_base)
        return out_twist


class SpeedLimiter:
    """
    Given a sensor topic and a few parameters, this class provides
    a way to restrain the velocity of the mobile base (/base_link)
    to safer values, in order to avoid collisions with stuff.
    """
    msg_type = {
        'laser': SM.LaserScan,
        'range': SM.Range,
    }
    defaults = {
        'obstacle_max_dist': 10.0, # [m] this should be a lot
        'stop_dist':          0.0, # [m] continuous behavior by default
        'speed_factor':       1.0, # [s-1] how much m.s-1 per m of range
    }

    def __init__(self, cfg_item):
        self.type = cfg_item['type'] # this parameter is required and has no default value
        if self.type not in self.msg_type:
            raise SpeedLimitException("type '{}' not implemented".format(self.type))

        self._read_config(cfg_item)
        self.sensors = {}
        self.tf = Transformer(fixed=cfg_item['fixed'])
        self.subscriber = rospy.Subscriber(cfg_item['topic'], self.msg_type[self.type], self._topic_callback)

    def limit_speed(self, vel):
        """ Returns a Twist, which will hopefully be safer to use in
        cluttered environments and should remain unchanged otherwise. """
        out_vel = vel
        for f, r in self.sensors.iteritems():
            try:
                sensor_vel = self.tf.vect(self.tf.transform_vel(out_vel, f).linear)
            except SpeedLimitException:
                self.tf.add_frame(f)
                continue
            srange, sdir = self._get_range(f)
            if srange > self.obstacle_max_dist:
                continue
            allowed_speed_in_sdir = srange * self.speed_factor
            projected_vel = numpy.dot(sensor_vel, sdir)
            if projected_vel < allowed_speed_in_sdir:
                continue
            if srange < self.stop_dist:
                return GM.Twist()
            vel_diff = -(projected_vel - allowed_speed_in_sdir) * self.tf.x_axis(f)
            if numpy.linalg.norm(vel_diff) > numpy.linalg.norm(sensor_vel):
                vel_diff *= numpy.linalg.norm(sensor_vel)/numpy.linalg.norm(vel_diff)
            out_vel = self.tf.add_vel_to_base(f, out_vel, vel_diff)
        return out_vel

    def _read_config(self, cfg):
        for c in self.defaults:
            if c in cfg:
                setattr(self, c, cfg[c])
            else:
                setattr(self, c, self.defaults[c])

    def _get_range(self, frame_name):
        srange = None
        sdir = None
        r = self.sensors[frame_name]
        if self.type == 'laser':
            lx = self.tf.x_axis(frame_name)
            ly = self.tf.y_axis(frame_name)
            ranges = list(r.ranges)
            for i,rr in enumerate(ranges):
                if rr < r.range_min:
                    ranges[i] = 1e3
            idx = r.ranges.index(min(ranges))
            hdg = idx * r.angle_increment + r.angle_min
            sdir = math.cos(hdg) * lx + math.sin(hdg) * ly
            srange = r.ranges[idx]
        elif self.type == 'range':
            srange = r.range
            if srange > r.max_range - 0.01:
                srange = 1e3
            sdir = self.tf.x_axis(frame_name)
        else:
            raise SpeedLimitException("panic.")
        return srange, sdir

    def _topic_callback(self, data):
        frame = data.header.frame_id
        if frame[0] != '/':
            frame = '/' + frame
        self.sensors[frame] = data
        self.tf.add_frame(frame)


class SpeedLimit:
    """
    An attempt at safer mobile base operation in cluttered and/or
    dynamic environments.
    ros param    : speed_limit/*
    subscribes to: speed_limit/cmd_vel_in
                 : [topics specified in config]
    publishes    : speed_limit/cmd_vel_out
    """
    def __init__(self):
        if not rospy.has_param('speed_limit'):
            rospy.logfatal('no configuration found')
            raise SpeedLimitException

        rospy.loginfo("speed_limit starting")
        conf = rospy.get_param('speed_limit')
        self.limiters = {}
        self.sub = rospy.Subscriber("~cmd_vel_in", GM.Twist, self._vel_cb)
        self.pub = rospy.Publisher("~cmd_vel_out", GM.Twist)

        for k,c in conf.iteritems():
            self.limiters[k] = SpeedLimiter(c)

    def limit_speed(self, vel):
        out_vel = vel
        for n,l in self.limiters.iteritems():
            out_vel = l.limit_speed(out_vel)
        return out_vel

    def _vel_cb(self, data):
        safe_vel = self.limit_speed(data)
        self.pub.publish(safe_vel)


def main():
    rospy.init_node("speed_limit")
    SpeedLimit()
    rospy.spin()

if __name__ == "__main__":
    main()
