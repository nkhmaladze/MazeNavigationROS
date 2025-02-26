#!/usr/bin/env python

import roslib; roslib.load_manifest('final_project')
import rospy
import rospkg
import tf
import transform2d
import numpy
import sys
import maze

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from kobuki_msgs.msg import SensorState

# we will run our controller at 100Hz
DELTA_T = 0.01

CONTROL_PERIOD = rospy.Duration(DELTA_T)

# one degree is pi/180 radians
DEG = numpy.pi / 180.0

# one foot is 0.3048 meters
FT = 0.3048

# one grid cell is 3 feet
CELL_SIZE = 3*FT

# look at +/- this angle to correct heading
HEADING_HALF_ANGLE = 5*DEG

############################################################
# for dealing with points

SEGMENT_DECIMATE = 8
SEGMENT_WINDOW_SIZE = 1
SEGMENT_MIN_COUNT = 3
SEGMENT_NORMAL_COS_TOL = 0.8
SEGMENT_NORMAL_DIST_TOL = 0.02

############################################################

ANGULAR_KP = 20 #8.5
ANGULAR_KD = 12 #4
ANGULAR_MAX_VEL = 5

# you should not need to change these 
ANGULAR_ERR_TOL = 3.0 * DEG
ANGULAR_VEL_TOL = 0.2

# used to determine when to switch from turnleft/turnright to straighten
ANGULAR_FINISH_TOL = 25.0 * DEG

##################################################

LINEAR_KP = 20 #11
LINEAR_KD = 11 #5
LINEAR_MAX_VEL = 1.5

# you should not need to change these
LINEAR_ERR_TOL = 0.01
LINEAR_VEL_TOL = 0.05

# used to determine when to switch from forward/backward to nudge
LINEAR_FINISH_TOL = 1.0 * FT

######################################################################

class PDController:

    # constructor
    def __init__(self,
                 name,
                 dt=0.1,
                 kp=0.5, 
                 kd=0.1, 
                 max_vel=None,
                 err_tol=0.01,
                 vel_tol=0.05):

        self.name = name
        
        self.dt = dt

        self.kp = kp
        self.kd = kd

        self.max_vel = max_vel

        self.err_tol = err_tol

        self.vel_tol = vel_tol

        self.reset()


    # reset the internal state
    def reset(self):

        self.prev_vel = 0.0
        self.prev_err = 0.0

    # update given the error
    def update(self, err=None):

        # ignore zany or non-existent errors 
        if err is None or numpy.isnan(err):
            err = self.prev_err

        # compute desired acceleration
        u = self.kp * err - self.kd * self.prev_vel

        # integrate desired acceleration to get desired velocity
        new_vel = self.prev_vel + u * self.dt

        # clip to max vel if needed
        if self.max_vel is not None:
            new_vel = numpy.clip(new_vel, -self.max_vel, self.max_vel)

        # we are done when error is small enough AND output velocity
        # is small enough
        err_ok = abs(err) < self.err_tol
        vel_ok = abs(new_vel) < self.vel_tol
            
        done = err_ok and vel_ok

        rospy.loginfo('%s err=%f prev_vel=%f new_vel=%f, err_ok=%s, vel_ok=%s',
                      self.name, err, self.prev_vel, new_vel, err_ok, vel_ok)

        # update internal state
        self.prev_vel = new_vel
        self.prev_err = err

        # return commanded velocity and flag
        return new_vel, done

######################################################################

# O(n) greedy segmentation of laser scan points into a set of line segments
def find_segments(pts):

    ##################################################
    # first step is to estimate normal vectors locally
    # using neighboring points in the scan

    # filter window size for central differencing approximation of normal
    k = SEGMENT_WINDOW_SIZE

    p_prev = pts[:-2*k]
    p_mid = pts[k:-k]
    p_next = pts[2*k:] 
    
    # get tangent vectors using central differencing
    n_mid = p_next - p_prev

    # rotate 90 degrees
    n_mid = n_mid[:, ::-1] * [-1, 1]

    # normalize 
    n_mid /= numpy.linalg.norm(n_mid, axis=1).reshape(-1, 1)

    # number of points in current segment
    count = 0

    # raw first-order moments
    b = numpy.zeros(2)

    # raw second-order moments
    A = numpy.zeros((2, 2))

    # centroid of segment
    centroid = None

    # normal of segment
    normal = None

    # starting index of segment
    idx0 = None

    # list of segments
    segments = []

    # for each point we have normal estimates for
    for i, p in enumerate(p_mid):

        # if no current segment, start a new segment at the current point
        if normal is None:
            
            normal = n_mid[i]
            idx0 = i

        # else if this point does not "belong" in the current segment
        elif (abs(numpy.dot(normal, n_mid[i])) < SEGMENT_NORMAL_COS_TOL
              or abs(numpy.dot(normal, p - centroid)) > SEGMENT_NORMAL_DIST_TOL):

            # emit the current segment
            if count >= SEGMENT_MIN_COUNT:
                segments.append((idx0 + k, count, centroid, normal))

            # start a new segment at the current point
            count = 0
            b[:] = 0
            A[:] = 0
            normal = n_mid[i]
            idx0 = i

        # update raw moments
        b += p
        A += numpy.outer(p, p)
        count += 1
            
        # central first-order moments
        centroid = b / count

        # compute central second-order moments to update the normal vector
        if count >= 2:

            mu11 = A[1,0] - centroid[0] * b[1]
            mu20 = A[0,0] - centroid[0] * b[0]
            mu02 = A[1,1] - centroid[1] * b[1]

            U = numpy.array([[mu20, mu11], [mu11, mu02]])

            # compute normal as eigenvector of central 2nd order
            # moment matrix corresponding to smaller eigenvalue
            [w, v] = numpy.linalg.eigh(U)
            normal = v[:, w.argmin()]

    # done with loop, if cur segment is non-trival add it to the list
    if count >= SEGMENT_MIN_COUNT:
        segments.append((idx0 + k, count, centroid, normal))

    # return the detected segments
    return segments

# rotate vector until maximally aligned with positive x axis
def rotate_pos_x(v):

    if abs(v[1]) > abs(v[0]):
        v = numpy.array([-v[1], v[0]])

    if v[0] < 0:
        v = -v
    
    return v

# compute weighted angular error for a list of line segments
# e.g. how far robot would have to turn to be perpendicular to
# the dominant grid direction

def weighted_angular_error(segments):

    total = 0.0
    total_weight = 0.0
    
    for (_, count, _, normal) in segments:

        normal = rotate_pos_x(normal)

        theta = numpy.arctan2(normal[1], normal[0])
        
        weight = count

        total += weight * theta
        total_weight += weight

    return total / total_weight

######################################################################

# Controller for manual control
class Controller:

    # Initializer
    def __init__(self):

        # Setup this node
        rospy.init_node('Controller')

        # Create a TransformListener - we will use it both to fetch
        # odometry readings and to get the transform between the
        # Kinect's depth image and the base frame.
        self.tf_listener = tf.TransformListener()

        # Stash the 2D transformation from depth image to base frame.
        self.base_from_depth = None

        # These member variables will be set by the laser scan
        # callback.
        self.angles = None
        self.ranges = None
        self.points = None

        # For safety
        self.should_stop = False
        self.stop_time = rospy.get_rostime() - rospy.Duration(100.0)

        # Set up maze
        self.maze_commands = []
        self.command_index = -1
        self.maze = None

        self.distance_to_fb_finish = None

        args = rospy.myargv(argv=sys.argv)[1:]

        if args[0] == 'solve':

            rospack = rospkg.RosPack()
            final_project_path = rospack.get_path('final_project')
            maze_file = final_project_path + '/data/' + args[1]

            x0, y0, dir0, x1, y1 = maze.split_command(args[2:])

            m = maze.Maze()
            m.load(maze_file)

            path = m.solve(x0, y0, x1, y1)

            self.maze_commands = maze.path_actions(path, dir0)

        else:

            self.maze_commands = args

        # create our PD controllers

        self.actrl = PDController('angular', DELTA_T, 
                                  ANGULAR_KP, ANGULAR_KD, 
                                  ANGULAR_MAX_VEL, 
                                  ANGULAR_ERR_TOL,
                                  ANGULAR_VEL_TOL)
        
        self.lctrl = PDController('linear', DELTA_T, 
                                  LINEAR_KP, LINEAR_KD,
                                  LINEAR_MAX_VEL, 
                                  LINEAR_ERR_TOL,
                                  LINEAR_VEL_TOL)

        # Let's start out in the initializing state
        self.reset_state('initializing')

        rospy.loginfo('maze_commands: ' + str(self.maze_commands))

        # Create a publisher for commanded velocity
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                                           Twist, queue_size=10)

        # Subscribe to laser scan data
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # Subscribe to bumpers/cliff sensors
        rospy.Subscriber('/mobile_base/sensors/core',
                         SensorState, self.sensor_callback)

        # Call the controll callback at 100 HZ
        rospy.Timer(CONTROL_PERIOD, self.control_callback)


    # Set the current state to the state given.
    #
    # You can pass reset_all=False to disable resetting the
    # self.start_time variable and internal states for the controller.
    #
    # This is useful if you want to switch
    # from one behavior to another (i.e. "turnright" to "straighten")
    # without causing discontinuities in controls
    def reset_state(self, state, reset_all=True):

        self.state = state
        self.start_pose = None

        if reset_all:

            self.start_time = rospy.get_rostime()
            self.labeled_state = state
            
            self.actrl.reset()
            self.lctrl.reset()
            
        if state == 'initializing':
            rospy.loginfo('waiting for TF and laser scan...')

    # Called whenever sensor messages are received.
    def sensor_callback(self, msg):

        if (msg.cliff or msg.bumper):
            self.should_stop = True
            self.stop_time = rospy.get_rostime()
        else:
            self.should_stop = False


    # Set up the transformation from depth camera to base frame.
    # We will need this later to deal with laser scanner.
    def setup_tf(self):

        # We might need to do this more than once because the
        # TransformListener might not be ready yet.
        try:

            ros_xform = self.tf_listener.lookupTransform(
                '/base_footprint', '/camera_depth_frame',
                rospy.Time(0))

        except tf.LookupException:

            return False

        self.base_from_depth = \
            transform2d.transform2d_from_ros_transform(ros_xform)

        print('base_from_depth =', str(self.base_from_depth))

        return True

    # Called whenever we hear from laser scanner. This just sets up
    # the self.angles, self.ranges, and self.points member variables.
    def scan_callback(self, msg):

        # Don't do anything til we have the transform from depth
        # camera to base frame.
        if self.base_from_depth is None:
            if not self.setup_tf():
                return

        # Get # of range returns
        count = len(msg.ranges)

        # Create angle array of size N
        self.angles = (msg.angle_min +
                       numpy.arange(count, dtype=float) * msg.angle_increment)

        # Create numpy array from range returns (note many could be
        # NaN indicating no return for a given angle).
        self.ranges = numpy.array(msg.ranges)

        # Points is a 2xN array of cartesian points in depth camera frame
        pts = self.ranges * numpy.vstack( ( numpy.cos(self.angles),
                                            numpy.sin(self.angles) ) )

        # This converts points from depth camera frame to base frame
        # and reshapes into an Nx2 array so that self.points[i] is the
        # point corresponding to self.angles[i].
        self.points = self.base_from_depth.transform_fwd(pts).T

    # Given a list of desired angles (e.g. [-5*DEG, 5*DEG], look up
    # the indices of the closest valid ranges to those angles. If
    # there is no valid range within the cutoff angular distance,
    # returns None.
    def lookup_angles(self, desired_angles, cutoff=3*DEG):

        # Don't return anything if no data.
        if self.angles is None:
            return None

        # Get indices of all non-NaN ranges
        ok_idx = numpy.nonzero(~numpy.isnan(self.ranges))[0]

        # Check all NaN
        if not len(ok_idx):
            return None

        # Build up array of indices to return
        indices = []

        # For each angle passed in
        for angle in desired_angles:

            # Find the closest index
            angle_err = numpy.abs(angle - self.angles[ok_idx])
            i = angle_err.argmin()

            # If too far away from desired, fail :(
            if angle_err[i] > cutoff:
                return None

            # Append index of closest
            indices.append(ok_idx[i])

        # Return the array we built up
        return indices

    # Look up points at the angles given (see lookup_angles for
    # interpretation of desired_angles, cutoff).
    def points_at_angles(self, desired_angles, cutoff=3*DEG):

        indices = self.lookup_angles(desired_angles, cutoff)
        if indices is None:
            return None

        return self.points[indices]

    # Gets the current pose of the robot w.r.t. odometry frame.
    def get_current_pose(self):

        try:
            ros_xform = self.tf_listener.lookupTransform(
                '/odom', '/base_footprint',
                rospy.Time(0))

        except tf.LookupException:
            return None

        xform2d = transform2d.transform2d_from_ros_transform(ros_xform)

        return xform2d

    def get_heading_error(self):

        if self.points is None:
            rospy.logwarn('no points yet in get_heading_error')
            return None

        ok = ~numpy.isnan(self.points.sum(axis=1))

        points = self.points[ok]

        if not len(points):
            rospy.logwarn('no valid points in get_heading_error')
            return None

        start = rospy.Time.now()

        d = SEGMENT_DECIMATE
        
        segments = find_segments(points[d//2::d])

        elapsed = (rospy.Time.now() - start).to_sec()

        rospy.loginfo('found %d segments in %f seconds',
                      len(segments), elapsed)

        if not len(segments):
            rospy.logwarn('no valid segments in get_heading_error')
            return None

        theta = weighted_angular_error(segments)

        return theta
    
    # Called 100 times per second to control the robot.
    def control_callback(self, timer_event=None):

        # Velocity we will command (modifed below)
        cmd_vel = Twist()

        time_since_stop = (rospy.get_rostime() - self.stop_time).to_sec()

        if self.should_stop or time_since_stop < 1.0:
            self.cmd_vel_pub.publish(cmd_vel)
            return

        # Flag for finished with state
        done = False

        # Get current pose
        cur_pose = self.get_current_pose()

        # Try to get relative pose
        if cur_pose is not None:
            if self.start_pose is None:
                self.start_pose = cur_pose.copy()
            rel_pose = self.start_pose.inverse() * cur_pose

        # Dispatch on state:
        if self.state == 'initializing':

            # Go from initializing to idle once TF is ready and we
            # have our first laser scan.
            if cur_pose is not None and self.angles is not None:
                done = True

        elif self.state == 'straighten':
            
            angular_error = self.get_heading_error()

            cmd_vel.angular.z, done = self.actrl.update(angular_error)

        elif self.state == 'turnleft' or self.state == 'turnright':

            if self.state == 'turnleft':
                goal_theta = numpy.pi/2
            else:
                goal_theta = -numpy.pi/2

            angular_error = goal_theta - rel_pose.theta

            cmd_vel.angular.z, _ = self.actrl.update(angular_error)
            
            if abs(angular_error) < ANGULAR_FINISH_TOL:
                self.reset_state('straighten', False)

        elif self.state == 'nudge':

            points = self.points_at_angles([0*DEG])

            if points is None:
                # No points - missing scan data?
                rospy.logwarn('points was None in nudge state!')

                cmd_vel.linear.x, _ = self.lctrl.update(None)
                cmd_vel.angular.z, _ = self.actrl.update(None)
                
            else:
                
                dist_to_wall = points[0][0]
                
                rounded_dist = numpy.floor(dist_to_wall/CELL_SIZE)*CELL_SIZE + 0.5*CELL_SIZE
                
                linear_error = dist_to_wall - rounded_dist

                cmd_vel.linear.x, done = self.lctrl.update(linear_error)

                angular_error = self.get_heading_error()

                cmd_vel.angular.z, _ = self.actrl.update(angular_error)

        elif self.state == 'forward' or self.state == 'backward':

            extra_advance = 0

            # merge forward/backward commands
            while (self.command_index + extra_advance + 1 < len(self.maze_commands) and
                   self.maze_commands[self.command_index + extra_advance + 1] == self.state):
                extra_advance += 1
            
            if self.state == 'forward':
                goal_x = CELL_SIZE * (1 + extra_advance)
            else:
                goal_x = CELL_SIZE * (1 + extra_advance)

            print('goal_x = ', goal_x, ' extra_advance =', extra_advance)
                
            linear_error = goal_x - rel_pose.x

            cmd_vel.linear.x, _ = self.lctrl.update(linear_error)

            angular_error = self.get_heading_error()

            cmd_vel.angular.z, _ = self.actrl.update(angular_error)
            
            if abs(linear_error) < LINEAR_FINISH_TOL:
                self.command_index += extra_advance
                self.reset_state('nudge', False)

        elif self.state != 'idle':

            # Die if invalid state :(
            rospy.logerr('invalid state {}'.format(self.state))
            rospy.signal_shutdown('invalid state')

        # Publish our velocity command
        self.cmd_vel_pub.publish(cmd_vel)

        if done:
            rospy.loginfo('done with state {}'.format(self.labeled_state))
            if self.command_index + 1 >= len(self.maze_commands):
                rospy.loginfo('all done!')
                rospy.signal_shutdown('all done, quitting!')
            else:
                self.command_index += 1
                self.reset_state(self.maze_commands[self.command_index])

    # Running the controller is just rospy.spin()
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        c = Controller()
        c.run()
    except rospy.ROSInterruptException:
        pass
