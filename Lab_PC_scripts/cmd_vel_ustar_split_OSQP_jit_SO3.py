#!/usr/bin/python3
import rospy, numpy as np
import rospkg
from catkin.find_in_workspaces import find_in_workspaces
from IPython import embed
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Twist, Vector3, Pose, PoseStamped
from tf.transformations import quaternion_from_matrix
from nav_msgs.msg import Path
from functools import partial
from jax.tree_util import register_pytree_node

import time

import jax
from jax import jit
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxopt import OSQP

jax.config.update("jax_enable_x64", True)


# ROS node reference Tianyu: https://github.com/tonylitianyu/golfbot/blob/master/nodes/moving#L63

count_start_point = 0

class ustar_m:

    def __init__(self, xref, xref_vel, scaler_t, ref_range_start, ref_range, vopt_max):
        self.xref = xref
        self.xref_vel = xref_vel
        self.scaler_t = scaler_t
        self.ref_range_start = ref_range_start # from where to start looking forward?
        self.ref_range = ref_range # look how much forward, horizon N
        self.vopt_max = vopt_max # maximum virtual control input
        # /passiveDS/desired_twist
        # /cartesian_impedance_controller/desired_twist

    @jit
    def quat_mult(self, q1, q2):
        q1_s = q1[0]
        q2_s = q2[0]
        q1_v = q1[1:].reshape((-1,1))
        q2_v = q2[1:].reshape((-1,1))
        scalar = q1_s*q2_s - q1_v.T @ q2_v
        skew = jnp.array([[0, -q1[3], q1[2]],
                            [q1[3], 0, -q1[1]],
                            [-q1[2], q1[1], 0]])
        vector = q1_s*q2_v + q2_s*q1_v + skew @ q2_v
        q_result = jnp.concatenate((scalar, vector), axis=0).flatten()
        return q_result

    @jit
    def compute_ustar_m(self, count, p_t, q_t, fp_t, w_t, xref, xref_vel):

        global count_start_point

        dist_ref = jnp.linalg.norm(xref[:, :3] - p_t.reshape((1,-1)), axis=1)
        closest_ind = jnp.argmin(dist_ref)

        chosen_ind = closest_ind + self.ref_range_start

        is_ind_limit = chosen_ind < xref.shape[0]

        loop_ind = 0

        xref_ind = jnp.where(is_ind_limit, chosen_ind, loop_ind)

        # count_start_point = jnp.where(is_ind_limit, count_start_point+1, 0)
        # xref_ind = jnp.where(count_start_point < 10, loop_ind, chosen_ind)

        # xref_ind = i

        xref_t = xref[xref_ind].reshape((-1,1)) # dim x 1

        # xref_t = xref_g

        qref = xref_t[3:]

        scale_q = 1

        scale_p = 1

        Q_opt = jnp.vstack((jnp.hstack((scale_p*jnp.eye(3), jnp.zeros((3,3)))), jnp.hstack((jnp.zeros((3,3)), scale_q*jnp.eye(3)))))
        # print(Q_opt.shape)
        e_q = q_t - qref
        e_q = e_q/jnp.linalg.norm(e_q)
        e_p = p_t - xref_t[:3]
        V = jnp.sum(jnp.square(2*e_p))/4 + jnp.sum(jnp.square(2*e_q))/4

        g_p_x = fp_t

        w_g_x = w_t
        g_w_x = jnp.vstack((0.0, w_g_x))

        g_xref = xref_vel[xref_ind].reshape((-1,1))
        g_p_xref = g_xref[:3]

        w_g_xref = g_xref[3:].reshape((-1,1))
        g_w_xref = jnp.vstack((0.0, w_g_xref))

        # print(g_w_xref.shape)

        quat_mult_q = self.quat_mult(g_w_x[:, 0], q_t[:, 0])
        quat_mult_q_ref = self.quat_mult(g_w_xref[:, 0], qref[:, 0])

        s_p = jnp.vdot(e_p, g_p_x - g_p_xref)
        s_q = jnp.vdot(e_q/2.0, quat_mult_q.reshape((-1,1)) - quat_mult_q_ref.reshape((-1,1)))

        s = s_p + s_q

        Q = jnp.array([[q_t[0, 0], q_t[1, 0], q_t[2, 0], q_t[3, 0]],
                        [-q_t[1, 0], q_t[0,0], q_t[3, 0], -q_t[2, 0]],
                        [-q_t[2, 0], -q_t[3, 0], q_t[0, 0], q_t[1, 0]],
                        [-q_t[3, 0], q_t[2, 0], -q_t[1, 0], q_t[0, 0]]])
        Q2 = Q[:, 1:] # 4 x 3
        Q2_minus = -Q2
        Q2_1 = jnp.vstack((Q2_minus[0, :], -Q2_minus[1:,:]))

        alpha_V = 60

        G_opt = 2*jnp.hstack((e_p.T, (e_q.T/2.0) @ Q2_1))
        h_opt = jnp.array([-alpha_V*V - 2*s])

        qp = OSQP()

        sol = qp.run(params_obj=(Q_opt,jnp.ones(Q_opt.shape[0])), params_ineq=(G_opt, h_opt)).params

        virtual_u = sol.primal.reshape((-1,1))

        vel_cmd_pos = fp_t + virtual_u[:3]
        vel_cmd_rot = w_t + virtual_u[3:]

        k_linear = 1
        k_angular = 1.0

        vel_cmd_pos_final = vel_cmd_pos * self.scaler_t * k_linear
        vel_cmd_rot_final = vel_cmd_rot * self.scaler_t * k_angular

        vel_cmd = jnp.vstack((vel_cmd_pos_final, vel_cmd_rot_final))

        # vel_cmd = jnp.copy(virtual_u)

        vel_cmd_final = jnp.where(count<2, jnp.zeros(vel_cmd.shape), vel_cmd)

        return vel_cmd_final[0], vel_cmd_final[1], vel_cmd_final[2], vel_cmd_final[3], vel_cmd_final[4], vel_cmd_final[5]
    
    def _tree_flatten(self):
        # You might also want to store self.b in either the first group
        # (if it's not hashable) or the second group (if it's hashable)
        return (self.xref, self.xref_vel, self.scaler_t, self.ref_range_start, 
                self.ref_range, self.vopt_max,), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children)

register_pytree_node(ustar_m, ustar_m._tree_flatten, ustar_m._tree_unflatten)

xt = jnp.zeros((3,1))
qt = jnp.array([1.0, 0.0, 0.0, 0.0]).reshape((-1,1))
fxt =  jnp.zeros((3,1))
fxt_rot =  jnp.zeros((3,1))

pub = rospy.Publisher('/cartesian_impedance_controller/desired_twist', Twist, queue_size=10)

def q_from_R(R):
    """ generates quaternion from 3x3 rotation matrix """
    _R = jnp.eye(4)
    _R = _R.at[:3, :3].set(R)
    return quaternion_from_matrix(_R)

def franka_callback(state_msg):
    global xt, qt
    O_T_EE = jnp.array(state_msg.O_T_EE).reshape(4, 4).T
    xt = jnp.array([O_T_EE[0, 3], O_T_EE[1, 3], O_T_EE[2, 3]]).reshape((-1,1))
    quat_ee = q_from_R(O_T_EE[:3, :3]) # x, y, z, w
    qt = jnp.array([quat_ee[3], quat_ee[0], quat_ee[1], quat_ee[2]]).reshape((-1,1))

def NN_callback(NN_msg):
    global fxt
    fxt = jnp.array([NN_msg.x, NN_msg.y, NN_msg.z]).reshape((-1,1))

def NN_callback_rot(NN_msg):
    global fxt_rot
    fxt_rot = jnp.array([NN_msg.x, NN_msg.y, NN_msg.z]).reshape((-1,1))

def pub_desired_vel(linearx, lineary, linearz, angularx, angulary, angularz):
    global pub
    # now = time.time()
    desired_twist = Twist()

    # vel = jnp.array([linearx, lineary, linearz])

    desired_twist.linear.x = linearx
    desired_twist.linear.y = lineary
    desired_twist.linear.z = linearz

    # desired_twist.linear.x = vel[0]
    # desired_twist.linear.y = vel[1]
    # desired_twist.linear.z = vel[2]

    # rospy.loginfo("Time for linear: %f ms", 1000*(time.time() - now))
    # now1 = time.time()
    desired_twist.angular.x = angularx
    desired_twist.angular.y = angulary
    desired_twist.angular.z = angularz

    # rospy.loginfo("Time for angualr: %f ms", 1000*(time.time() - now1))

    pub.publish(desired_twist)

def main():

    global xt, qt, fxt, fxt_rot, count_start_point
    ## Load data

    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')

    ref_file = curr_path + '/config/Data_Trajs/trajectory_ref.npy'
    with open(ref_file, 'rb') as f:
        xref = jnp.load(f)[::3]
        xref_vel = jnp.load(f)[::3]
        scaler_t = jnp.load(f)

    ## Initialization for the class object

    freq = 500
    ref_range_start = 7 # from where to start looking forward?
    ref_range = 5 # look how much forward, horizon N
    vopt_max = 1 # maximum virtual control input
    
    ## Initialize node

    rospy.init_node('cmd_vel_ustar_split_OSQP_jit_SO3', anonymous=True)
    rate = rospy.Rate(freq) # Hz

    rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, 
                                        franka_callback, queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/my_DS_plan/ODE_vel', Vector3, NN_callback,
                                    queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/my_DS_plan/ODE_vel_rot', Vector3, NN_callback_rot,
                                    queue_size=1, tcp_nodelay=True)

    ustar_obj = ustar_m(xref, xref_vel, scaler_t, ref_range_start, ref_range, vopt_max)
    count = 0
    
    while not rospy.is_shutdown():
        # O_T_EE = jnp.array(sub_obj.state_data.O_T_EE).reshape(4, 4).T
        now = time.time()
        # xt = xt.at[0, 1].set(O_T_EE[0, 3])
        # xt = xt.at[1, 1].set(O_T_EE[1, 3])
        # xt = xt.at[2, 1].set(O_T_EE[2, 3])
        # rospy.loginfo("Time for processing node: %f ms", 1000*(time.time() - now))
        # NN_data_t = sub_obj.NN_data
        # fxt = jnp.array([[NN_data_t.x], [NN_data_t.y], [NN_data_t.z]])
        count += 1
        count = jnp.where(count<100, count, 2)
        f1x, f1y, f1z, f1x_rot, f1y_rot, f1z_rot = ustar_obj.compute_ustar_m(count, xt, qt, fxt, fxt_rot, xref, xref_vel)
        # rospy.loginfo("Time for computation: %f ms", 1000*(time.time() - now))
        # rospy.loginfo("Current state[0]: %f ", ustar_obj.xt[0, 1])
        # rospy.loginfo("Time scale: %f /s", scaler_t)
        pub_desired_vel(f1x, f1y, f1z, f1x_rot, f1y_rot, f1z_rot)
        # rospy.loginfo(count_start_point)
        # rospy.loginfo("Current state: %f, %f, %f", x_t[0], x_t[1], x_t[2])
        # rospy.loginfo("Time for computation: %f ms", 1000*(time.time() - now))
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass