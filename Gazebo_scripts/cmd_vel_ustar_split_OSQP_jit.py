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

import time

import jax
from jax import jit
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxopt import OSQP

jax.config.update("jax_enable_x64", True)


# ROS node reference Tianyu: https://github.com/tonylitianyu/golfbot/blob/master/nodes/moving#L63

class ustar_m:

    def __init__(self, xref, xref_vel, scaler_t):
        # self.dim = xref.shape[-1] # dimension of the state space
        self.f1x = jnp.zeros((3,1))
        self.xref_g_i = 0 # Initialize index of purely time parameterized reference trajectory
        self.xref = xref
        self.xref_vel = xref_vel
        self.scaler_t = scaler_t
        self.f1x_start = False
        self.dim = int(self.xref.shape[-1]/2) ## 
        self.ref_range_start = 7 # from where to start looking forward?
        self.ref_range = 5 # look how much forward, horizon N
        self.vopt_max = 1 # maximum virtual control input
        self.xt = jnp.zeros((3,1))
        self.xt_p = jnp.zeros((3,1))
        self.fxt =  jnp.zeros((3,1))
        self.fxt_d =  jnp.zeros((3,1))
        self.state_sub = rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, 
                                          self.franka_callback, queue_size=1, tcp_nodelay=True)
        self.NN_sub = rospy.Subscriber('/my_DS_plan/ODE_vel', Vector3, self.NN_callback,
                                        queue_size=1, tcp_nodelay=True)
        self.NN_sub_d = rospy.Subscriber('/my_DS_plan/ODE_vel_d', Vector3, self.NN_callback_d,
                                        queue_size=1, tcp_nodelay=True)
        self.pub = rospy.Publisher('/cartesian_impedance_controller/desired_twist', Twist, queue_size=10)
    
    def franka_callback(self, state_msg):
        self.xt_p = self.xt
        O_T_EE = jnp.array(state_msg.O_T_EE).reshape(4, 4).T
        # self.xt = self.xt.at[0, 1].set(O_T_EE[0, 3])
        # self.xt = self.xt.at[1, 1].set(O_T_EE[1, 3])
        # self.xt = self.xt.at[2, 1].set(O_T_EE[2, 3])
        self.xt = jnp.array(O_T_EE[:3,3]).reshape((-1,1))

    def NN_callback(self, NN_msg):
        # self.fxt = self.fxt.at[0, 1].set(NN_msg.x)
        # self.fxt = self.fxt.at[1, 1].set(NN_msg.y)
        # self.fxt = self.fxt.at[2, 1].set(NN_msg.z)
        self.fxt = jnp.array([NN_msg.x, NN_msg.y, NN_msg.z]).reshape((-1,1))

    def NN_callback_d(self, NN_msg):
        # self.fxt = self.fxt.at[0, 1].set(NN_msg.x)
        # self.fxt = self.fxt.at[1, 1].set(NN_msg.y)
        # self.fxt = self.fxt.at[2, 1].set(NN_msg.z)
        self.fxt_d = jnp.array([NN_msg.x, NN_msg.y, NN_msg.z]).reshape((-1,1))

    @staticmethod
    def q_from_R(R):
        """ generates quaternion from 3x3 rotation matrix """
        _R = jnp.eye(4)
        _R = _R.at[:3, :3].set(R)
        return jnp.array(quaternion_from_matrix(_R))

    @partial(jit, static_argnums=(0,))
    def compute_ustar_m(self, state_data, state_data_p, NN_vel, NN_vel_d):

        # quat_ee = self.q_from_R(O_T_EE[:3, :3])

        x_t = state_data
        x_t_p = state_data_p

        x_d = x_t.reshape((-1,1)) - x_t_p.reshape((-1,1))

        xref_d = jnp.vstack((jnp.diff(self.xref[:, :self.dim], axis=0),
                             jnp.zeros((1, self.dim))))

        vopt_bound = self.vopt_max * (1 / self.scaler_t)

        ## scale down since demos are fast
        # Mixing task := 0.1, wiping big board := 0.5, Spiral big board := 0.5,
        # Wiping big board 2 half := 0.1
        # Spiral 2 := 0.1
        # Cap := 0.1
        # Massage := 1

        k_fx = 0.3
        fx = k_fx*NN_vel
        fx_d = NN_vel_d

        r = 0.05  # CBF radius LR

        ## time parameterized target trajectory

        self.xref_g_i = jnp.where(self.xref_g_i >= self.xref.shape[0], 
                  self.xref.shape[0] - 1,
                  self.xref.shape[0] + 1)

        xref_i = self.xref[self.xref_g_i, :]

        dist_to_goal = jnp.linalg.norm(self.xref[-1, :self.dim] - x_t)
        
        
        ## command to go to closest xref => xref_t

        alpha = 0.0 # weightage for cosine distance
        
        # self.f1x = jnp.where(self.f1x_start, self.f1x, fx)
        # self.f1x_start = jnp.where(self.f1x_start, True, True)
        # f1x_u = self.f1x / jnp.linalg.norm(self.f1x)
        # xref_vel_u = self.xref_vel / jnp.linalg.norm(self.xref_vel, axis=1).reshape((-1, 1))
        # cos_angle = jnp.clip(jnp.dot(xref_vel_u, f1x_u), -1, 1)
        dist_ref = jnp.linalg.norm(self.xref[:, :self.dim] - x_t.reshape((1,-1)), axis=1)
        dist_ref_u = (1-alpha) * dist_ref / 1

        dot_all = jnp.dot(xref_d, x_d)

        cosine_dist = alpha * jnp.abs(1 - (dot_all[:, 0] / (jnp.linalg.norm(xref_d, axis=1) * jnp.linalg.norm(x_d))))
        
        # rospy.loginfo("cosine dist shape: %f", cosine_dist.shape[0])

        dist_ref_combine = jnp.add(cosine_dist, dist_ref_u)

        # rospy.loginfo("dist ref combine shape: %f", dist_ref_combine.shape[1])
        
        # closest_ind = jnp.argmin((1-alpha) * dist_ref_u + alpha * cosine_dist)
        closest_ind = jnp.argmin(dist_ref_u)
        # closest_ind = jnp.argmin(dist_ref_combine)

        ## looking forward

        # chosen_ind_all = jnp.arange(closest_ind + self.ref_range_start, closest_ind + self.ref_range_start + self.ref_range, 1, dtype=int)
        # chosen_ind = jnp.clip(chosen_ind_all, 0, self.xref.shape[0]-1)

        chosen_ind = closest_ind + self.ref_range_start

        is_ind_limit = chosen_ind < self.xref.shape[0]

        # loop_ind = 45 # Massage loop
        # loop_ind = 20 # Towel loop
        loop_ind = self.xref.shape[0]-1

        xref_ind = jnp.where(is_ind_limit, chosen_ind, loop_ind)
            
        xref_t = self.xref[xref_ind, :self.dim].reshape((-1,1))
        fxref = self.xref_vel[xref_ind, :self.dim].reshape((-1,1))       

        grad_V = 2*(x_t - xref_t)
        V = jnp.sum(jnp.square(grad_V))/4
        alpha_V = 40

        ## OSQP solver solution

        Q = jnp.eye(fx.shape[0])
        G = jnp.vstack((grad_V.T, jnp.eye(fx.shape[0]), -jnp.eye(fx.shape[0])))
        h = jnp.hstack((jnp.array([-alpha_V * V  - jnp.sum(grad_V * (fx - fxref))]),
                       jnp.repeat(vopt_bound, repeats=2*fx.shape[0], axis=0)))

        qp = OSQP()
        sol = qp.run(params_obj=(Q,jnp.zeros(fx.shape[0])), params_ineq=(G, h)).params
        
        ustar = sol.primal.reshape((-1,1))

        ## Closed form solution

        # ineq = jnp.sum(grad_V * (fx - fxref)) + alpha_h * V

        # dist_to_ref = jnp.linalg.norm(x_t - xref_t)

        # ustar_closed = -((ineq)/(4 * V)) * grad_V
        # is_ineq = ineq <=0
        # is_dist_to_ref_small = dist_to_ref < 1e-3
        # ustar = jnp.where(jnp.logical_or(is_ineq, is_dist_to_ref_small),
        #             jnp.zeros((3,1)), ustar_closed)

        # vopt_chosen = jax.lax.cond(dist_to_goal < 2*r, 
        #              self.near_to_goal,
        #              self.far_from_goal,
        #              x_t, fx, r)        

        f1x = jnp.add(fx, ustar)
        
        k = 1

        vopt_chosen = f1x * self.scaler_t * k

        return tuple(vopt_chosen)
    
    def pub_desired_vel(self, linearx, lineary, linearz):
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
        desired_twist.angular.x = 0
        desired_twist.angular.y = 0
        desired_twist.angular.z = 0

        # rospy.loginfo("Time for angualr: %f ms", 1000*(time.time() - now1))

        self.pub.publish(desired_twist)


def main():
    ## Load data

    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')

    ref_file = curr_path + '/config/Data_Trajs/trajectory_ref.npy'
    with open(ref_file, 'rb') as f:
        xref = jnp.load(f)
        xref_vel = jnp.load(f)
        scaler_t = jnp.load(f)
    
    ## Initialize node

    rospy.init_node('cmd_vel_ustar_split_OSQP_jit', anonymous=True)
    rate = rospy.Rate(1000) # Hz

    ## Publisher

    # pub = rospy.Publisher('/cartesian_impedance_controller/desired_twist', Twist, queue_size=10)
    
    ## Initialize object

    # sub_obj = sub_data() # Subscribers

    ustar_obj = ustar_m(xref, xref_vel, scaler_t)
    
    while not rospy.is_shutdown():
        # O_T_EE = jnp.array(sub_obj.state_data.O_T_EE).reshape(4, 4).T
        now = time.time()
        # xt = xt.at[0, 1].set(O_T_EE[0, 3])
        # xt = xt.at[1, 1].set(O_T_EE[1, 3])
        # xt = xt.at[2, 1].set(O_T_EE[2, 3])
        # rospy.loginfo("Time for processing node: %f ms", 1000*(time.time() - now))
        # NN_data_t = sub_obj.NN_data
        # fxt = jnp.array([[NN_data_t.x], [NN_data_t.y], [NN_data_t.z]])
        f1x, f1y, f1z = ustar_obj.compute_ustar_m(ustar_obj.xt, ustar_obj.xt_p,
                                                  ustar_obj.fxt, ustar_obj.fxt_d)
        # rospy.loginfo("Time for computation: %f ms", 1000*(time.time() - now))
        # rospy.loginfo("Current state[0]: %f ", ustar_obj.xt[0, 1])
        # rospy.loginfo("Time scale: %f /s", scaler_t)
        ustar_obj.pub_desired_vel(f1x, f1y, f1z)
        # rospy.loginfo("Current state: %f, %f, %f", x_t[0], x_t[1], x_t[2])
        rospy.loginfo("Time for computation: %f ms", 1000*(time.time() - now))
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass