# CLF-CBF-NODE

This repo contains the code base for the below project, which is accepted to appear in [ICRA 2024](https://2024.ieee-icra.org/).

## Paper
Farhad Nawaz, Tianyu Li, Nikolai Matni, Nadia Figueroa, "Learning Complex Motion Plans using Neural ODEs with Safety and Stability Guarantees" _arXiv:2308.00186_, 2023. (available at https://arxiv.org/abs/2308.00186). 

**TL;DR**: We learn the motions of wiping and stirring tasks from 3 demonstrations, and generate safe, reactive plans online at 1 KHz that converge to a target trajectory. Project webpage: https://sites.google.com/view/lfd-neural-ode/home

<table>
  <tr>
    <td>Teaching</td>
     <td>Nominal task</td>
     <td>Disturbance</td>
     <td>Obstacles</td>
  </tr>
  <tr>
    <td><img src="Exp_GIF/gif_randy_teaching_AdobeExpress.gif"></td>
    <td><img src="Exp_GIF/gif_randy_AdobeExpress.gif"></td>
    <td><img src="Exp_GIF/gif_randy_dist_1_AdobeExpress.gif" ></td>
    <td><img src="Exp_GIF/gif_2_pixl_AdobeExpress_obst.gif"></td>
  </tr>
 </table>

## Dataset

The $\texttt{Dataset}$ folder contains the following datasets:
1. **2D_drawing**: our dataset obtained by drawing trajectories on a 2D window
2. **IROS_dataset**: 2D periodic trajetories of the letters 'I, R, O' and 'S' from https://github.com/robotgradient/iflow
3. **Franka_demos_Full_pose**: our full pose (position in $\mathbb{R}^3$ + orientation in $\mathcal{SO}(3)$ ) demonstrations from the Franka robot arm
3. **clfd_data**: Full pose demonstrations from the clfd data set: https://github.com/sayantanauddy/clfd

## Learn NODE and implement CLF-CBF-NODE 

The $\texttt{Notebooks}$ folder contains jupyter notebooks for learning a Neural ODE (NODE) model and implementing the CLF-CBF QP with the NODE using a simple single integrator model for the different datasets. It also contains visualisations of the learnt vector field and animation of the implemented trajectory.

## Implement on the Franka robot arm

We give instructions and scripts to implement our CLF-CBF-NODE approach on the Franka robot arm. We tested all our experiments on Ubuntu 20.04.6 LTS, ROS Noetic. 

### Installation

* Install [libfranka](https://github.com/frankaemika/libfranka) by following instructions from [here](https://frankaemika.github.io/docs/installation_linux.html#building-from-source). Install libfranka outside your catkin workspace (e.g. ``catkin_ws``). We recommend to build from source. 

* Install [franka_ros](https://github.com/frankaemika/franka_ros), by following instructions form [here](https://frankaemika.github.io/docs/installation_linux.html#building-the-ros-packages).

* Setup and verify [realtime kernel](https://github.com/penn-figueroa-lab/lab_wiki/wiki/Real-Time-Kernel-Patch-in-Ubuntu) in your PC, by following instructions from [here](https://github.com/penn-figueroa-lab/lab_wiki/wiki/Franka#pc-setup). This is not necessary for gazebo simulation.

* Follow these [installation instructions](https://github.com/farhadnawaz/franka_interactive_controllers.git) to install ``franka_interactive_controllers`` package, which contains the necessary controllers, NODE models and demonstrated trajectories of different tasks. The NODE models and trajectories are present in ``/franka_interactive_controllers/config/``, which have to changed in the respective python scripts for different tasks.

* ``pip install -r requirements.txt`` to install necessary python packages.

### Catkin make

* Build your catkin workspace.

```
cd <your-catkin-workspace>
catkin_make -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=/home/<path-to-libfranka>/libfranka/build
source devel/setup.bash
```

### Test real robot

* Start the robot by following steps 1-8 from [here](https://github.com/penn-figueroa-lab/lab_wiki/wiki/Franka#using-franka).

* Check if ``libfranka`` is working: from the terminal, go to ``/<path-to-libfranka>/libfranka/build/examples/`` and run one of the examples. Typically, ``<path-to-libfranka>`` is your home directory. If you encounter any problem, please check the libfranka installation again. The ``robot_ip:=172.16.0.2`` might be different for you.

    * To perform a communication test (robot will move to an initial configuration): ``./communication_test 172.16.0.2``.
    * To move the elbow while keeping a static end-effector pose: ``./generate_elbow_motion 172.16.0.2``.
    * To move the end-effector in a circular motion with joint impedance controller: ``./joint_impedance_control 172.16.0.2``.

* Check a controller on the real robot.

```
roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=172.16.0.2
```

### Experiment setup

* Move all the scripts from ``<path-to-scripts>`` to ``/franka_interactive_controllers/scripts/``, where ``<path-to-scripts>``:=``/Gazebo_scripts/`` for gazebo simulation, or ``/Lab_PC_scripts/`` for real robot implementation. 

* For gazebo, check the launch file ``/franka_interactive_controllers/launch/simulate_panda_gazebo.launch`` for the initial joint configuration of the task you want to run.

* In the ``/franka_interactive_controllers/launch/NODE_model.launch`` file, use the relevant ROS node ``NODE_model_vel_*`` where ``*`` is (i) ``jit`` for 3D Neural ODE, or (ii) ``SO3`` for full pose Neural ODE.

* Change the NODE ``.eqx`` models and data trajectories (``.npy`` files) in the scripts --- ``NODE_model_vel_*.py``, ``my_visuals.py`` and ``cmd_vel_ustar_split_OSQP_**.py`` --- that are present in ``/franka_interactive_controllers/scripts/``, where ``**`` can be 

    (i)``jit`` for 3D CLF-NODE (disturbance rejection)

    (ii)``jit_SO3`` for full pose CLF-NODE (disturbance rejection), or
    
    (iii)``obstacle`` for 3D CLF-CBF-NODE (obstacle avoidance+disturbance rejection)

### Running the experiment

* In the first terminal, run 

```
roslaunch franka_interactive_controllers <launch-file> controller:=passiveDS_impedance
```

where, ``launch-file``:=``simulate_panda_gazebo.launch`` for gazebo simulation, or ``franka_interactive_bringup.launch`` for real robot implementation. We can also change the ``controller`` from ``passiveDS_impedance`` to ``cartesian_twist_impedance``, etc.

* In the second terminal, run 

```
roslaunch franka_interactive_controllers NODE_model.launch
```

* In the third terminal, run 
```
rosrun franka_interactive_controllers cmd_vel_ustar_split_OSQP_**.py
```
