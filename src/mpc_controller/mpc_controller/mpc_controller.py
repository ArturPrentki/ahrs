import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time

import numpy as np
import pinocchio as pin
import crocoddyl

# Import your custom messages
from wbc_control_msgs import MpcTrajectory, MpcStateStep, FootState
import numpy as np
import pinocchio as pin
import crocoddyl

def create_phase_model(rmodel, feet_in_contact, x_default, dt=0.02):
    """
    rmodel: pinocchio.Model (must be FreeFlyer)
    feet_in_contact: list of strings (e.g., ['FL_foot', 'RR_foot'])
    x_default: nominal state [q(19), v(18)]
    dt: integration time step
    """
    
    # 1. State and Actuation
    state = crocoddyl.StateMultibody(rmodel)
    # Floating base is unactuated, so nu = nv - 6 (12 for a quadruped)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu
    
    # 2. Contact Model Collection
    contactModel = crocoddyl.ContactModelMultiple(state, nu)
    
    for foot_name in feet_in_contact:
        frame_id = rmodel.getFrameId(foot_name)
        # 3D contact (Point contact, standard for quadrupeds)
        # Assumes contact is at z=0 in the world frame
        contact_ref = np.array([0., 0., 0.]) 
        contact3d = crocoddyl.ContactModel3D(
            state, 
            frame_id, 
            contact_ref, 
            pin.LOCAL_WORLD_ALIGNED, 
            nu
        )
        contactModel.addContact(foot_name + "_contact", contact3d)

    # 3. Cost Model Collection
    costModel = crocoddyl.CostModelSum(state, nu)
    
    # --- State Regularization Cost ---
    # Encourages the robot to stay near its nominal posture
    stateResidual = crocoddyl.ResidualModelState(state, x_default, nu)
    
    # Weigh floating base differently than joints and velocities
    stateWeights = np.concatenate([
        np.array([0, 0, 0]),       # Base position (usually unpenalized to allow movement)
        np.array([10, 10, 10]),    # Base orientation
        np.full(12, 1.0),          # Joint positions
        np.full(rmodel.nv, 0.1)    # All velocities
    ])
    stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights)
    stateCost = crocoddyl.CostModelResidual(state, stateActivation, stateResidual)
    costModel.addCost("state_reg", stateCost, 1.0)
    
    # --- Control Regularization Cost ---
    # Minimizes torque usage
    ctrlResidual = crocoddyl.ResidualModelControl(state, nu)
    ctrlCost = crocoddyl.CostModelResidual(state, ctrlResidual)
    costModel.addCost("ctrl_reg", ctrlCost, 0.01)
    
    # --- Friction Cone Cost (For stance feet) ---
    mu = 0.5  # Friction coefficient
    Rsurf = np.eye(3) # Surface normal (flat ground)
    cone = crocoddyl.FrictionCone(Rsurf, mu, 4, False)
    coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(cone.lb, cone.ub)
    )
    
    for foot_name in feet_in_contact:
        frame_id = rmodel.getFrameId(foot_name)
        coneResidual = crocoddyl.ResidualModelContactFrictionCone(
            state, frame_id, cone, nu
        )
        coneCost = crocoddyl.CostModelResidual(state, coneActivation, coneResidual)
        # Notice we name it dynamically so Crocoddyl can track it
        costModel.addCost(foot_name + "_friction", coneCost, 0.1)
        
    # --- Swing Foot Tracking (For swing feet) ---
    # If a foot is NOT in contact, we usually want it to track a swing trajectory
    all_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    swing_feet = [f for f in all_feet if f not in feet_in_contact]
    
    for foot_name in swing_feet:
        frame_id = rmodel.getFrameId(foot_name)
        # In a real MPC, you update this reference dynamically in the control loop
        target_pos = np.array([0.0, 0.0, 0.1]) # e.g., 10cm ground clearance
        
        footResidual = crocoddyl.ResidualModelFrameTranslation(
            state, frame_id, target_pos, nu
        )
        footCost = crocoddyl.CostModelResidual(state, footResidual)
        costModel.addCost(foot_name + "_swing", footCost, 100.0) # High weight to ensure clearance

    # 4. Assemble the Differential Model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, 
        actuation, 
        contactModel, 
        costModel, 
        0.0,  # inv_damping (regularization for the contact solver)
        True  # enable_force (tells Crocoddyl to calculate contact forces for the WBC)
    )
    
    # 5. Integrate to create the discrete-time Action Model
    action_model = crocoddyl.IntegratedActionModelEuler(dmodel, dt)
    return action_model

class CrocoddylMPCNode(Node):
    def __init__(self):
        super().__init__('crocoddyl_mpc_node')
        
        # 1. Initialize Pinocchio model
        # Replace with the actual path to your URDF
        self.rmodel = pin.buildModelFromUrdf("path/to/quadruped.urdf", pin.JointModelFreeFlyer())
        self.rdata = self.rmodel.createData()
        

        # 2. Setup ROS 2 Publishers and Subscribers
        self.state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10
        )
        
        # Publish the custom MpcTrajectory instead of Float64MultiArray
        self.trajectory_pub = self.create_publisher(
            MpcTrajectory,
            '/mpc_trajectory',
            10
        )
        
        # 3. Create a timer for the control loop (50 Hz)
        self.dt = 0.02  # seconds
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        # Dummy solver and state placeholders
        self.current_state = None
        problem = None
        self.solver = self.solver = crocoddyl.SolverFDDP(problem)
        self.mpc_horizon = 10 # Number of steps to send to WBC
        
        self.get_logger().info("Crocoddyl MPC Node Initialized")

    def state_callback(self, msg):
        # In a real quadruped, you need base pose/twist from an estimator (e.g., IMU + Kinematics)
        # combined with msg.position (joints) and msg.velocity (joints).
        # This assumes msg.position contains [base_pos(3), base_quat(4), joints(12)]
        
        if len(msg.position) + len(msg.velocity) == self.rmodel.nq + self.rmodel.nv:
            self.current_state = np.concatenate([msg.position, msg.velocity])
        else:
            self.get_logger().debug("Waiting for full state vector...")

    def control_loop(self):
        if self.current_state is None:
            return
            
        # 1. Solve the OCP
        init_xs = [self.current_state] * (self.problem.T + 1)
        init_us = [np.zeros(self.problem.nu)] * self.problem.T
        self.solver.solve(init_xs, init_us, max_iter=1, is_feasible=False)
        
        # 2. Create the Trajectory Message
        traj_msg = MpcTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.header.frame_id = "odom" # Or "world"
        
        # Assuming solver.xs contains the converged state sequence
        # For this stub, we create dummy xs and us to demonstrate the packing logic
        action_model = create_phase_model(self.rmodel,[],)
        dummy_xs = [np.zeros(self.rmodel.nq + self.rmodel.nv) for _ in range(self.mpc_horizon)]
        dummy_us = [np.zeros(12) for _ in range(self.mpc_horizon)] # Assuming 12 joint torques
        
        # Iterate over the horizon to pack the message
        for i in range(self.mpc_horizon):
            step_msg = MpcStateStep()
            
            # Time offset for the interpolator
            time_offset_sec = float(i * self.dt)
            step_msg.time_offset.sec = int(time_offset_sec)
            step_msg.time_offset.nanosec = int((time_offset_sec - int(time_offset_sec)) * 1e9)
            
            # Extract q and v from Crocoddyl state vector x
            x = dummy_xs[i] # In reality: x = self.solver.xs[i]
            q = x[:self.rmodel.nq]
            v = x[self.rmodel.nq:]
            
            # Convert numpy arrays to lists for ROS 2 message
            step_msg.q = q.tolist()
            step_msg.v = v.tolist()
            
            # Accelerations (often computed via finite differences if not explicitly in the state)
            if i < len(dummy_xs) - 1:
                v_next = dummy_xs[i+1][self.rmodel.nq:]
                a = (v_next - v) / self.dt
            else:
                a = np.zeros(self.rmodel.nv)
            step_msg.a = a.tolist()
            
            # Feedforward torques (optimal u)
            u = dummy_us[i] if i < len(dummy_us) else np.zeros(12) # In reality: u = self.solver.us[i]
            step_msg.tau_ff = u.tolist()
            
            # Foot States (Requires extracting contact forces from Crocoddyl's running data)
            # Example structure for 4 feet:
            foot_names = ["FL", "FR", "RL", "RR"]
            for foot in foot_names:
                f_msg = FootState()
                f_msg.name = foot
                
                # In a real setup, extract these from solver.problem.runningDatas[i]
                # e.g., force = data.differential.multibody.contacts.contacts[foot+'_contact'].f.linear
                f_msg.is_contact = True # Determine from your contact schedule
                f_msg.contact_force.x = 0.0
                f_msg.contact_force.y = 0.0
                f_msg.contact_force.z = 0.0
                
                # Kinematics (positions/velocities) can be computed using pin.updateFramePlacements
                f_msg.position.x = 0.0
                
                step_msg.feet.append(f_msg)
                
            traj_msg.steps.append(step_msg)
            
        # 3. Publish to the C++ WBC
        self.trajectory_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CrocoddylMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()