#include <torch/extension.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include "whole_body_controller/quad_prog.hpp"
#include "whole_body_controller/wbc_math.hpp"

#include <pinocchio/parsers/urdf.hpp>
class BatchedWBC {
public:
int num_envs_;
std::vector<std::unique_ptr<wbc::QuadProg>> solvers_;

// We need to store gains here instead of ROS parameters
Eigen::MatrixXd Kp_base_;
Eigen::MatrixXd Kd_base_;

BatchedWBC(int num_envs, const std::string& urdf_string) : num_envs_(num_envs) {
    solvers_.resize(num_envs);
    
    // Initialize gains (replace with your actual sizes/values)
    Kp_base_ << 40.0, 40.0, 50.0, 40.0, 40.0, 40.0; // Proportional
    Kd_base_ << 8.0, 8.0, 10.0, 8.0, 8.0, 8.0;      // Derivative

    Eigen::Matrix<double, 12,1> q_nom;
    q_nom << 0.1,  0.8, -1.5, // FL (hip, thigh, calf)
            -0.1,  0.8, -1.5, // FR 
             0.1,  1.0, -1.5, // RL
            -0.1,  1.0, -1.5; // RR

    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(urdf_string,
                                        pinocchio::JointModelFreeFlyer(), model);
    std::vector<std::string> foot_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};

    for(int i = 0; i < num_envs; i++) {
        solvers_[i] = std::make_unique<wbc::QuadProg>();
        solvers_[i]->init(model, "base", foot_names , q_nom);
    }
}

// Tensors should be shape [num_envs, N], on CPU, contiguous, and ideally float64 (double)
torch::Tensor solve(
    torch::Tensor observations, // this is pos,quat, lv,av, joint(pos,vel)
    torch::Tensor actions,      // this is pose(Z, rX, rY), foot_posx4, contactsx4
    torch::Tensor commands,     // this is base vel(X, Y, rZ)
    double dt                    // e.g., 0.002
) {
    // Ensure tensors are on CPU and contiguous for fast C++ pointer access
    observations = observations.contiguous().cpu().to(torch::kFloat64);
    actions = actions.contiguous().cpu().to(torch::kFloat64);
    commands = commands.contiguous().cpu().to(torch::kFloat64);
    auto ret_device = observations.device();
    // Output tensor for torques
    auto torques_out = torch::zeros({num_envs_, 12}, torch::kFloat64);
    double* observation_ptr = observations.data_ptr<double>();
    double* action_ptr = actions.data_ptr<double>();
    double* command_ptr = commands.data_ptr<double>();
    double* tau_out_ptr = torques_out.data_ptr<double>();
    constexpr int obs_stride = 41;
    constexpr int act_stride = 19;
    constexpr int cmd_stride = 3;
    pybind11::gil_scoped_release release;
    #pragma omp parallel for
    for (int i = 0; i < num_envs_; i++) {
        double* o = observation_ptr + (i * obs_stride);
        double* a = action_ptr + (i * act_stride);
        double* c = command_ptr + (i * cmd_stride);

        Eigen::Vector3d pos_eigen(o[0], o[1], o[2]);
        Eigen::Quaterniond quat_eigen(o[3], o[4], o[5], o[6]); 
        pinocchio::SE3 T_act(quat_eigen.toRotationMatrix(), pos_eigen);

        Eigen::Vector3d lin_vel(o[7], o[8], o[9]);
        Eigen::Vector3d ang_vel(o[10], o[11], o[12]);
        pinocchio::Motion V_act(lin_vel, ang_vel);

        double base_z_target = a[0];
        double roll_target   = a[1];
        double pitch_target  = a[2];

        auto heading_to_world = wbc::WbcMath::construct_heading_to_world_transform(T_act);

        pinocchio::SE3 T_ref = pinocchio::SE3::Identity();
        T_ref.translation() = Eigen::Vector3d(pos_eigen.x(), pos_eigen.y(), base_z_target);

        Eigen::AngleAxisd rollAngle(roll_target, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch_target, Eigen::Vector3d::UnitY());
        T_ref.rotation() = heading_to_world.rotation() * (pitchAngle * rollAngle).toRotationMatrix();

        // FIX: Properly separate (vx, vy) for linear and (wz) for angular velocity
        Eigen::Vector3d v_ref_linear_local(c[0], c[1], 0.0);
        Eigen::Vector3d v_ref_angular_local(0.0, 0.0, c[2]);
        
        Eigen::Vector3d v_ref_linear_world = heading_to_world.rotation() * v_ref_linear_local;
        Eigen::Vector3d v_ref_angular_world = heading_to_world.rotation() * v_ref_angular_local;
        pinocchio::Motion V_ref(v_ref_linear_world, v_ref_angular_world);

        pinocchio::Motion A_ref = pinocchio::Motion::Zero();

        std::array<wbc::FootRef, 4> foot_refs;
        std::array<wbc::LegJointState, 4> joint_states;
        
        for (int leg = 0; leg < 4; ++leg) {
            int action_foot_offset = 3 + (leg * 3);
            int observation_foot_offset = 13 + (leg * 7);
            
            Eigen::Vector3d foot_pos_local(a[action_foot_offset], a[action_foot_offset + 1], a[action_foot_offset + 2]);
            foot_refs[leg].pos = heading_to_world.act(foot_pos_local);
            foot_refs[leg].vel = Eigen::Vector3d::Zero();
            foot_refs[leg].acc = Eigen::Vector3d::Zero();
            bool is_contact = (a[15 + leg] > 0.5);

            foot_refs[leg].desired_contact = is_contact ? 1.0 : 0.0;
            foot_refs[leg].phase = is_contact ? wbc::FootPhase::STANCE : wbc::FootPhase::SWING;

            joint_states[leg].pos << o[observation_foot_offset], o[observation_foot_offset + 1], o[observation_foot_offset + 2];
            joint_states[leg].vel << o[observation_foot_offset + 3], o[observation_foot_offset + 4], o[observation_foot_offset + 5];
            foot_refs[leg].actual_contact = o[observation_foot_offset + 6] > 0.5;
        }
        solvers_[i]->update_state(T_act, V_act, joint_states);
        solvers_[i]->update_dynamics();

        Eigen::Matrix<double, 6, 1> a_cmd = wbc::WbcMath::compute_base_acceleration(
            T_act, V_act, T_ref, V_ref, A_ref, Kp_base_, Kd_base_);

        Eigen::Matrix<double, 18, 1> q_ddot_opt = Eigen::Matrix<double, 18, 1>::Zero();
        Eigen::Matrix<double, 12, 1> F_opt = Eigen::Matrix<double, 12, 1>::Zero();
        
        bool success = solvers_[i]->solve(a_cmd, foot_refs, q_ddot_opt, F_opt);

        Eigen::Matrix<double, 12, 1> tau_cmd;
        if (success) {
            tau_cmd = wbc::WbcMath::compute_inverse_dynamics(
                solvers_[i]->get_mass_matrix(), 
                solvers_[i]->get_nle(),
                solvers_[i]->get_contact_jacobian(), 
                q_ddot_opt, F_opt);
        } else {
            for (int j = 0; j < 12; j++) {
                tau_cmd(j) = -2.0 * o[13 + (j/3)*7 + 3 + (j%3)]; 
            }
        }

        for(int j = 0; j < 12; j++) {
            tau_out_ptr[i * 12 + j] = tau_cmd(j);
        }
    }

    // Return torques, ensuring they match the device of the input tensors
    return torques_out.to(torch::kFloat32).to(ret_device); 
}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BatchedWBC>(m, "BatchedWBC")
        .def(py::init<int, const std::string&>(), 
             py::arg("num_envs"), py::arg("urdf_string"))
        .def("solve", &BatchedWBC::solve, 
             py::arg("observations"), 
             py::arg("actions"), 
             py::arg("commands"), 
             py::arg("dt"));
}
