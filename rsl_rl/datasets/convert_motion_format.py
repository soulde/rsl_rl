"""Convert motion data from robot_lab format to BeyondMimic format.

robot_lab format keys:
    fps, dof_names, body_names, dof_positions, dof_velocities,
    body_positions, body_rotations, body_linear_velocities, body_angular_velocities

BeyondMimic format keys:
    fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, body_names
"""

import argparse
import numpy as np


def convert_robotlab_to_beyondmimic(input_file: str, output_file: str):
    """Convert robot_lab NPZ format to BeyondMimic format."""
    print(f"Loading: {input_file}")
    data = np.load(input_file, allow_pickle=True)

    # Check if already in BeyondMimic format
    if "joint_pos" in data:
        print("Already in BeyondMimic format, copying...")
        output_data = {k: data[k] for k in data.keys()}
    elif "dof_positions" in data:
        print("Converting from robot_lab format...")
        output_data = {
            "fps": data["fps"],
            "joint_pos": data["dof_positions"],
            "joint_vel": data["dof_velocities"],
            "body_pos_w": data["body_positions"],
            "body_quat_w": data["body_rotations"],
            "body_lin_vel_w": data["body_linear_velocities"],
            "body_ang_vel_w": data["body_angular_velocities"],
        }
        # Preserve body_names if available
        if "body_names" in data:
            output_data["body_names"] = data["body_names"]
        if "dof_names" in data:
            output_data["joint_names"] = data["dof_names"]
    else:
        raise ValueError(f"Unknown format in {input_file}")

    # Print summary
    print(f"Output: {output_file}")
    print(f"  fps: {output_data['fps']}")
    print(f"  joint_pos: {output_data['joint_pos'].shape}")
    print(f"  joint_vel: {output_data['joint_vel'].shape}")
    print(f"  body_pos_w: {output_data['body_pos_w'].shape}")
    print(f"  body_quat_w: {output_data['body_quat_w'].shape}")
    print(f"  body_lin_vel_w: {output_data['body_lin_vel_w'].shape}")
    print(f"  body_ang_vel_w: {output_data['body_ang_vel_w'].shape}")
    if "body_names" in output_data:
        print(f"  body_names: {len(output_data['body_names'])} bodies")

    np.savez(output_file, **output_data)
    print("Conversion completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert motion data format")
    parser.add_argument("-i", "--input", required=True, help="Input NPZ file")
    parser.add_argument("-o", "--output", required=True, help="Output NPZ file")
    args = parser.parse_args()

    convert_robotlab_to_beyondmimic(args.input, args.output)
