# Neural-IK

This project aims to develop a neural network model to assist in robotic movement planning. Given the 3D coordinates (x, y, z) of a target item in space, the model will predict how small joint angle adjustments (dθ) in a robotic arm affect the robot's proximity to that target.

## Objective
The ultimate goal is to iteratively minimize the distance between the robot’s end-effector and the target item by optimizing the set of angular movements (dθ) for each joint using gradient-based techniques.

## How It Works:
- Input: Cartesian coordinates (x, y, z) of the target obtained through an egocentric camera placed on the actuator's wrist.

- Output: A set of small angle changes dθ₁, dθ₂, ..., dθₙ for each joint in the robot.

- Loss Function: Distance between the robot's end-effector (after applying dθ) and the target.

- Optimization: Sum of dθs per joint is used to compute cumulative movement needed.

## Future Steps:
- Implement forward kinematics to verify end-effector position based on joint angles.

- Train the model to predict useful dθ values for any given target position.

- Evaluate convergence and total joint effort required.
