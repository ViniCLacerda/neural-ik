import numpy as np
import csv

LINK_LENGTHS = [1.0, 1.0, 1.0]

def forward_kinematics(theta, link_lengths=LINK_LENGTHS):
    x, y, angle = 0.0, 0.0, 0.0
    for i in range(3):
        angle += theta[i]
        x += link_lengths[i] * np.cos(angle)
        y += link_lengths[i] * np.sin(angle)
    return np.array([x, y])

def generate_sample(theta_init, target_pos, link_lengths=LINK_LENGTHS, alpha=0.1, max_iter=10):
    epsilon = 1e-4
    theta = theta_init.copy()
    
    for _ in range(max_iter):
        current_pos = forward_kinematics(theta, link_lengths)
        error = target_pos - current_pos

        if np.linalg.norm(error) < 1e-3:
            break

        J = np.zeros((2, 3))
        for i in range(3):
            theta_eps = theta.copy()
            theta_eps[i] += epsilon
            pos_eps = forward_kinematics(theta_eps, link_lengths)
            J[:, i] = (pos_eps - current_pos) / epsilon

        dtheta = np.linalg.pinv(J) @ error
        theta += alpha * dtheta

    total_dtheta = theta - theta_init
    return np.concatenate([theta_init, target_pos, total_dtheta])

def generate_dataset(n_samples=1000, output_file="dataset.csv"):
    data = []
    for _ in range(n_samples):
        theta_init = np.random.uniform(-np.pi / 2, np.pi / 2, 3)
        target_theta = np.random.uniform(-np.pi / 2, np.pi / 2, 3)
        target_pos = forward_kinematics(target_theta)
        sample = generate_sample(theta_init, target_pos)
        data.append(sample)

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['theta0', 'theta1', 'theta2', 'x_target', 'y_target', 'dtheta0', 'dtheta1', 'dtheta2'])
        writer.writerows(data)

    print(f"Dataset salvo em {output_file}")

if __name__ == "__main__":
    generate_dataset(n_samples=2000)
