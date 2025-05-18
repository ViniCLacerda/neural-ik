import torch
import numpy as np
from model import MLP
from dataset import forward_kinematics

model = MLP()
model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict_dtheta(theta, x_target, y_target):
    input_tensor = torch.tensor([[*theta, x_target, y_target]], dtype=torch.float32)
    with torch.no_grad():
        dtheta = model(input_tensor).numpy().flatten()
    return dtheta

if __name__ == "__main__":
    theta_init = np.array([0.0, 0.0, 0.0])
    target_pos = np.array([1.5, 1.5])

    predicted_dtheta = predict_dtheta(theta_init, *target_pos)
    print("Theta inicial:", theta_init)
    print("Posição alvo:", target_pos)
    print("dTheta previsto:", predicted_dtheta)
    theta_final = theta_init + predicted_dtheta
    final_pos = forward_kinematics(theta_final)
    print("Erro (distância euclidiana):", np.linalg.norm(final_pos - target_pos))


np.savez('inference_results.npz', theta_init=theta_init, theta_final=theta_final, target_pos=target_pos)

