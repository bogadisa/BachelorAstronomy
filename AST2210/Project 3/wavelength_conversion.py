import numpy as np

z = 0.005476

O_III = np.array([2320.951, 4363.210, 4958.11, 5006.843])
H_alpha = np.array([6562.819])
N_II = np.array([5764.590, 6548.050, 6583.460])
H_beta = np.array([4861.333])
He_II = np.array([5875.624])
He_I = np.array([9063.27])
S_II = np.array([6716.440, 6730.810	])
Pa9 = np.array([9229.014])
Pa10 = np.array([9014.909])
Pa11 = np.array([8862.782])

def z_to_lambda(z, lambda_EM):
    return (z + 1)*lambda_EM

O_III_shifted = np.array([z_to_lambda(z, O) for O in O_III])
H_alpha_shifted = np.array([z_to_lambda(z, H) for H in H_alpha])
N_II_shifted = np.array([z_to_lambda(z, N) for N in N_II])
H_beta_shifted = np.array([z_to_lambda(z, H_2) for H_2 in H_beta])
He_II_shifted = np.array([z_to_lambda(z, He) for He in He_II])
He_I_shifted = np.array([z_to_lambda(z, He_2) for He_2 in He_I])
S_II_shifted = np.array([z_to_lambda(z, S_2) for S_2 in S_II])
Pa9_shifted = np.array([z_to_lambda(z, Pa) for Pa in Pa9])
Pa10_shifted = np.array([z_to_lambda(z, Pa_2) for Pa_2 in Pa10])
Pa11_shifted = np.array([z_to_lambda(z, Pa_3) for Pa_3 in Pa11])


def get_shifted_wavelengths():
    return [O_III_shifted, H_alpha_shifted, N_II_shifted, H_beta_shifted, He_II_shifted, He_I_shifted, S_II_shifted, Pa9_shifted, Pa10_shifted, Pa11_shifted], ["[O III]", r"$H\alpha$", "[N II]", r"$H\beta$", "He II", "He I", "[S II]", "Pa9", "Pa10", "Pa11"]

print("Estimated observed emission lines:")
print(f"[O III] = {[O_III_shifted[i] for i in range(len(O_III_shifted))]}")
print(f"H-aplha = {[H_alpha_shifted[i] for i in range(len(H_alpha_shifted))]}")
print(f"[N II] = {[N_II_shifted[i] for i in range(len(N_II_shifted))]}")

