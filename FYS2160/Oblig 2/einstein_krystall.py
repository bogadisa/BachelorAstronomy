import numpy as np
import matplotlib.pyplot as plt

def factorial(x):
    if x!=0 and x!=1:
        return x*factorial(x-1)
    else:
        return 1

def omega(N, q):
    return factorial(N - 1 + q)/factorial(q)/factorial(N-1)


q = int(input("q="))
Na = int(input("Na="))
Nb = int(input("Nb="))

choice = input("Letter of task (e, g)?\n")
if choice =="e":
    omega_array_a = np.zeros(q+1)
    omega_array_b = np.copy(omega_array_a)
    qa_array = np.copy(omega_array_a)

    for i in range(q+1):
        omega_array_a[i] = omega(Na, q-i)
        if Nb != 0:
            omega_array_b[i] = omega(Nb, i)
        else:
            omega_array_b[i] = 1
        qa_array[i] = q-i

        print(f"Probability of microstates where qa={qa_array[i]} is 1/{omega_array_a[i]*omega_array_b[i]:.0f}")


    plt.plot(qa_array, omega_array_a)
    plt.plot(qa_array, omega_array_b)
    plt.show()
    print(f"Total amount of microstates {sum(omega_array_a*omega_array_b)}")

if choice =="g":
    a = 1
    omega_array_a = np.zeros(q+1)
    omega_array_b = np.copy(omega_array_a)
    qa_array = np.copy(omega_array_a)

    for i in range(q+1):
        omega_array_a[i] = omega(Na, q-i)
        if Nb != 0:
            omega_array_b[i] = omega(Nb, i)
        else:
            omega_array_b[i] = 1
        qa_array[i] = q-i

    probability_distribution = omega_array_a*omega_array_b/sum(omega_array_a*omega_array_b)
    print(f"The most likely state has a probability of {max(probability_distribution)}")
    #plt.plot(qa_array, omega_array_a/sum(omega_array_a))
    #plt.plot(qa_array, omega_array_b/sum(omega_array_b))
    plt.plot(qa_array, probability_distribution)
    plt.xlabel(r"$q_A$")
    plt.ylabel(r"Probability")
    plt.show()