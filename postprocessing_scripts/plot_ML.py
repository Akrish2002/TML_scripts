import numpy as np
import matplotlib.pyplot as plt

def plot_thickness(filename, a, label, linestyle="-"):
    data = np.genfromtxt(filename, delimiter=",", skip_header=1) 
    time_steps = data[:, 0]  
    y = data[:, a]
    #momentum_thickness = data[:, 1] 
    plt.plot(time_steps, y, linestyle, label=label)

#Mixing Layer
plot_thickness("integrand_data_n256.csv", 3, label="n256", linestyle="-")
plot_thickness("integrand_data_n512.csv", 3, label="n512", linestyle="--")
plot_thickness("integrand_data_n1024.csv", 3, label="n1024", linestyle="--")

plt.xlabel("Time")
plt.ylabel("Mixing Layer Thickness")
plt.legend()
plt.grid(True)

plt.savefig("Mixing_Layer_overlay.png")
