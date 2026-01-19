import numpy as np                                                              
import matplotlib.pyplot as plt                                                 
import os
                                                                                

n128_data = "integrand_data_n128.csv"
n256_data = "integrand_data_n256.csv"
n512_data = "integrand_data_n512.csv"
n1024_data = "integrand_data_n1024.csv"
files = [["n128", "n256", "n512", "n1024"], [n128_data, n256_data, n512_data, n1024_data]]
integrands = [["Time"], ["Momentum_Thickness", "VolFrac_Thickness", "Mixing_Thickness"]]


def plot_thickness(filename, a, label, linestyle="-"):                          
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)                
    time_steps = data[:, 0]                                                     
    y = data[:, a]                                                              
    plt.plot(time_steps, y, linestyle, label=label)                             
                                                    

def perform_plotting(specific_dataset = False):
    for ii in range(3):
        if(specific_dataset):
            start = specific_dataset - 1
            stop  = specific_dataset
        else:
            start = 0
            stop  = 4 
        for jj in range(start, stop):
            if not os.path.exists(files[1][jj]):
                print(f"--File {files[1][jj]} not found, skipping.")
                continue

            #if jj == 0:
            #    continue

            plot_thickness(files[1][jj], 1, label=files[0][jj], linestyle="-")  
            plt.xlabel(integrands[0][0])                                                              
            plt.ylabel(integrands[1][ii])                                                
            plt.legend()                                                                    
            plt.grid(True)                                                                  
                                                                                            
            plt.savefig(f"{integrands[1][ii]}_overlay.png") 


def main(): 
    arg = input("--To plot all? ")
    if(arg == "Yes"):
        perform_plotting()
    else:
        to_plot = input("--Which dataset do you wish to plot? ")
        perform_plotting(to_plot)


if __name__ == "__main__":
    main()

