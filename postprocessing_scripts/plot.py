import numpy as np                                                              
import matplotlib.pyplot as plt                                                 
                                                                                

n256_data = "integrand_data_n256.csv"
n512_data = "integrand_data_n512.csv"
n1024_data = "integrand_data_n1024.csv"
files = [["n256", "n512", "n1024"], [n256_data, n512_data, n1024_data]]
integrands = [["Time"], ["Momentum_Thickness", "VolFrac_Thickness", "Mixing_Thickness"]]


def plot_thickness(filename, a, label, linestyle="-"):                          
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)                
    time_steps = data[:, 0]                                                     
    y = data[:, a]                                                              
    plt.plot(time_steps, y, linestyle, label=label)                             
                                                    

def perform_plotting():
    for ii in range(3):
        for jj in range(3):
            plot_thickness(files[1][jj], 1, label=files[0][jj], linestyle="-")  

            plt.xlabel(integrands[0][0])                                                              
            plt.ylabel(integrands[1][ii])                                                
            plt.legend()                                                                    
            plt.grid(True)                                                                  
                                                                                            
            plt.savefig(f"{integrands[1][ii]}_overlay.png") 


def main(): 
    perform_plotting()


if __name__ == "__main__":
    main()

