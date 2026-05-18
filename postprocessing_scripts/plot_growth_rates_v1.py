import numpy as np                                                              
import matplotlib.pyplot as plt                                                 
import os
import argparse
                                                                                
datasetparser = argparse.ArgumentParser(description="Read optional CSV datasets")

datasetparser.add_argument("--dataset_1", type=str, default=None, help="CSV file for dataset 1")
datasetparser.add_argument("--dataset_2", type=str, default=None, help="CSV file for dataset 2")
datasetparser.add_argument("--dataset_3", type=str, default=None, help="CSV file for dataset 3")
datasetparser.add_argument("--dataset_4", type=str, default=None, help="CSV file for dataset 4")
datasetparser.add_argument("--dataset_5", type=str, default=None, help="CSV file for dataset 5")
datasetparser.add_argument("--dataset_6", type=str, default=None, help="CSV file for dataset 6")

datasets = datasetparser.parse_args()
dataset_1 = datasets.dataset_1
dataset_2 = datasets.dataset_2
dataset_3 = datasets.dataset_3
dataset_4 = datasets.dataset_4
dataset_5 = datasets.dataset_5
dataset_6 = datasets.dataset_6
files = [
    ["dataset_1", "dataset_2", "dataset_3", "dataset_4", "dataset_5", "dataset_6"],
   [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6],
]
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
            if files[1][jj] is None:
                continue

            if not os.path.exists(files[1][jj]):
                print(f"--File {files[1][jj]} not found, skipping.")
                continue

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

