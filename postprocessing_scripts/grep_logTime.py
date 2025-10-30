import re, pathlib                                                              
import numpy as np                                                              
import csv                                                                      
import os                                                                       
import sys
                                                                                
                                                                                
def grep_logTime(filename, st):                                                 
    text = pathlib.Path(filename).read_text()                                   
    pat = re.compile(rf"\b{re.escape(st)}\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    m = pat.search(text)                                                        
    n = float(m.group(1)) if m else None                                        
                                                                                
    return n                                                                    
                                                                                

def grep_ctr(st, ctr_file="incompressible_tml.ctr"):                            
    """ To grep all the required data from the CTR file                         
                                                                                
    Args:                                                                       
        st (string) : The variable whose data is to be grep-ed                  
                                                                                
    Return:                                                                     
        The variable's value                                                    
    """                                                                         
                                                                                
    text = pathlib.Path(ctr_file).read_text()                                   
    pat = re.compile(rf"\b{re.escape(st)}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    m = pat.search(text)                                                        
    n   = float(m.group(1)) if m else None                                      
                                                                                
    return n
                                                                                

def single_log(shift_step=0):
    nx_g        = int(grep_ctr("nx"))
    output_step = int(grep_ctr("output_step"))
    dt          = grep_ctr("dt")                                                       
    delta       = (2 * np.pi) / 100                                             
    U_g         = 3.1830988618379066                                            

    first_time_step_number  = 0
    final_time_step_file    = max(pathlib.Path(".").glob("time_step*"), key=lambda p: p.stat().st_mtime, default=None)
    final_time_step_number  = int(re.search(r'time_step-(\d+)$', final_time_step_file.name).group(1)) if final_time_step_file else None

    f = open(f"Time_n{nx_g}.csv", "w", newline="")                                 
    w = csv.writer(f)                                                           
    w.writerow(["Time_step", "Time", "Time_Normalized"])                         
                                                                                
    for i in range(0, final_time_step_number, output_step):
        time_step    = i                              
        time         = dt * time_step
        t_normalized = (time * U_g) / delta                                     
                                                                                
        w.writerow([time_step, time, t_normalized])                             
        #Push to OS buffers                                                     
        f.flush()                                                               
        os.fsync(f.fileno())                                                    
    

def multiple_log(shift_step=0):                                                                     
    log  = sorted(pathlib.Path(".").glob("log_*"))                               
    nx_g = grep_ctr("nx_g")
                                                                                
    f = open(f"Time_n{nx_g}.csv", "w", newline="")                                 
    w = csv.writer(f)                                                           
    w.writerow(["Time_step", "Time", "TimeNormalized"])                         
                                                                                
    output_step = grep_ctr("output_step")                                                          
    dt          = grep_ctr("dt")                                                       
    delta       = (2 * np.pi) / 100                                             
    U_g         = 3.1830988618379066                                            
                                                                                
    for i, filename in enumerate(log):                                          
        time_step    = i * output_step + shift_step                             
        time         = (grep_logTime(filename, 'Time'))                         #time = dt * time_step; Well not technical    ly, since of adapt time
        t_normalized = (time * U_g) / delta                                     
                                                                                
        w.writerow([time_step, time, t_normalized])                             
        #Push to OS buffers                                                     
        f.flush()                                                               
        os.fsync(f.fileno())                                                    
                                                                                

if __name__ == "__main__":                                                      
    #If multiple log files
    if len(sys.argv) > 1:
        #shiftstep=0
        multiple_log()
    
    else:
        #shiftstep=0
        single_log()
