import matplotlib as mpl
from cycler import cycler

def paper_style():
     defaultcolor = "r"
     mpl.rcParams.update({                                                           
            "axes.prop_cycle": cycler(color=[defaultcolor]),

             # Figure                                                                
             "figure.figsize": (3.5, 2.625),                                         
                                                                                     
             #Font                                                                   
             "font.family"                   : "serif",                              
             "font.serif"                    : ["STIXGeneral"],                      
             "axes.formatter.use_mathtext"   : True,                                 
             "mathtext.fontset"              : "cm",                                 
             "font.size"                     : 14,                                  
                                                                                     
             #Ticks                                                                  
             "xtick.direction"       : "in",                                         
             "xtick.major.size"      : 3,                                            
             "xtick.major.width"     : 0.5,                                          
             "xtick.minor.size"      : 1.5,                                          
             "xtick.minor.width"     : 0.5,                                          
             "xtick.minor.visible"   : True,                                         
             "xtick.top"             : True,                                         
                                                                                     
             "ytick.direction"       : "in",                                         
             "ytick.major.size"      : 3,                                            
             "ytick.major.width"     : 0.5,                                          
             "ytick.minor.size"      : 1.5,                                          
             "ytick.minor.width"     : 0.5,                                          
             "ytick.minor.visible"   : True,                                         
             "ytick.right"           : True,                                         
                                                                                     
             #Linewidth                                                              
             "lines.linewidth"   : 1.0,                                              
                                                                                     
             #Axis                                                                   
             "axes.linewidth"    : 1.2,                                              
             "axes.labelsize"    : 14,                                               
             "axes.titlesize"    : 14,                                                
             "axes.grid"         : True,                                             
             "axes.axisbelow"    : True,                                             
             "axes.edgecolor"    : "k",                                              
                                                                                     
             #Grid                                                                   
             "grid.linewidth"    : 0.5,                                              
             "grid.linestyle"    : "--",                                             
             "grid.color"        : "0.45",                                           
             "grid.alpha"        : 0.25,                                             
                                                                                     
             #Legend                                                                 
             "legend.frameon"    : True,                                             
             "legend.framealpha" : 1.0,                                              
             "legend.fancybox"   : False,                                            
             "legend.edgecolor"  : "none",                                           
             "legend.numpoints"  : 1,                                                
             "legend.loc"        : "best",                                           
             "legend.fontsize"   : 10,                                                
                                                                                     
             #Saving                                                                 
             "savefig.bbox"      : "tight",                                          
             "savefig.pad_inches": 0.05,                                             
         })
