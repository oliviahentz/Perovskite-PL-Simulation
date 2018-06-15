# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 19:12:35 2017

@author: Olivia
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
       

class one_dim_Distribution(object):
    '''A one_dim_Distribution object is a distribution of charged particles 
    along a line. There is an electric field defined by a magnitude (V/micron) 
    and starting and ending positions along the line. The position of 
    charged particles can be updated to reflect movement over time under the 
    effects of drift and diffusion.

    Input:
        FRACTION_FAST_NEG (fraction of defects which are negative and have 
            mobilities on the order of 10^-10 cm^2/Vsec)
        FRACTION_FAST_POS (fraction of defects which are positive and have 
            mobilities on the order of 10^-10 cm^2/Vsec)
        FRACTION_SLOW_NEG (fraction of defects which are negative and have 
            mobilities on the order of 10^-11 cm^2/Vsec)
        FRACTION_SLOW_POS (fraction of defects which are positive and have 
            mobilities on the order of 10^-11 cm^2/Vsec)       
        ELECTRIC_FIELD_STRENGTH (voltage applied accross electrodes)
        MOBILITY_POS (mobilitiy of fast, positive defects, micron^2/Vhr)
        MOBILITY_NEG (mobilitiy of fast, negative defects, micron^2/Vhr)
        MOBILITY_POS_SLOW (mobilitiy of slow, positive defects, micron^2/Vhr)
        MOBILITY_NEG_SLOW (mobilitiy of slow, negative defects, micron^2/Vhr)
               
        BIASED_WIDTH (distance between electrodes, microns)
        SAMPLE_SIZE (x-dimension of sample region, microns)
        DEFECT_CONC (number of defects per square micron)
        
    
    Initialization: initializing a Distribution object randomly generates a 
    list of particle positions, with the length of the list given by the 
    DEFECT_CONC and SAMPLE_SIZE'''
    
    def __init__ (self,FRACTION_FAST_NEG,FRACTION_FAST_POS,FRACTION_SLOW_NEG,
                  FRACTION_SLOW_POS,ELECTRIC_FIELD_STRENGTH,MOBILITY_POS,
                  MOBILITY_NEG,MOBILITY_POS_SLOW,MOBILITY_NEG_SLOW,
                  BIASED_WIDTH,SAMPLE_SIZE,DEFECT_CONC):
                      
        self.FRACTION_FAST_NEG = FRACTION_FAST_NEG
        self.FRACTION_FAST_POS = FRACTION_FAST_POS
        self.FRACTION_SLOW_POS = FRACTION_SLOW_POS
        self.FRACTION_SLOW_NEG = FRACTION_SLOW_NEG        
        self.BIASED_WIDTH = BIASED_WIDTH 
        self.DEFECT_CONC = DEFECT_CONC
        self.SAMPLE_SIZE = SAMPLE_SIZE
        self.ELECTRIC_FIELD_STRENGTH = ELECTRIC_FIELD_STRENGTH
        self.MOBILITY_POS = MOBILITY_POS
        self.MOBILITY_NEG = MOBILITY_NEG
        self.MOBILITY_POS_SLOW = MOBILITY_POS_SLOW
        self.MOBILITY_NEG_SLOW = MOBILITY_NEG_SLOW
        
        self.DIFFUSIVITY_POS = self.MOBILITY_POS*0.0257
        self.DIFFUSIVITY_NEG = self.MOBILITY_NEG*0.0257
        self.DIFFUSIVITY_POS_SLOW = self.MOBILITY_POS_SLOW*0.0257
        self.DIFFUSIVITY_NEG_SLOW = self.MOBILITY_NEG_SLOW*0.0257
        #um^2/min, D = mukBT/q        
        
        self.DIFFUSION_STD_POS = (2.0*self.DIFFUSIVITY_POS)**0.5
        self.DIFFUSION_STD_NEG = (2.0*self.DIFFUSIVITY_NEG)**0.5
        self.DIFFUSION_STD_POS_SLOW = (2.0*self.DIFFUSIVITY_POS_SLOW)**0.5
        self.DIFFUSION_STD_NEG_SLOW = (2.0*self.DIFFUSIVITY_NEG_SLOW)**0.5
        #varience = 2Dt
        
        self.BIASED_REGION_START = int(self.SAMPLE_SIZE/2- self.BIASED_WIDTH/2) 
        self.BIASED_REGION_END = int(self.SAMPLE_SIZE/2 + self.BIASED_WIDTH/2)
        
        ELECTRIC_FIELD = [0 for n in range(self.BIASED_REGION_START)]
        ELECTRIC_FIELD2 = [self.ELECTRIC_FIELD_STRENGTH/self.BIASED_WIDTH 
            for m in range(self.BIASED_REGION_END-self.BIASED_REGION_START)]
        ELECTRIC_FIELD3 = [0 for n in range(self.SAMPLE_SIZE-
            self.BIASED_REGION_END)]
        ELECTRIC_FIELD.extend(ELECTRIC_FIELD2)
        ELECTRIC_FIELD.extend(ELECTRIC_FIELD3)
        self.ELECTRIC_FIELD = ELECTRIC_FIELD
        
        self.DRIFT_STRENGTH_POS = (self.MOBILITY_POS*
                                        np.asfarray(self.ELECTRIC_FIELD))
        self.DRIFT_STRENGTH_NEG = (self.MOBILITY_NEG*
                                        np.asfarray(self.ELECTRIC_FIELD))
        self.DRIFT_STRENGTH_POS_SLOW = (self.MOBILITY_POS_SLOW*
                                        np.asfarray(self.ELECTRIC_FIELD))
        self.DRIFT_STRENGTH_NEG_SLOW = (self.MOBILITY_NEG_SLOW*
                                        np.asfarray(self.ELECTRIC_FIELD))
        
        self.defects = self.SAMPLE_SIZE*np.random.rand(int(self.DEFECT_CONC*
                        self.SAMPLE_SIZE))
        
        
    def update_defects(self,step):
        ''' Update the location of each defect.'''
        
        if step % 5 != 0:
            self.update_electric_field()        
        
        #Defects of varied sign (negative (n) or positive (p)) and mobility 
        # (fast (f) or slow (s)) are differentiated by their position in 
        #the list of defects. The order of the defect types is:
        #[neg fast, neg slow, pos fast, pos slow] and the start and end of 
        #each type is defined by the fraction of defects which are each type
        nf_end = int(len(self.defects)*self.FRACTION_FAST_NEG) 
            #index indicating end of negative, fast defects
        ns_end = nf_end + int(len(self.defects)*self.FRACTION_SLOW_NEG)
            #index indicating end of negative, slow defects
        pf_end = ns_end + int(len(self.defects)*self.FRACTION_FAST_POS)
            #index indicating end of positive, fast defects. The rest of the 
            #defects in the list are positive, slow defects
        
        movement_values = [0 for x in range(len(self.defects))]
        
        #NEGATIVE DEFECTS
        for point in range(ns_end):
            if point in range(0,nf_end):
                #Drift
                movement_values[point] -= \
                        self.DRIFT_STRENGTH_NEG[int(self.defects[point])]
                #Diffusion
                movement_values[point] += \
                        np.random.normal(0,self.DIFFUSION_STD_NEG)
            if point in range(nf_end,ns_end):
                #Drift
                movement_values[point] -= \
                        self.DRIFT_STRENGTH_NEG_SLOW[int(self.defects[point])]
                #Diffusion
                movement_values[point] += \
                        np.random.normal(0,self.DIFFUSION_STD_NEG_SLOW)
                
        #POSITIVE DEFECTS
        for point in range(ns_end,len(self.defects)):
            if point in range(ns_end,pf_end):
                #Drift
                movement_values[point] += \
                        self.DRIFT_STRENGTH_POS[int(self.defects[point])]
                #Diffusion
                movement_values[point] += \
                        np.random.normal(0,self.DIFFUSION_STD_POS)
            if point in range(pf_end,len(self.defects)):
                #Drift
                movement_values[point] += \
                        self.DRIFT_STRENGTH_POS_SLOW[int(self.defects[point])]
                #Diffusion
                movement_values[point] += \
                        np.random.normal(0,self.DIFFUSION_STD_POS_SLOW)
        
        self.defects += movement_values 
        
        #Handle defects which go off the end of the sample
        for point in range(len(self.defects)):
            if self.defects[point] < 0.0:
                self.defects[point] = 1
            if self.defects[point] >= self.SAMPLE_SIZE:
                self.defects[point] = self.SAMPLE_SIZE-1
        
#        
    def update_electric_field(self):
        '''Update the electric field based on the applied voltage and the
        location of charged defects'''
        
        E = self.get_ELECTRIC_FIELD()
        defects = np.round(self.get_defects(),0)
        FRACTION_NEG = self.FRACTION_FAST_NEG + self.FRACTION_SLOW_NEG
        
        for n in range(self.SAMPLE_SIZE):
            distance = np.asarray(defects - n)
            if FRACTION_NEG >= 1.0:
                elec_change_pos = 0
            else:
                elec_change_pos = -(8.00 *10**-5* 1/(distance[int(len(defects)*
                    FRACTION_NEG):])**2)*np.sign(distance[int(len(defects)*
                    FRACTION_NEG):])
            if FRACTION_NEG <= 0:
                elec_change_neg = 0
            else:
                elec_change_neg = (8.00 *10**-5*1/(distance[:int(len(defects)*
                    FRACTION_NEG)])**2)*np.sign(distance[:int(len(defects)*
                    FRACTION_NEG)])
                    
            E[n] += np.nansum(elec_change_neg) + np.nansum(elec_change_pos)
            
        self.DRIFT_STRENGTH_POS = self.MOBILITY_POS*np.asfarray(E)
        self.DRIFT_STRENGTH_NEG = self.MOBILITY_NEG*np.asfarray(E)
        self.DRIFT_STRENGTH_POS_SLOW = self.MOBILITY_POS_SLOW*np.asfarray(E)
        self.DRIFT_STRENGTH_NEG_SLOW = self.MOBILITY_NEG_SLOW*np.asfarray(E)
 
    def plot_electric_field(self):
        '''Plot the current electric field along the sample'''
        
        x = [1 + m for m in range(0,self.SAMPLE_SIZE)]
        E = self.DRIFT_STRENGTH_POS/self.MOBILITY_POS
        plt.plot(x,E)
    
    def get_ELECTRIC_FIELD(self):
        '''Get the electric field at each micron size step along the sample'''
        ELECTRIC_FIELD = [0 for n in range(self.BIASED_REGION_START)]
        ELECTRIC_FIELD2 = [self.ELECTRIC_FIELD_STRENGTH/self.BIASED_WIDTH for 
                m in range(self.BIASED_REGION_END-self.BIASED_REGION_START)]
        ELECTRIC_FIELD3 = [0 for n in range(self.SAMPLE_SIZE-
                self.BIASED_REGION_END)]
        ELECTRIC_FIELD.extend(ELECTRIC_FIELD2)
        ELECTRIC_FIELD.extend(ELECTRIC_FIELD3)
        return ELECTRIC_FIELD
            
    def get_histogram(self,data,color):
        '''Plot a histogram of the defect positions'''
        
        smoothed = gaussian_filter(data, sigma=2)
        plt.plot(smoothed, color = color, linewidth=2)
        plt.axis([0,self.SAMPLE_SIZE,0,self.DEFECT_CONC+5])
        plt.xlabel("Distance (um)")
        plt.ylabel("Number of Defects")
        plt.title("Defect Migration")
        
    def get_defects(self):
        '''Get the current positions of all defects'''
        return self.defects
        
def main():
    
    MOBILITY_POS = 3.5*10**-10*(10**4)*(10**4)*60*60
    #um^2/Vhr
    MOBILITY_NEG = 3.825*10**-10*(10**4)*(10**4)*60*60
    #um^2/Vhr
    MOBILITY_POS_SLOW = 1.8*10**-11*(10**4)*(10**4)*60*60
    MOBILITY_NEG_SLOW = 2.5*10**-11*(10**4)*(10**4)*60*60
    BIASED_WIDTH = 195
    #um
    SAMPLE_SIZE = 400
    DEFECT_CONC = 50
    
    FRACTION_SLOW_NEG = 1.0
    FRACTION_SLOW_POS = 0
    FRACTION_FAST_NEG = 0
    FRACTION_FAST_POS = 0
    #Defects/um, about 10^16 defects/cm^3
    
    ELECTRIC_FIELD_STRENGTH = 12

    plt.figure()
    curves = []
    colors = ["black","red", "orangered", "orange","yellow","lime","cyan",
              "skyblue","blue","darkviolet","magenta"]
    
    for smooth in range(10):
        #Simulation is run 10 times and averaged
    
        dist = one_dim_Distribution(FRACTION_FAST_NEG,FRACTION_FAST_POS,
                            FRACTION_SLOW_NEG,FRACTION_SLOW_POS,
                            ELECTRIC_FIELD_STRENGTH,MOBILITY_POS,
                            MOBILITY_NEG,MOBILITY_POS_SLOW,
                            MOBILITY_NEG_SLOW,BIASED_WIDTH,
                            SAMPLE_SIZE,DEFECT_CONC)
                            
        curve_smooth = []
        curve_smooth.append(np.histogram(dist.defects,bins = SAMPLE_SIZE)[0]) 
        
        for time in range(51):
            dist.update_defects(time)
#            if time in [0,1,2,3,4,5,22,27,30]:
            if time in [1,3,5,7,23,26,29,49]:
                curve_smooth.append(np.histogram(dist.defects,
                                                 bins = SAMPLE_SIZE)[0])
        curves.append(curve_smooth)
        
    average = np.average(curves,0)
    
    for timestamp in range(len(curves[0])):       
        dist.get_histogram(average[timestamp],colors[timestamp])
        
    plt.savefig("1D_defect_migration_25e-12.svg")
    plt.close()

if __name__ == "__main__":
    main()