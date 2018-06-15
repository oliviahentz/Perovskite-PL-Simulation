
import numpy as np
import matplotlib.pyplot as plt
import random
import math


class Distribution(object):
    '''A Distribution object is a distribution of charged particles in a 2D 
    space. The 2D space has an electric field along the x-direction, defined 
    by a magnitude (V/micron) and starting and ending x-axis positions. 
    Grain boundaries are lines in the 2D space along which charged particles 
    have increased mobility, defined by the GRAIN_FACTOR. The position of 
    charged particles can be updated to reflect movement over time under the 
    effects of drift and diffusion.

    Input:
        ELECTRIC_FIELD_STRENGTH (voltage applied accross electrodes)
        MOBILITY (mobility of carriers in grains, cm2/Vsec)
        BIASED_WIDTH (distance between electrodes, microns)
        SAMPLE_SIZE (x-dimension of sample region, microns)
        DEFECT_CONC (number of defects per square micron)
        GRAINS (number of grain boundaries in sample)
        NUM_ANGLES (number of possible grain boundary angles)
        GRAIN_FACTOR (mobility at grain boundaries/mobility in grains)
        Barrier (mobility perpendicular to grain boundaries/mobility in grains)
    
    Initialization: initializing a Distribution object randomly generates a 
    list of particle positions, with the length of the list given by the 
    DEFECT_CONC and SAMPLE_SIZE'''
    
    def __init__ (self,ELECTRIC_FIELD_STRENGTH,MOBILITY,BIASED_WIDTH,
                  SAMPLE_SIZE,DEFECT_CONC,GRAINS,NUM_ANGLES,GRAIN_FACTOR,
                  BARRIER):
        self.BIASED_WIDTH = BIASED_WIDTH   
        self.GRAINS = GRAINS
        self.NUM_ANGLES = NUM_ANGLES
        self.ELECTRIC_FIELD_STRENGTH = ELECTRIC_FIELD_STRENGTH
        self.SAMPLE_SIZE = SAMPLE_SIZE
        self.BARRIER = BARRIER
        self.DEFECT_CONC = DEFECT_CONC

        self.MOBILITY = MOBILITY
        self.MOBILITY_Grain = MOBILITY*GRAIN_FACTOR
        self.MOBILITY_PERP = MOBILITY/BARRIER
        
        #Diffusivity (D) = mobility * (Boltzmann constant* Temp./charge), 
        #(micron^2/time step)
        self.DIFFUSIVITY = self.MOBILITY*0.0257
        self.DIFFUSIVITY_Grain = self.MOBILITY_Grain * 0.0257
        self.DIFFUSIVITY_Perp = self.MOBILITY_PERP *0.0257
        
        #varience = 2Dt, t = time per update step        
        self.DIFFUSION_STD = (2.0*self.DIFFUSIVITY)**0.5
        self.DIFFUSION_STD_Grain = (2.0*self.DIFFUSIVITY_Grain)**0.5
        self.DIFFUSION_STD_Grain_perp = (2.0*self.DIFFUSIVITY_Perp)**0.5
        
        self.BIASED_REGION_START = int(self.SAMPLE_SIZE/2 - BIASED_WIDTH/2) 
        self.BIASED_REGION_END = int(self.SAMPLE_SIZE/2 + BIASED_WIDTH/2)

        
        self.DRIFT_STRENGTH = (self.MOBILITY*(self.ELECTRIC_FIELD_STRENGTH/
                            self.BIASED_WIDTH))
        self.DRIFT_STRENGTH_GRAIN = (self.MOBILITY_Grain*
                            (self.ELECTRIC_FIELD_STRENGTH/self.BIASED_WIDTH))
        self.DRIFT_STRENGTH_GRAIN_perp = self.DRIFT_STRENGTH/self.BARRIER
        
        #Grain boundary locations are stored by their slope, intercept 
        #and angle relative to the horizontal.
        self.grain_boundaries_slope = []
        self.grain_boundaries_intercept = []
        self.grain_boundaries_angles = []
        
        #Initiate a set of defects by random selection of an x and y position. 
        # Point i has position (defectsx[i], defectsy[i])
        defectsx = SAMPLE_SIZE*np.random.rand(int(self.DEFECT_CONC*
                        self.SAMPLE_SIZE*self.SAMPLE_SIZE/4)) 
        defectsy = SAMPLE_SIZE/4*np.random.rand(int(self.DEFECT_CONC*
                        self.SAMPLE_SIZE*self.SAMPLE_SIZE/4))
        
        
        self.defectsx = np.asfarray(defectsx)
        self.defectsy = np.asfarray(defectsy)
        
    def set_grain_boundaries(self):
        '''Randomly select the location and angle of grain boundaries in the 
        sample. The number of grain boundaries is set by self.GRAINS and the 
        number of possibe angles for these boundaries is set by 
        self.NUM_ANGLES. '''
        
        angles = [] #grain boundary angles, defined as angle above x-axis
        locations = [] #grain boundary y-intercepts
        slopes = [] #grain boundary slopes
        
        #randomly pick angles for grain boundaries
        angles = math.pi* np.random.rand(self.NUM_ANGLES)   
        slopes = [math.tan(x) for x in angles]
        
        for ang in angles: #generate grain boundaries with the selected angles
            b_values = []
            for value in range(int(self.GRAINS/self.NUM_ANGLES)):
                loc = [random.random()*(self.SAMPLE_SIZE-1),
                       random.random()*(self.SAMPLE_SIZE-1)]
                b = loc[1] - math.tan(ang) * loc[0] #grain boundary y-intercept
                b_values.append(b)
                
            locations.append(b_values)
            

            self.grain_boundaries_slope = np.asarray(slopes)
            self.grain_boundaries_intercept = np.round(locations,3) 
            self.grain_boundaries_angles = angles
            
          
    
    def update_defects(self):
        ''' Update the location of each defect.'''
             
        movement_valuesx = [0 for x in range(len(self.defectsx))]
        movement_valuesy = [0 for x in range(len(self.defectsy))]
        movement_driftx = [0 for x in range(len(self.defectsx))]
        movement_drifty = [0 for x in range(len(self.defectsy))]
               
        atboundaries_val = np.round((np.outer(self.grain_boundaries_slope,
                                self.defectsx) - [self.defectsy 
                                for x in range(self.NUM_ANGLES)]),3) 
        atboundaries = []        
        for slope in range(self.NUM_ANGLES):  
            atboundaries.append(np.in1d(atboundaries_val[slope],
                                -self.grain_boundaries_intercept[slope]))
        atboundaries = np.asarray(atboundaries).transpose() 
        #atboundaries is an array of shape (num points, num grain boundaries), 
        #1 indicates point at grain broundary, 0 indicates not at boundary

        for point in range(len(self.defectsx)):
            angles = []
            
            grains_found = np.where(atboundaries[point])[0] 
            #grains_found is an array of indices indicating the angles of 
            #grain boundaries through this point
            
            for grain_boundary in grains_found:
                angles.append(self.grain_boundaries_angles[grain_boundary])
                angles.append(self.grain_boundaries_angles[grain_boundary] + 
                                math.pi)
            
            #Diffusion            
            if len(angles) != 0: #point is on one or more grain boundaries
                diffusion_angle = random.choice(angles)
                diff_dist_par = np.random.normal(0,self.DIFFUSION_STD_Grain)
                diff_dist_perp = np.random.normal(0,
                                        self.DIFFUSION_STD_Grain_perp)

                movement_valuesx[point] += (math.sin(diffusion_angle) * 
                    diff_dist_perp + math.cos(diffusion_angle) * diff_dist_par)
                    
                movement_valuesy[point] += (math.sin(diffusion_angle) * 
                    diff_dist_par - math.cos(diffusion_angle) * diff_dist_perp)

            else: # point is in a grain
                diffusion_angle = random.uniform(0,2*math.pi)
                diff_dist = np.random.normal(0,self.DIFFUSION_STD)

                movement_valuesx[point] += math.cos(diffusion_angle)*diff_dist
                movement_valuesy[point] += math.sin(diffusion_angle)*diff_dist
        
            #Drift
            if self.BIASED_REGION_START < self.defectsx[point] < \
                    self.BIASED_REGION_END:
                        
                if angles == []: # point is in a grain
                    movement_driftx[point] += self.DRIFT_STRENGTH
                    
                else: # point is on one or more grain boundaries
                    angle = diffusion_angle
                    
                    movement_driftx[point] += (math.cos(angle)*math.cos(angle)*
                        self.DRIFT_STRENGTH_GRAIN +  math.sin(angle)*
                        math.sin(angle)*self.DRIFT_STRENGTH_GRAIN_perp)
                        
                    movement_drifty[point] += (math.sin(angle)*math.cos(angle)*
                        self.DRIFT_STRENGTH_GRAIN - math.sin(angle)*
                        math.cos(angle)*self.DRIFT_STRENGTH_GRAIN_perp)       
                    
        self.defectsx += movement_valuesx
        self.defectsy += movement_valuesy
        self.defectsx += movement_driftx
        self.defectsy += movement_drifty

        #handle points that go out of range
        for point in range(len(self.defectsx)):
            if self.defectsx[point] < 0.0:
                self.defectsx[point] -= movement_valuesx[point]
            if self.defectsx[point] > self.SAMPLE_SIZE:
                self.defectsx[point] -= movement_valuesx[point]
            if self.defectsy[point] < 0.0:
                self.defectsy[point] -= movement_valuesy[point]
                if self.defectsy[point] < 0.0:
                    self.defectsy[point] += self.SAMPLE_SIZE/4
            if self.defectsy[point] > self.SAMPLE_SIZE/4:
                self.defectsy[point] -= movement_valuesy[point]
                if self.defectsy[point] > self.SAMPLE_SIZE/4:
                    self.defectsy[point] -= (self.SAMPLE_SIZE/4)
                
        return self.defectsx, self.defectsy
        
            
    def get_histogram(self,i):
        '''Generate a histogram of of the defect positions along the y-axis. 
        i indicates the color to be used for the histogram, selected to 
        match experimental data for a given time step'''
        
        colors=["black","red", "orangered", "orange","yellow","lime","cyan",
                "skyblue","blue","darkviolet","magenta","black","red"]
                
        data = np.histogram2d(self.defectsy,self.defectsx,bins = 
            [self.SAMPLE_SIZE/4,self.SAMPLE_SIZE])
            
        integrated = np.sum(data[0],0)/(self.SAMPLE_SIZE/4)
        plt.plot(integrated, color = colors[i])
        plt.axis([0,self.SAMPLE_SIZE,0,self.DEFECT_CONC+10])
        plt.xlabel("Distance (um)")
        plt.ylabel("Number of Defects")
        plt.title("Defect Migration")

        
    def get_defects(self):
        ''' return a list of defect positions of the form 
        [x-positions, y-positions]'''
        
        return [self.defectsx, self.defectsy]


    def update_histogram(self,step):
        '''Generate a histogram of the current defect positions and then update
        positions with one time step'''
        
        self.get_histogram(step)
        self.update_defects()


def main():
    SAMPLE_SIZE = 400
    NUM_ANGLES = 20
    ELECTRIC_FIELD_STRENGTH = 12
    DEFECT_CONC = 55
    BARRIER = 100
    MULT = 5 #length of time step in minutes
    MOBILITY = 14*10**-13*(10**4)*(10**4)*60*MULT
    BIASED_WIDTH = 185
    GRAIN_FACTOR = 1500
    GRAINS = 5100

    dist = Distribution(ELECTRIC_FIELD_STRENGTH,MOBILITY,BIASED_WIDTH,
                SAMPLE_SIZE,DEFECT_CONC,GRAINS,NUM_ANGLES,GRAIN_FACTOR,BARRIER)
    dist.set_grain_boundaries()

    for time in range(int(56 * 60/MULT)):                        
        times = np.asarray([0,2,4,6,7,24,31,48,55]) * 60/MULT
        times_list = [x for x in times]
        if time in times:
            plt.show()
            i = times_list.index(time)
            dist.update_histogram(i)
        else:
            plt.show()
            dist.update_defects()
    plt.savefig("Large_grains_defect_migration.jpeg")
    plt.close()

    
if __name__ == "__main__":
    main()
    