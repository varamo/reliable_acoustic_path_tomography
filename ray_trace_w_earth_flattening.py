#! python3
''' ray_trace_w_earth_flattening.py - Class developed in order to compute 
ray tracing for the RAP Tomography project. Expected inputs are as follows:

1. Surface distance from vessel to hydrophone (x_dist)
2. Elevation of transmission (tx_z)
3. Latitude of transmission (tx_lat)
4. Azimuth between vessel and hydrophone (alpha)
5. Hydrophone latitude (rx_lat)
6. Hydrophone longitude (rx_lon)
7. Hydrophone depth (rx_z)

The code derives a SS profile from a HOTS CTD cast and fits this to range
between the source (vessel) to the receiver (seafloor). An earth flattening
transformation is performed (according to Chadwell 2010) on the SS profile.
Lastly, the corresponding RAP launch angle is found using the Newton-Raphson
method and it's corresponding ray tracing information is outputted. '''

import numpy as np
from gsw import SA_from_SP, z_from_p, p_from_z, sound_speed
from math import sin, cos, tan, asin, pi, sqrt
class Ray_Tracing:


    def __init__(self, x_dist, tx_z, tx_lat, alpha, rx_lat, rx_lon, rx_z):
        self.x_dist =   x_dist
        self.tx_z   =   -tx_z
        self.tx_lat =   tx_lat
        self.alpha  =   alpha
        self.rx_lat =   rx_lat
        self.rx_lon =   rx_lon
        self.rx_z   =   -rx_z
        
        # Fix incorrect z measurements
        if self.tx_z < 0:
            self.tx_z = 0



    # Converts HOT CTD data into a SS profile for a given transmission
    def ctd_data(self):

        # Open/read CTD cast file
        ctd_fname = 'h306a0202.ctd'
        with open(ctd_fname, 'r') as ctd_file:
            ctd_contents = np.loadtxt(ctd_file, skiprows=6)
        
        # Extract Presseure, Temperature, and Salinity
        pres = []           # (dbars)
        temp = []           # (Celsius from ITS-90)
        sal = []            # (Sp converted to Sa)
        for line in ctd_contents:
            pres.append(line[0])
            temp.append(line[1])
            sal.append(SA_from_SP(line[2], pres[-1], self.rx_lon, self.rx_lat))
        
        # Convert the pressure into a depth (m)
        N_mean = 2.32               # WGS84 avg geoid height over area
        z_CTD = -(z_from_p(pres, self.rx_lat) + N_mean)

        # Truncate CTD to fit range between source and receiver:
        # Remove layers above where transmission occurred
        rm_ind_up = [i for i, x in enumerate(z_CTD) if x <= self.tx_z]
        z_CTD = np.delete(z_CTD, rm_ind_up[:-1])
        temp = np.delete(temp, rm_ind_up[:-1])
        sal = np.delete(sal, rm_ind_up[:-1])

        # Set first depth to be the transducer depth and interpolate
        z_diff = self.tx_z - z_CTD[0]
        temp[0] = temp[0] + ((temp[1] - temp[0]) * 
            (z_diff / (z_CTD[1] - z_CTD[0])))
        sal[0] = sal[0] + ((sal[1] - sal[0]) * 
            (z_diff / (z_CTD[1] - z_CTD[0])))
        z_CTD[0] = self.tx_z
        
        # Remove layers if below where reception occurred
        if z_CTD[-1] > self.rx_z:
            rm_ind_dn = [i for i, x in enumerate(z_CTD) if x > self.rx_z]
            z_CTD = np.delete(z_CTD, rm_ind_dn[:])
            temp = np.delete(temp, rm_ind_dn[:])
            sal = np.delete(sal, rm_ind_dn[:])

        # Add layers if above where reception occurred
        else:
            lyr_thck = -np.mean(z_CTD[-5:-1] - z_CTD[-4:])
            add_lyr = int(np.floor((self.rx_z - z_CTD[-1]) / lyr_thck))
            z_CTD = np.concatenate([z_CTD, z_CTD[-1] + [lyr_thck] * np.linspace(1, add_lyr, add_lyr)])
            z_CTD = np.append(z_CTD, self.rx_z)

            # Interpolate and add layers for temp and sal
            temp_grad = temp[-2] - temp[-1]
            temp = np.concatenate([temp, temp[-1] + [temp_grad] * np.linspace(1, add_lyr+1, add_lyr+1)])
            sal_grad = sal[-2] - sal[-1]
            sal = np.concatenate([sal, sal[-1] + [sal_grad] * np.linspace(1, add_lyr+1, add_lyr+1)])
        
        
        # Create SS profile
        pres = p_from_z(-z_CTD, self.rx_lat)        # updated pressure
        SS_pro = sound_speed(sal, temp, pres)

        return SS_pro, z_CTD



    # Performs a earth flattening transformation for a given SS and depth 
    # profile (see Chadwell 2010)
    def earth_flat_transform(self, SS, z):

        # Earth's semi-major axis and eccentricity relative to WGS84
        major_ax = 6_378_137
        ecc = 0.0818191908426215

        # Calculate earth flattening variables
        M_r = major_ax * (1 - ecc ** 2) / ((1 - (ecc ** 2) * 
            (sin(self.tx_lat * pi / 180) ** 2)) ** 1.5)
        N_r = major_ax / sqrt(1 - (ecc ** 2) * 
            (sin(self.tx_lat * pi / 180) ** 2))

        self.alpha = self.alpha * pi / 180
        R_alpha = (((cos(self.alpha) ** 2) / M_r) + 
            ((sin(self.alpha) ** 2) / N_r)) ** -1
        
        # SS and depth profile transformed
        z_flt = -R_alpha * np.log(R_alpha / (R_alpha + z))
        SS_flt = SS * (R_alpha / (R_alpha + z))
        
        return SS_flt, z_flt

    
    
    # Calculate the ray tracing (Snells Law) given the inputted variables
    def ray_trace(self, SS, z):

        def snell_law(theta0, SS, z):
            # Ray parameter
            a = sin(theta0) / SS[0]

            # Incident angles of each layer
            thetas = [theta0]
            for SS_lyr in SS[1:-1]:
                thetas.append(asin(a * SS_lyr))

            # Surface distance
            r = []
            for i in range(len(thetas)):
                r.append((z[i + 1] - z[i]) * tan(thetas[i]))
            
            # Total surface distance
            x_diff = sum(r)

            return x_diff, thetas, r

        
        # List of possible launch angles
        theta0_all = np.linspace(0.0001, 87.6 * pi / 180, 90)

        # Loop through launch angles to find two that provide closest to
        # RAP path
        x_diff = []
        for t_num, theta0 in enumerate(theta0_all):

            # Ray tracing to find range
            x_temp, _, _ = snell_law(theta0, SS, z)
            x_diff.append(x_temp)
            
            # Check if we are past surface range and save the surface
            # distance with their respective launch angles
            if (self.x_dist - x_diff[-1] < 0) and (len(x_diff) > 1):
                x_low, x_high = x_diff[t_num - 1:t_num + 1]
                t_low, t_high = theta0_all[t_num - 1:t_num + 1]
                break

        
        # Interpolate new launch angle to begin Newton-Raphson method
        # search for exact launch angle and corresponding ray properties
        theta_new = t_low + (self.x_dist / ((x_low - x_high) / (t_low - t_high))) - \
            (x_low / ((x_low - x_high) / (t_low - t_high)))
        t1 = t_low
        x1 = x_low

        x_diff = 1_000_000          # reset x distance to begin N-R loop

        # Loop until angle found that provides a distance within 0.1 mm
        while abs(x_diff - self.x_dist) > 0.0001:

            # Find range and ray angles along path
            x_diff, thetas, r = snell_law(theta_new, SS, z)

            # Newton-Raphson method
            t2 = theta_new
            x2 = x_diff

            theta_new = t2 + (self.x_dist / ((x2 - x1) / (t2 - t1))) - \
                (x2 / ((x2 - x1) / (t2 - t1)))
            
            t1 = t2                 # update theta and x distance
            x1 = x2                 # for next N-R correction
        

        # Now that the correct launch angle is found we can calculate
        # the travel time and output its corresponding ray properties
        arc_dist = []           # distance in each layer
        tt = []                 # travel time in each layer
        for i in range(len(thetas)):
            arc_dist.append((z[i + 1] - z[i]) / cos(thetas[i])) 
            tt.append((z[i + 1] - z[i]) / (SS[i] * cos(thetas[i])))

        tot_arc = sum(arc_dist)         # total ray arc length
        tot_tt = sum(tt)                # total travel time
        SS_avg = np.mean(SS)            # vertically averaged SS profile

        # Package data to return
        ray_path_info = {
            'arc_dist': arc_dist,
            'tot_arc_len': tot_arc,
            'theta0': t2,
            'SS_pro': SS,
            'z_pro': z,
            'SS_pro_avg': SS_avg,
            'surface_dist': r,
            'tt': tot_tt,
            'theta_pro': thetas[1:]
        }
        return ray_path_info


####### Sample input given below #######
# test = Ray_Tracing(500, -6.5, 22.5, 45, 22, -158, -4729.92 - 6.96)
# SS, z = test.ctd_data()
# SS_flt, z_flt = test.earth_flat_transform(SS, z)
# ray_info = test.ray_trace(SS_flt, z_flt)
# print(ray_info)


# import scipy.io as sio
# SS_matlab = sio.loadmat('SS_sample.mat')['SS_elps']
# SS_matlab = [val for sub_l in SS_matlab for val in sub_l]
# print(SS_matlab)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# plt_sample = ax.plot(SS_matlab - SS, z)
# plt.xlabel('Difference (m/s)')
# plt.ylabel('Depth (m)')
# plt.title('SS Profile Difference (Matlab - Python)')
# plt.grid(True)
# plt.gca().invert_yaxis()
# plt.show()

