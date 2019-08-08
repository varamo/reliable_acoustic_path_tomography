#! python3
''' SS_position_RAP_inversion.py - Script that takes in experimental acoustic
data collected by the Kilo Moan at the ALOHA Cabled Observatory (ACO) and 
performs Reliable Acoustic Path (RAP) Tomography to compute for a Sound Speed
field and seafloor mounted hydrophone position correction. This script is
to be used with "ray_trace_w_earth_flattening.py", "grid_space_distance.py". '''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from math import sqrt, pi
from geographiclib.geodesic import Geodesic
from ray_trace_w_earth_flattening import Ray_Tracing
from grid_space_distance import pixel_distance
from guassian_covariance_matrix import gauss_cov_matrix

# Finds the center of a pixel given the tick marks
def pixel_center(ticks):
    center_pix = []
    for pix in range(len(ticks) - 1):
        center_pix.append((ticks[pix] + ticks[pix + 1]) / 2)
    return center_pix


# Empirical Orthogonal Function to provide z-dependence on transmission
# (derived from previous HOTS CTD casts)
EOF_MODE_1 = sio.loadmat('EOF_mode_1.mat')['EOF_mode']
EOF_MODE_1 = [val for sub_l in EOF_MODE_1 for val in sub_l]     # make flat list
EOF_MODE_1 = np.append(EOF_MODE_1, [EOF_MODE_1[-1]] * 4)        # ensure sufficient EOF length
EOF_LAMBDA_1 = sio.loadmat('EOF_lambda_1.mat')['EOF_lambda']
########### EOF FILL - EDIT #################


# Current estimated position of hydrophone
N_mean = 2.30               # Geoid height at lat/lon
HEM_lat = 22.738772
HEM_lon = -158.006186
HEM_z = -4729.92 - N_mean


# Load experimental data
RAP_data = sio.loadmat('RAP_test_HEM_flat_earth.mat')['RAP_test'][0][0]
t_act   =   RAP_data['t_act'][0]        # Reception time
t_tx    =   RAP_data['t_tx'][0]         # Transmission time
t_diff1 =   RAP_data['t_diff'][0]       # Actual vs. estimated tt (Matlab)
tx_lon  =   RAP_data['lon'][0]          # Longitude of vessel
tx_lat  =   RAP_data['lat'][0]          # Latitude of vessel
# x_dist  =   RAP_data['x_dist'][0]       # Surface distance from hydrophone
z_tx    =   RAP_data['z'][0]            # Elevation of vessel
heading =   RAP_data['heading'][0]      # Heading of vessel
x_err   =   RAP_data['lon_err'][0]      # Longitude uncertainty
y_err   =   RAP_data['lat_err'][0]      # Latitude uncertatinty
z_err   =   RAP_data['z_err'][0]        # Elevation uncertainty


# Initialize variables
num_tx = len(t_act)         # Number of transmissions
g_bound = 28450             # Grid bound size for area of interest

# Get lat/lon boundaries for area of interest
max_lat = Geodesic.WGS84.Direct(HEM_lat, HEM_lon, 0, g_bound)['lat2']
max_lon = Geodesic.WGS84.Direct(HEM_lat, HEM_lon, 90, g_bound)['lon2']
min_lat = Geodesic.WGS84.Direct(HEM_lat, HEM_lon, 180, g_bound)['lat2']
min_lon = Geodesic.WGS84.Direct(HEM_lat, HEM_lon, 270, g_bound)['lon2']

# Split area up into evenly spaced grid points
n_cols = 11
n_rows = 11
x_ticks = np.linspace(min_lon, max_lon, n_cols + 1)     # Pixel Ticks
y_ticks = np.linspace(min_lat, max_lat, n_rows + 1)     # Pixel Ticks
center_x = pixel_center(x_ticks)            # Pixel Center
center_y = pixel_center(y_ticks)            # Pixel Center


# Vessel distance, bearing, and angle relative to hydrophone for each tx
x_dist = []
azmth = []
ves_brng = []
for lat, lon in zip(tx_lat, tx_lon):
    x_dist.append(Geodesic.WGS84.Inverse(lat, lon, HEM_lat, HEM_lon)['s12'])
    azmth.append(Geodesic.WGS84.Inverse(lat, lon, HEM_lat, HEM_lon)['azi1'])
    ves_brng.append(90 - azmth[-1])
ves_brng = np.array(ves_brng) * pi / 180        # convert to radians



# num_tx = 10
# Pre-allocation of SS observation matrix (G_SS) and other variables
G_SS = np.zeros((len(center_y) * len(center_x), num_tx), dtype=float)
ray_angles = np.zeros((2, num_tx))          # save tx and rx ray angle
est_tt2 = np.zeros(num_tx)                   # estimated tt given surface range

# Loop through all transmissions to solve for the SS observation matrix
SS_end = []                                 # SS where rx was made 
for t_num in range(num_tx):
# for t_num in range(10):

    # Compute ray tracing to determine inital launch angle of transmission
    # and ray arc lengths for each depth layer
    ray = Ray_Tracing(x_dist[t_num], z_tx[t_num], tx_lat[t_num], azmth[t_num], HEM_lat, HEM_lon, HEM_z)
    SS, z = ray.ctd_data()                      # CTD derived SS profile
    SS, z = ray.earth_flat_transform(SS, z)     # flat earth correction
    ray_info = ray.ray_trace(SS, z)             # ray tracing

    arc_lens                =   np.array(ray_info['arc_dist'])
    SS_avg                  =   ray_info['SS_pro_avg']
    surf_dist               =   ray_info['surface_dist']
    thetas                  =   np.array(ray_info['theta_pro'])
    ray_angles[:, t_num]    =   thetas[0], thetas[-1]
    est_tt2[t_num]          =   ray_info['tt']
    SS_end.append(SS[-1])


    # Calculate the surface distance traveled in each pixel with respect
    # to the given x and y tick marks
    pix_dist, pix_num = pixel_distance(tx_lat[t_num], tx_lon[t_num], HEM_lat, HEM_lon, x_ticks, y_ticks)


    # Find arc lengths that correspond to total surface distance in each pixel
    pix_arc = []            # arc length for each pixel (related to pix_dist)
    arc_depth = [0]         # arc length depth begins at tx "surface"
    cur_pix = 0             # start at pixel with transmission
    tot_surf_dist = 0       # keep track of surf dist traveled
    tot_arc_dist = 0        # keep track of ray arc len traveled
    for z_lyr, (cur_dist, arc_len) in enumerate(zip(surf_dist, arc_lens)):

        # Check if current arc length reaches beyond current picel
        if tot_surf_dist + cur_dist >= pix_dist[cur_pix]:
            # Find proportion of current layer that is within pixel
            lyr_prop = (pix_dist[cur_pix] - tot_surf_dist) / cur_dist
            pix_arc.append(tot_arc_dist + (arc_len * lyr_prop))
            
            # Update arc/surface length, arc depth, and current pixel
            tot_arc_dist = arc_len * (1 - lyr_prop)
            tot_surf_dist = cur_dist * (1 - lyr_prop)
            arc_depth.append(z_lyr + 1)
            cur_pix += 1
        
        # Add to current arc/surface length
        else:
            tot_surf_dist += cur_dist
            tot_arc_dist += arc_len

    # Ensure pixel with reception has been registered
    if len(pix_arc) < len(pix_dist):
        arc_depth.append(z_lyr + 1)
        pix_arc.append(tot_arc_dist)
    assert len(pix_arc) == len(pix_dist)


    # Add information to the SS observation matrix (G_SS)
    for pixel in range(len(pix_arc)):
        G_SS[pix_num[pixel]][t_num] = sum(EOF_MODE_1[arc_depth[pixel]: (arc_depth[pixel + 1] - 1)] *
            (-arc_lens[arc_depth[pixel]: (arc_depth[pixel + 1] - 1)] / \
            (((SS[arc_depth[pixel]: (arc_depth[pixel + 1] - 1)])) ** 2)))


# Create rx hydrophone position observation matrix (G_pos)
G_x = np.cos(ves_brng[:num_tx]) * np.sin(ray_angles[-1][:]) / SS_end
G_y = np.sin(ves_brng[:num_tx]) * np.sin(ray_angles[-1][:]) / SS_end
G_z = np.cos(ray_angles[-1][:]) / SS_end
G_pos = np.column_stack((G_x, G_y, G_z))
########## TAKE OUT EDIT OF NUM_TX HERE TOO ############


# Combine the SS and hydrophone position observation matrices
G = np.column_stack((np.transpose(G_SS), G_pos))


# Model uncertainty (Cm) due to apriori SS field and hydrophone position (x,y,z)
SS_uncertainty = sqrt(EOF_LAMBDA_1)
pos_uncertainty = [100, 100, 5]

len_scale = 5_000           # Gaussian covariance length scale (meters)
Cm = gauss_cov_matrix(SS_uncertainty, pos_uncertainty, center_x, center_y, len_scale)


# Data uncertainty (Cd) due to rx time (from xcorrs, etc.) and vessel position uncertainty
t_uncertainty = 0.001
########## TAKE OUT EDIT OF NUM_TX HERE AND BEFORE LOOP ABOVE ############
x_anomaly = (((np.sin(ray_angles[0][:]) / SS_avg) * np.cos(ves_brng[:num_tx])) ** 2) * (x_err[:num_tx]) ** 2
y_anomaly = (((np.sin(ray_angles[0][:]) / SS_avg) * np.sin(ves_brng[:num_tx])) ** 2) * (y_err[:num_tx]) ** 2
z_anomaly = ((np.cos(ray_angles[0][:]) / SS_avg) ** 2) * (z_err[:num_tx]) ** 2
SS_anomaly = x_anomaly + y_anomaly + z_anomaly

Cd = np.diag(SS_anomaly + (t_uncertainty) ** 2)
#########################################################################



# Estimated arrival time (convert to matlab timing)
est_tt1 = (t_act[:num_tx] - t_tx[:num_tx]) * 3600 * 24 - t_diff1[:num_tx]          # matlab est tt
t_est2 = (t_tx[:num_tx] + (est_tt2  / (3600 * 24)))     # est arrival (Python)
t_diff2 = (t_act[:num_tx] - t_est2) * 3600 * 24         # actual vs. estimated tt (Python)
py_mat_diff = est_tt1 - est_tt2                         # tt estimate diff between matlab and python code


# Compute the generalized inverse (GI == Cm * G' / (G * Cm * G' + Cd))
a = Cm @ np.transpose(G)                    # a == Cm * G'
b = G @ Cm @ np.transpose(G) + Cd           # b == G * Cm * G' + Cd
GI = a @ np.linalg.pinv(b)

# SS and hydrophone position perturbation
d1 = t_diff1[:num_tx]                         # data vector
m1 = GI @ d1
m1_SS = m1[:-3]
m1_pos = m1[-3:]
# print(m_SS, m_pos)








################################ FIGURES ################################


x_dist = np.array(x_dist) / 1000    # convert surface distance into km
t_diff1 = t_diff1 * 1000            # convert tt perturbations into ms
t_diff2 = t_diff2 * 1000            # convert tt perturbations into ms

# Difference between Matlab and Python travel times
fig, ax = plt.subplots()
mat_t_plt = ax.scatter(x_dist[:num_tx], t_diff1[:num_tx], color='b')
py_t_plt = ax.scatter(x_dist[:num_tx], t_diff2, color='r')
ax.legend((mat_t_plt, py_t_plt), ('Matlab', 'Python'))
plt.title('TT Perturbations (HEM)')
plt.xlabel('Range (km)')
plt.ylabel('(ms)')
plt.grid(True)
plt.show()


# Matlab and Python travel times w.r.t. range
# fig, ax = plt.subplots()
# mat_t_plt = ax.scatter(x_dist[:num_tx], est_tt1[:num_tx], color='b')
# py_t_plt = ax.scatter(x_dist[:num_tx], est_tt2, color='r')
# ax.legend((mat_t_plt, py_t_plt), ('Matlab', 'Python'))
# plt.title('Travel times (HEM)')
# plt.xlabel('Range (km)')
# plt.ylabel('(s)')
# plt.grid(True)
# plt.show()

# Difference in Matlab and Python tt w.r.t. range
fig, ax = plt.subplots()
mat_t_plt = ax.scatter(x_dist[:num_tx], est_tt1[:num_tx] - est_tt2)
plt.title('Difference between Matlab and Python TT (HEM)')
plt.xlabel('Range (km)')
plt.ylabel('(s)')
plt.grid(True)
plt.ylim(min(est_tt1[:num_tx] - est_tt2) - 0.0001, max(est_tt1[:num_tx] - est_tt2) + 0.0001)
plt.show()



