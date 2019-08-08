#! python3
''' guassian_covariance_matrix.py - Function to create a Guassian model 
error covariance matrix given SS and rx position uncertainty '''

from geographiclib.geodesic import Geodesic
import numpy as np
from math import exp

def gauss_cov_matrix(SS_uncertain, pos_uncertain, center_x, center_y, len_scale):

    # Pre-allocate covariance matrix
    num_rows_cols = len(center_x) * len(center_y)
    Cm = np.zeros((num_rows_cols, num_rows_cols))


    # X and Y matrix to loop over (contains all coordinate matches)
    x_mat = center_x * len(center_y)
    y_mat = [[y] * len(center_x) for y in center_y]
    y_mat = [sub_y for y in y_mat for sub_y in y]       # flatten list
    
    for num1, (x_pos1, y_pos1) in enumerate(zip(x_mat, y_mat)):
        for num2, (x_pos2, y_pos2) in enumerate(zip(x_mat, y_mat)):

            # Distance between pixel centers
            pix_dist = Geodesic.WGS84.Inverse(y_pos1, x_pos1, y_pos2, x_pos2)['s12']

            # Guassian covariance matrix
            Cm[num1][num2] = (SS_uncertain ** 2) * \
                exp(-(pix_dist ** 2) / (len_scale ** 2))
    

    # Add rx hydrophone position uncertainty to model error matrix (Cm)
    Cm_pos = np.diag(np.array(pos_uncertain) ** 2)
    Cm = np.vstack((Cm, np.zeros((3, len(Cm[0])))))
    Cm = np.hstack((Cm, np.zeros((len(Cm), 3))))
    Cm[-3:, -3:] = Cm_pos


    return Cm




####### Sample input given below #######

# center_x = [-158.25797648860953, -158.20761839088763, -158.15726029316573, 
# -158.1069021954438, -158.05654409772194, -158.006186, -157.9558279022781, 
# -157.90546980455622, -157.8551117068343, -157.8047536091124, -157.7543955113905]
# center_y = [22.505215879296898, 22.551926277897934, 22.598636676498977, 
# 22.645347075100013, 22.692057473701055, 22.73876787230209, 22.785478270903134, 
# 22.83218866950417, 22.878899068105213, 22.92560946670625, 22.97231986530729]
# print(gauss_cov_matrix(4.5, [20, 20, 5], center_x, center_y, 10_000))

