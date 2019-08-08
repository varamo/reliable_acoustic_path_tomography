#! python3
''' Function used to calculate the surface distance traveled by an acoustic
transmission in each pixel of a provided grid space. '''

from geographiclib.geodesic import Geodesic
def pixel_distance(tx_lat, tx_lon, rx_lat, rx_lon, x_ticks, y_ticks):
    

    # Function to find position along transmission where it crosses over
    # the given x or y tick mark (check_lat=True for y tick)
    def find_cross(xing_coord, tx_coord, check_lat):
        x1 = xing_coord - tx_coord
        t1 = 0
        coord_new = 0
        dist_new = 1000

        # Loop through until crossing coordinate is found using N-R method
        while abs(coord_new - xing_coord) > 0.00001:

            # Check if lat or lon value was input
            if check_lat:
                coord_new = tx_line.Position(dist_new)['lat2']
            else:
                coord_new = tx_line.Position(dist_new)['lon2']

            # Newton-Raphson method
            t2 = dist_new
            x2 = coord_new

            dist_new = t2 + (xing_coord / ((x2 - x1) / (t2 - t1))) - \
                (x2 / ((x2 - x1) / (t2 - t1)))
            
            t1 = t2                 # update theta and x distance
            x1 = x2                 # for next N-R correction

        return dist_new


    # Find x and y ticks in between tx and rx
    lat_xing = []
    lon_xing = []
    for lat, lon in zip(y_ticks, x_ticks):
        # Check if current latitude is between tx and rx
        cond1_lat = (lat > rx_lat) and (lat < tx_lat)
        cond2_lat = (lat < rx_lat) and (lat > tx_lat)
        if cond1_lat or cond2_lat:
            lat_xing.append(lat)
        # Check if current longitude is between tx and rx
        cond1_lon = (lon > rx_lon) and (lon < tx_lon)
        cond2_lon = (lon < rx_lon) and (lon > tx_lon)
        if cond1_lon or cond2_lon:
            lon_xing.append(lon)

    # Starting pixel row and column
    num_col = len(x_ticks) - 1
    for i, y_tick in enumerate(y_ticks[1:]):
        # Find first occurence (loops from smaller to larger values)
        if tx_lat < y_tick:
            pix_row = i
            break
    for i, x_tick in enumerate(x_ticks[1:]):
        # Find first occurence
        if tx_lon < x_tick:
            pix_col = i
            break
    cur_pix = pix_col + (pix_row * num_col)


    # Create geodesic line between tx and rx
    tx_line = Geodesic.WGS84.InverseLine(tx_lat, tx_lon, rx_lat, rx_lon)
    tot_dist = Geodesic.WGS84.Inverse(tx_lat, tx_lon, rx_lat, rx_lon)['s12']

    # Check to see if there were not any x or y tick crossings
    cond1 = not lat_xing
    cond2 = not lon_xing
    if cond1 and cond2:
        # No crossings: total distance traveled is within rx pixel
        return [tot_dist], [cur_pix]


    # Find current distance and pixel location for each tick mark crossing
    crossing_data = {'surf_dist': [], 'pix_row': [], 'pix_col': []}
    for lat in lat_xing:
        x_dist = find_cross(lat, tx_lat, True)
        crossing_data['surf_dist'].append(x_dist)
        crossing_data['pix_row'].append(pix_row)
        crossing_data['pix_col'].append(-1)         # Unknown col
        # Advance to next row
        if lat < rx_lat:
            pix_row += 1
        else:
            pix_row -= 1
    for lon in lon_xing:
        x_dist = find_cross(lon, tx_lon, False)
        crossing_data['surf_dist'].append(x_dist)
        crossing_data['pix_col'].append(pix_col)
        crossing_data['pix_row'].append(-1)         # Unknown row
        # Advance to next column
        if lon < rx_lon:
            pix_col += 1
        else:
            pix_col -= 1
    

    # Sort crossings by distance from transmission (sort by surf_dist)
    list_sorted = list(zip(*sorted(zip(crossing_data['surf_dist'], crossing_data['pix_row'], crossing_data['pix_col']))))
    x_dist, pix_row, pix_col = list_sorted


    # Find corresponding pixel num for each crossing
    pix_num = [cur_pix]                     # starting pixel
    for row, col in zip(pix_row, pix_col):
        if row != -1:
            # Increase or decrease pixel number entire row
            if lat < rx_lat:
                cur_pix += num_col
            else:
                cur_pix -= num_col
        elif col != -1:
            # Increase or decrease pixel number by column
            if lon < rx_lon:
                cur_pix += 1
            else:
                cur_pix -= 1
        pix_num.append(cur_pix)
    
    # Convert distances to list and add final pixel
    x_dist = list(x_dist)
    x_dist.append(tot_dist)


    # Calculate distance for each pixel by subtracting earlier distance
    pix_dist = [y - x for x, y in zip(x_dist, x_dist[1:])]
    pix_dist.insert(0, x_dist[0])

    
    return pix_dist, pix_num




####### Sample input given below #######

# x_ticks = [-158.28315554, -158.23279744, -158.18243934, -158.13208124, 
# -158.08172315, -158.03136505, -157.98100695, -157.93064885, -157.88029076,
# -157.82993266, -157.77957456, -157.72921646]
# y_ticks = [22.48186068, 22.52857108, 22.57528148, 22.62199188, 22.66870227,
# 22.71541267, 22.76212307, 22.80883347, 22.85554387, 22.90225427,
# 22.94896467, 22.99567506]
# tx_lat = 22.5367083249615
# tx_lon = -158.114860262648
# rx_lat = 22.738772
# rx_lon = -158.006186
# print(pixel_distance(tx_lat, tx_lon, rx_lat, rx_lon, x_ticks, y_ticks))





