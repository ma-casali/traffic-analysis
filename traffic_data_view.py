import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib
import time

standard_car_length = 15 # ft

traffic_data = pd.read_csv(
    'data/ngsim/trajectories-0750am-0805am.txt',
      sep=r"\s+",
        header=None
        )

traffic_data.columns = [
    "Vehicle_ID", 
    "Frame_ID",
    "Total_Frames", # [100 ms]
    "Global_Time", #[ms]
    "Local_X", # [ft]
    "Local_Y", # [ft]
    "Global_X", # [ft]
    "Global_Y", # [ft]
    "v_Length", # [ft]
    "v_Width", # [ft]
    "v_Class",
    "v_Vel", # [ft/s]
    "v_Acc", # [ft/s^2]
    "Lane_ID",
    "Preceding",
    "Following",
    "Spacing", # [ft]
    "Headway" # [s]
]

# number of lanes
num_lanes = len(np.unique(traffic_data['Lane_ID']))

[v_ID, v_ID_ind] = np.unique(traffic_data['Vehicle_ID'], return_index=True)\

# make a  num_vehicle x 2 x len(t) array
t = np.arange(np.min(traffic_data['Global_Time']), np.max(traffic_data['Global_Time'])+100, 100)
vehicle_pos = np.full((2, len(t), len(v_ID)), np.nan)

vehicle_dict = {}
for iter in range(len(v_ID)):
    vehicle_dict[str(v_ID[iter])] = traffic_data[:][v_ID_ind[iter-1]:v_ID_ind[iter]]
    # match the times up to t and fill in an array with those times
    vehicle_t = traffic_data['Global_Time'][v_ID_ind[iter-1]:v_ID_ind[iter]]
    vehicle_pos[0, np.isin(t, vehicle_t), iter] = traffic_data['Global_X'][v_ID_ind[iter-1]:v_ID_ind[iter]]
    vehicle_pos[1, np.isin(t, vehicle_t), iter] = traffic_data['Global_Y'][v_ID_ind[iter-1]:v_ID_ind[iter]]

# rotate the coordinates so that the road is aligned with the x-axis
# find the angle of the road by looking at the first and last position of the middle car
middle_index = len(v_ID) // 2
# first and last time index of middle car
t1 = np.where(~np.isnan(vehicle_pos[0, :, middle_index]))[0][0]
t2 = np.where(~np.isnan(vehicle_pos[0, :, middle_index]))[0][-1]
x1 = vehicle_pos[0, t1, middle_index]
y1 = vehicle_pos[1, t1, middle_index]
x2 = vehicle_pos[0, t2, middle_index]
y2 = vehicle_pos[1, t2, middle_index]
angle = np.arctan2(y2 - y1, x2 - x1)
rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
for i in range(len(v_ID)):
    for j in range(len(t)):
        if not np.isnan(vehicle_pos[0, j, i]):
            vehicle_pos[:, j, i] = rotation_matrix @ vehicle_pos[:, j, i]

# extrapolate position of middle car to provide reference for others
middle_index = len(v_ID) // 2
avg_vehicle_pos = [None, None]
avg_vehicle_pos[0] = scipy.interpolate.interp1d(t[~np.isnan(vehicle_pos[0, :, middle_index])], vehicle_pos[0, ~np.isnan(vehicle_pos[0, :, middle_index]), middle_index], fill_value='extrapolate')
avg_vehicle_pos[1] = scipy.interpolate.interp1d(t[~np.isnan(vehicle_pos[1, :, middle_index])], vehicle_pos[1, ~np.isnan(vehicle_pos[1, :, middle_index]), middle_index], fill_value='extrapolate')

vehicle_pos_norm = np.copy(vehicle_pos)
for i in range(len(v_ID)):
    vehicle_pos_norm[0, :, i] = vehicle_pos[0, :, i] - avg_vehicle_pos[0](t)
    vehicle_pos_norm[1, :, i] = vehicle_pos[1, :, i] - avg_vehicle_pos[1](t)

# at each time index, subtract the mean position from all cars
for i in range(len(t)):
    mean_x = np.nanmean(vehicle_pos[0, i, :])
    mean_y = np.nanmean(vehicle_pos[1, i, :])
    vehicle_pos_norm[0, i, :] = vehicle_pos[0, i, :] - mean_x
    vehicle_pos_norm[1, i, :] = vehicle_pos[1, i, :] - mean_y

#region Local Density
# (rho_l: the number of vehicles per unit length in the local neighborhood of a vehicle)
preceding_distance = np.full((len(t), len(v_ID)), np.nan)
following_distance = np.full((len(t), len(v_ID)), np.nan)
rho_l = np.full((len(t), len(v_ID)), np.nan)
for i in range(len(v_ID)):
    vehicle_data = vehicle_dict[str(v_ID[i])]
    vehicle_data = vehicle_data.reset_index(drop=True)
    if len(vehicle_data['v_Vel']) == 0:
        vehicle_velocity = np.ones(len(t)) * np.nan
    else:
        vehicle_velocity = np.interp(t, vehicle_data['Global_Time'], vehicle_data['v_Vel'])

    preceding_vehicle_id = np.unique(vehicle_data['Preceding'])
    following_vehicle_id = np.unique(vehicle_data['Following'])

    for j in range(len(preceding_vehicle_id)):
        if preceding_vehicle_id[j] == 0:
            continue
        vehicle_p_data = vehicle_dict[str(preceding_vehicle_id[j])]
        vehicle_p_data = vehicle_p_data.reset_index(drop=True)
        time_vals, times, times_p = np.intersect1d(vehicle_data['Global_Time'], vehicle_p_data['Global_Time'], return_indices = True)
        preceding_distance[np.isin(t, time_vals), i] = np.sqrt((vehicle_data['Global_X'].iloc[times].values - vehicle_p_data['Global_X'].iloc[times_p].values)**2 + (vehicle_data['Global_Y'].iloc[times].values - vehicle_p_data['Global_Y'].iloc[times_p].values)**2)

    for j in range(len(following_vehicle_id)):
        if following_vehicle_id[j] == 0:
            continue
        vehicle_f_data = vehicle_dict[str(following_vehicle_id[j])]
        vehicle_f_data = vehicle_f_data.reset_index(drop=True)
        time_vals, times, times_f = np.intersect1d(vehicle_data['Global_Time'], vehicle_f_data['Global_Time'], return_indices = True)
        following_distance[np.isin(t, time_vals), i] = np.sqrt((vehicle_data['Global_X'].iloc[times].values - vehicle_f_data['Global_X'].iloc[times_f].values)**2 + (vehicle_data['Global_Y'].iloc[times].values - vehicle_f_data['Global_Y'].iloc[times_f].values)**2)

    # filter out where rho goes to infinity and set to nan
    rho_l[:, i] = (preceding_distance[:, i] + following_distance[:, i]) / standard_car_length / vehicle_velocity
    rho_l[np.isinf(rho_l[:, i]), i] = np.nan

#region Local Stiffness

#region Plotting The Relative Positions
# find first index where all vehicles have data
first_index = np.where(~np.isnan(vehicle_pos[0, :, middle_index]))[0][0]

fig, ax = plt.subplots()
xlims = [np.nanmin(vehicle_pos_norm[0, :, :]), np.nanmax(vehicle_pos_norm[0, :, :])]
ylims = [np.nanmin(vehicle_pos_norm[1, :, :]), np.nanmax(vehicle_pos_norm[1, :, :])]

for i in range(first_index, first_index + int(60/0.1)):
    ax.clear()
    ax.scatter(vehicle_pos_norm[0, i, :], vehicle_pos_norm[1, i, :], c=rho_l[i, :], cmap='viridis')
    ax.set_xlabel("Rotated Global_X [ft]")
    ax.set_ylabel("Rotated Global_Y [ft]")
    ax.set_title(f"Vehicle positions at time index {i}")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    plt.pause(0.1)  # pause to update the figure

plt.show()  # Keep the last frame open
