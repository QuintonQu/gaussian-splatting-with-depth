import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot the z-axis density frequency of the depth maps')
parser.add_argument('-fp', type=str, help='Path to the folder containing the .npy files')
parser.add_argument('-gt', type=str, help='Path to the folder containing the ground truth .npy files')
parser.add_argument('-s', type=int, help='Start index of the file list (for tests)', default=0)
args = parser.parse_args()

folder_path_gt = './' + args.gt + '/depth'
# folder_path_gt = './' + args.gt + '/Sonar_raw'
if args.fp is None:
    folder_path = None
    print("No prediction folder path is given")
    save_folder_path = './' + args.gt + '/z_compare'
else:
    folder_path = './' + args.fp + '/train/ours_30000/z_density'
    save_folder_path = './' + args.fp + '/train/ours_30000/z_compare'

# Create the save folder if it doesn't exist
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# Get a list of all .npy files in the folder
if folder_path is not None:
    file_list = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
else:
    file_list = [file for file in os.listdir(folder_path_gt) if file.endswith('.npy')]
# Sort the list of files to ensure consistent ordering
file_list.sort()

# Iterate over each file and plot its contents
for file in file_list:
    # num = int(file.split('.')[0])
    # num += args.s
    # new_file = str(num).zfill(4) + '.npy'
    new_file = file

    # Flatten the data
    data = np.load(os.path.join(folder_path_gt, new_file))
    flattened_data = data.flatten()
    # data = data / np.max(data)

    # Calculate the histogram
    bin_edges = np.linspace(3, 8, 201)
    hist, bin_edges = np.histogram(flattened_data, bins=bin_edges)

    # Normalize the histogram to [0, 1] minmax
    hist = hist / np.max(hist)
    # bin_edges = np.linspace(0, 8, 201)

    # Plot the histogram as a line graph
    plt.plot(bin_edges[:-1], hist, label="gt_depth")
    
    # BLOCK: predicted by GS
    if folder_path is not None:
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path)

        # Normalize the data to [0, 1] minmax
        # data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = np.sum(data, axis=1)
        data = data / np.max(data)

        plt.plot(bin_edges[:-1], data, label="predicted_by_gs")

    plt.xlabel('Z-axis')
    plt.ylabel('Sum of Density (Normalized)')
    plt.title('Line Graph of Z-axis Density Frequency')
    
    plt.legend()
    # Save the figure with origin file name
    plt.savefig(os.path.join(save_folder_path, os.path.splitext(file)[0] + '.png'))
    plt.close()
