import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot the z-axis density frequency of the depth maps')
parser.add_argument('-fp', type=str, help='Path to the folder containing the .npy files')
parser.add_argument('-gt', type=str, help='Path to the folder containing the ground truth .npy files')
parser.add_argument('-s', type=int, help='Start index of the file list (for tests)', default=0)
parser.add_argument('-i', type=int, help='Index of the file to plot', default=0)
args = parser.parse_args()

folder_path = '/dartfs-hpc/rc/home/r/f006gmr/gaussian-splatting/' + args.fp + '/train/ours_' + str(args.i) + '/z_density'
folder_path_gt = '/dartfs-hpc/rc/home/r/f006gmr/gaussian-splatting/' + args.gt + '/depth'
save_folder_path = '/dartfs-hpc/rc/home/r/f006gmr/gaussian-splatting/' + args.fp + '/train/ours_' + str(args.i) + '/z_compare'

# Create the save folder if it doesn't exist
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# Get a list of all .npy files in the folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
# Sort the list of files to ensure consistent ordering
file_list.sort()

# Iterate over each file and plot its contents
for file in file_list:
    num = int(file.split('.')[0])
    num += args.s
    new_file = str(num).zfill(4) + '.npy'
    gt_data = np.load(os.path.join(folder_path_gt, new_file))
    print('Ground Truth Max:', np.max(gt_data), 'Min:', np.min(gt_data))

    # Clip the data to [0, 8]
    # gt_data = np.clip(data, 0, 8)
    
    
    # BLOCK: predicted by GS
    file_path = os.path.join(folder_path, file)
    pred_data = np.load(file_path)
    # pred_data = np.clip(data, 0, 8)
    print('Predicted Max:', np.max(pred_data), 'Min:', np.min(pred_data))

    loss = np.mean(np.abs(gt_data - pred_data))

    # Plot two hotmaps in one figure
    # Size 21:9
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(21, 9)
    ax[0].imshow(gt_data, cmap='hot', interpolation='nearest', vmin=0, vmax=8)
    ax[0].set_title('Ground Truth Depth')
    ax[1].imshow(pred_data, cmap='hot', interpolation='nearest', vmin=0, vmax=8)
    ax[1].set_title('Predicted Depth by GS')
    plt.suptitle('L1 Loss: ' + str(loss))

    # colorbar from 0 to 8
    cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ax[1].imshow(pred_data, cmap='hot', interpolation='nearest', vmin=0, vmax=8), cax=cax)
    cbar.set_label('Depth Value (m)')

    # Save the figure with origin file name
    plt.savefig(os.path.join(save_folder_path, os.path.splitext(file)[0] + '.png'))
    plt.close()
