CS180 Project 1 - Colorizing the Prokudin-Gorskii Collection

This project implements image colorization by splitting BGR plates and aligning G/R channels to B using SSD/NCC metrics.

Files:
- colorize_skel.py: Main implementation script
- data/: Contains input images (.jpg and .tif files)
- out/: Contains output colorized images organized by trials

Usage:
python colorize_skel.py

The script automatically processes all images in the data folder and saves results to the out folder.

Important: You must adjust the hyperparameters, especially the TRIAL_NUM variable, to add new results to corresponding folders. The script creates trial-specific directories (e.g., out/trial_1/, out/trial_2/) based on the TRIAL_NUM setting.
