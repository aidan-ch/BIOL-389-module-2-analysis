import math

import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import pandas as pd
import os


class Trial:
    def __init__(self, csv_path, trial_num):
        self.hist_data = {
            "Data File Name": list(),
            "is_density": list(),
            "Counts/Densities": list(),
            "Edges": list()
        }
        self.trial_num = trial_num
        self.df = pd.read_csv(csv_path, dtype=np.float32)


trial_1_data = "BIOL 389 - module 2 data/test_attracted_odorant_light/test0_larvae1.csv"
trial_2_data = "BIOL 389 - module 2 data/test_attracted_odorant_light/test1_larvae1.csv"

# coordinates of the artifical target
target_x = 420
target_y = 120

legend_fontsize = 7
subplot_title_fontsize = 8
figure_title_fontsize = "x-large"


def initialize_directories():
    if not os.path.isdir("Plots/Histograms"):
        os.makedirs("Plots/Histograms")

    if not os.path.isdir("Plots/Trajectories"):
        os.makedirs("Plots/Trajectories")

    if not os.path.isdir("Plots/Velocity plots"):
        os.makedirs("Plots/Velocity plots")

    if not os.path.isdir("Plots/Distance plots"):
        os.makedirs("Plots/Distance plots")

    if not os.path.isdir("Extracted Data"):
        os.makedirs("Extracted Data")

    if not os.path.isdir("Extracted Data/Histogram Data"):
        os.makedirs("Extracted Data/Histogram Data")

    if not os.path.isdir("Extracted Data/Velocity Data"):
        os.makedirs("Extracted Data/Velocity Data")



def create_smoothed_bearing_histogram(
        trial, subplot, segment, is_density: bool,
        my_color="blue"):
    smoothing_window = 5

    if segment == "whole":
        arr_bearing = trial.df["Bearing"].rolling(window=smoothing_window,
                                                  min_periods=1,
                                                  center=True).mean()
    else:
        frame_closest = trial.df["Distance"].idxmin()

        if segment == "early":
            arr_bearing = trial.df.iloc[:frame_closest]["Bearing"].rolling(window=smoothing_window,
                                                                           min_periods=1,
                                                                           center=True).mean()
        elif segment == "late":
            arr_bearing = trial.df.iloc[frame_closest + 1:]["Bearing"].rolling(window=smoothing_window,
                                                                               min_periods=1,
                                                                               center=True).mean()
        # segment is not a valid string
        else:
            raise ValueError("Invalid segment argument provided ("
                             "expected whole, early, or late)")

    trial_num = trial.trial_num
    num_frames = arr_bearing.shape[0]

    # bin_width is in degrees
    bin_width = 15
    bins = np.arange(0, 181, bin_width)

    counts, edges, patches = subplot.hist(
        arr_bearing,
        bins=bins,
        density=is_density,
        ec="white",
        fc=my_color,
        label="Trial {} ({} frames)".format(trial_num, num_frames),
        alpha=0.5)

    if is_density:
        style_histogram_subplot(subplot, "Probability Density")
    else:
        style_histogram_subplot(subplot, "Frequency (frames)")

    return counts, edges, patches


def add_subplot_legend(subplot):
    subplot.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.087),
        fontsize=legend_fontsize,
        framealpha=0.2,
        ncol=1,
        borderpad=0.3,
        labelspacing=0.3,
        handletextpad=0.4
    )


def style_histogram_subplot(subplot, ylabel):
    bin_width = 15
    subplot.set(
        xlim=[0, 180],
        xticks=range(0, 181, bin_width * 2),
        xlabel="Degrees",
        ylabel=ylabel
    )
    subplot.tick_params(axis="x", labelrotation=45)
    subplot.tick_params(axis="both", labelsize=9)


# hist_data = tuple ( counts, edges, patches), returned by any of the histogram generating functions in this program
# saves bin_start, bin_end, counts/densities in a csv
def save_histogram_data(trial):
    data_dict = trial.hist_data

    for i, file_name in enumerate(data_dict["Data File Name"]):
        counts = data_dict["Counts/Densities"][i]
        edges = data_dict["Edges"][i]
        is_density = data_dict["is_density"][i]

        if is_density:
            hist_type = "Density"
        else:
            hist_type = "Frequency"

        hist_df = pd.DataFrame({
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            hist_type: counts.astype(float)
        })

        hist_df.to_csv(
            "Extracted Data/Histogram Data/{}.csv".format(file_name),
            index=False)


# maybe generate moving average?

def trajectory_plot(trial):
    x_arr = np.array(trial.df["X"], dtype=np.float32)
    y_arr = np.array(trial.df["Y"], dtype=np.float32)
    trial_number = trial.trial_num
    plt.figure()

    plt.plot(x_arr, y_arr, marker='o', markersize=1, c="black")  # path

    # Mark start point (green)
    plt.scatter(x_arr[0], y_arr[0], label="Start", c="green")

    # Mark end point (red)
    plt.scatter(x_arr[-1], y_arr[-1], label="End", c="orange")

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Larval Trajectory - Trial {}".format(trial_number))

    # mark the target
    plt.scatter(target_x, target_y, marker='x', s=100,
                label="Target (420, 120)")

    plt.legend()
    plt.axis("equal")  # keeps scale consistent

    plt.savefig(fname="Plots/Trajectories/Trajectory_{}".format(trial_number))
    plt.close()


# return numpy array of average velocity over non-overlapping windows
# POSITIVE VELOCITY = MOVING TOWARDS THE TARGET
# BUT WILL BE NORMALIZED TO [-1, 1] SINCE VELOCITY UNITS ARE WEIRD (pixels? vs. time)
def velocity_v_time(df):
    distance = df["Distance"].to_numpy(dtype=float)

    num_frames = distance.shape[0]

    # video is 5Hz, frame duration is therefore 200ms or 0.2s
    frame_duration = 0.2

    # window = interval (in frames) over which we are calculating our average velocity for a given time point
    # window MUST BE GREATER THAN 1
    # e.g. if window = 2, then Velocity at t = 0s is computed using the displacement from frame 1 to frame 2 (i.e. looking at 2 frames, the window)
    window = 5
    window_duration = window * frame_duration

    # cant get velocity for the last incomplete window
    velocity_arr = np.empty(shape=(num_frames - 1) // window)
    i = 0

    while i < len(velocity_arr):
        dist_before = distance[i * window]
        dist_after = distance[(i + 1) * window]

        # doing steps of 1s
        # before - after b/c positive velocity = distance is decreasing
        velocity = (dist_before - dist_after) / window_duration
        velocity_arr[i] = velocity

        i += 1

    # normalize by dividing all elements by maximum velocity
    max_velocity = np.max(np.abs(velocity_arr))

    i = 0
    while i < len(velocity_arr):
        velocity_arr[i] = velocity_arr[i] / max_velocity
        i += 1

    return velocity_arr, window


def velocity_plot(trial):
    df = trial.df
    trial_num = trial.trial_num
    velocity_arr, window = velocity_v_time(df)

    fig, ax = plt.subplots()

    frame_duration = 0.2
    total_time = velocity_arr.shape[0] * window * frame_duration

    time_arr = np.arange(0, velocity_arr.shape[0], 1) * window * frame_duration
    ax.scatter(time_arr, velocity_arr, c="green", marker=".", s=0.5)

    ax.set(title="Velocity (normalized) - Trial {} vs. Time".format(trial_num),
           xlabel="Time (s)",
           ylabel="Radial Velocity (normalized)")

    # tick every 15 seconds, label every 60 seconds
    ax.set_xticks(np.arange(0, total_time + 1, 60))
    ax.set_xticks(np.arange(0, total_time + 1, 15), minor=True)

    # plot the location where we were closest to the target
    arr_distance = df["Distance"]
    frame_smallest_distance = arr_distance.idxmin()
    time_smallest_distance = frame_smallest_distance * frame_duration

    ax.axvline(x=time_smallest_distance, linestyle="--",
               label="Closest to target")

    # show the label of vertical line when closest to target
    ax.legend()
    plt.savefig("Plots/Velocity plots/Velocity - Trial {}".format(trial_num))
    plt.close()

    return velocity_arr, time_arr


def save_velocity_data(v_arr, t_arr, file_name):
    v_df = pd.DataFrame({
        "Time (s)": t_arr,
        "Velocity": v_arr.astype(float)
    })

    v_df.to_csv(
        "Extracted Data/Velocity Data/{}.csv".format(file_name),
        index=False)


def generate_smooth_bearing_histograms(trial_1, trial_2):
    # we are saving the data from each histogram separately in "trial".hist_data attribute but saving images in groups of 3 histograms

    # 1x3 array of plots (trial 1 and 2 overlayed for each) --> [ Whole trial, Until min distance, after min distance ]
    fig_frequency, ax_frequency = plt.subplots(
        1,
        3,
        layout="constrained")

    # subplot customizations + initialization
    fig_frequency.suptitle("Bearing Distribution Across Trial Segments",
                           fontweight='bold',
                           fontsize=figure_title_fontsize)
    ax_frequency[0].set_title(
        "Whole Trial",
        fontsize=subplot_title_fontsize
    )
    ax_frequency[1].set_title(
        "Until min(distance)",
        fontsize=subplot_title_fontsize
    )
    ax_frequency[2].set_title(
        "After min(distance)",
        fontsize=subplot_title_fontsize
    )

    fig_density, ax_density = plt.subplots(
        1,
        3,
        layout="constrained")

    fig_density.suptitle(
        "Bearing Density Distribution Across Trial Segments",
        fontweight='bold', fontsize=figure_title_fontsize
    )

    ax_density[0].set_title(
        "Whole Trial",
        fontsize=subplot_title_fontsize
    )
    ax_density[1].set_title(
        "Until min(distance)",
        fontsize=subplot_title_fontsize
    )
    ax_density[2].set_title(
        "After min(distance)",
        fontsize=subplot_title_fontsize
    )

    # WHOLE TRIAL
    counts, edges = create_smoothed_bearing_histogram(
        trial_1,
        ax_frequency[0],
        "whole",
        is_density=False,
        my_color="red"
    )[:-1]
    trial_1.hist_data["Data File Name"].append(
        "Bearing counts (whole trial) - Trial 1")
    trial_1.hist_data["is_density"].append(False)
    trial_1.hist_data["Counts/Densities"].append(counts)
    trial_1.hist_data["Edges"].append(edges)

    counts, edges = create_smoothed_bearing_histogram(
        trial_2,
        ax_frequency[0],
        "whole",
        is_density=False,
        my_color="blue"
    )[:-1]
    trial_2.hist_data["Data File Name"].append(
        "Bearing counts (whole trial) - Trial 2")
    trial_2.hist_data["is_density"].append(False)
    trial_2.hist_data["Counts/Densities"].append(counts)
    trial_2.hist_data["Edges"].append(edges)

    # UNTIL MIN
    counts, edges = create_smoothed_bearing_histogram(
        trial_1,
        ax_frequency[1],
        "early",
        is_density=False,
        my_color="red"
    )[:-1]

    trial_1.hist_data["Data File Name"].append(
        "Bearing counts (until min(distance)) - Trial 1")
    trial_1.hist_data["is_density"].append(False)
    trial_1.hist_data["Counts/Densities"].append(counts)
    trial_1.hist_data["Edges"].append(edges)

    counts, edges = create_smoothed_bearing_histogram(
        trial_2,
        ax_frequency[1],
        "early",
        is_density=False,
        my_color="blue"
    )[:-1]

    trial_2.hist_data["Data File Name"].append(
        "Bearing counts (until min(distance)) - Trial 2")
    trial_2.hist_data["is_density"].append(False)
    trial_2.hist_data["Counts/Densities"].append(counts)
    trial_2.hist_data["Edges"].append(edges)

    # AFTER MIN
    counts, edges = create_smoothed_bearing_histogram(
        trial_1,
        ax_frequency[2],
        "late",
        is_density=False,
        my_color="red"
    )[:-1]
    trial_1.hist_data["Data File Name"].append(
        "Bearing counts (after min(distance)) - Trial 1")
    trial_1.hist_data["is_density"].append(False)
    trial_1.hist_data["Counts/Densities"].append(counts)
    trial_1.hist_data["Edges"].append(edges)

    counts, edges = create_smoothed_bearing_histogram(
        trial_2,
        ax_frequency[2],
        "late",
        is_density=False,
        my_color="blue"
    )[:-1]
    trial_2.hist_data["Data File Name"].append(
        "Bearing counts (after min(distance)) - Trial 2")
    trial_2.hist_data["is_density"].append(False)
    trial_2.hist_data["Counts/Densities"].append(counts)
    trial_2.hist_data["Edges"].append(edges)

    for ax in ax_frequency:
        add_subplot_legend(ax)

    fig_frequency.savefig("Plots/Histograms/Histograms - bearing frequency")
    fig_frequency.show()
    plt.close(fig_frequency)
    ###########################################################################
    # DENSITY HISTOGRAMS

    # WHOLE TRIAL
    counts, edges = create_smoothed_bearing_histogram(
        trial_1,
        ax_density[0],
        "whole",
        is_density=True,
        my_color="red"
    )[:-1]
    trial_1.hist_data["Data File Name"].append(
        "Bearing density (whole trial) - Trial 1")
    trial_1.hist_data["is_density"].append(True)
    trial_1.hist_data["Counts/Densities"].append(counts)
    trial_1.hist_data["Edges"].append(edges)

    counts, edges = create_smoothed_bearing_histogram(
        trial_2,
        ax_density[0],
        "whole",
        is_density=True,
        my_color="blue"
    )[:-1]
    trial_2.hist_data["Data File Name"].append(
        "Bearing density (whole trial) - Trial 2")
    trial_2.hist_data["is_density"].append(True)
    trial_2.hist_data["Counts/Densities"].append(counts)
    trial_2.hist_data["Edges"].append(edges)

    # UNTIL MIN
    counts, edges = create_smoothed_bearing_histogram(
        trial_1,
        ax_density[1],
        "early",
        is_density=True,
        my_color="red"
    )[:-1]

    trial_1.hist_data["Data File Name"].append(
        "Bearing density (until min(distance)) - Trial 1")
    trial_1.hist_data["is_density"].append(True)
    trial_1.hist_data["Counts/Densities"].append(counts)
    trial_1.hist_data["Edges"].append(edges)

    counts, edges = create_smoothed_bearing_histogram(
        trial_2,
        ax_density[1],
        "early",
        is_density=True,
        my_color="blue"
    )[:-1]

    trial_2.hist_data["Data File Name"].append(
        "Bearing density (until min(distance)) - Trial 2")
    trial_2.hist_data["is_density"].append(True)
    trial_2.hist_data["Counts/Densities"].append(counts)
    trial_2.hist_data["Edges"].append(edges)

    # AFTER MIN
    counts, edges = create_smoothed_bearing_histogram(
        trial_1,
        ax_density[2],
        "late",
        is_density=True,
        my_color="red"
    )[:-1]
    trial_1.hist_data["Data File Name"].append(
        "Bearing density (after min(distance)) - Trial 1")
    trial_1.hist_data["is_density"].append(True)
    trial_1.hist_data["Counts/Densities"].append(counts)
    trial_1.hist_data["Edges"].append(edges)

    counts, edges = create_smoothed_bearing_histogram(
        trial_2,
        ax_density[2],
        "late",
        is_density=True,
        my_color="blue"
    )[:-1]
    trial_2.hist_data["Data File Name"].append(
        "Bearing density (after min(distance)) - Trial 2")
    trial_2.hist_data["is_density"].append(True)
    trial_2.hist_data["Counts/Densities"].append(counts)
    trial_2.hist_data["Edges"].append(edges)

    for ax in ax_density:
        add_subplot_legend(ax)

    fig_density.savefig("Plots/Histograms/Histograms - bearing density")
    fig_density.show()
    plt.close(fig_density)

def format_distance_subplot(subplot, x_label, y_label):
    subplot.set_xlabel(x_label)
    subplot.set_ylabel(y_label)

    subplot.tick_params(axis="x", labelrotation=45)


    return subplot

def distance_plot_best_fit(trial, segment, subplot):
    if segment == "whole":
        # grab every 5th frame --> sampling distance every 1 second
        distance_array = trial.df.iloc[::5]["Distance"]

    else:
        frame_closest = trial.df["Distance"].idxmin()

        if segment == "early":
            distance_array = trial.df.iloc[:frame_closest:5]["Distance"]
        elif segment == "late":
            distance_array = trial.df.iloc[frame_closest + 1::5]["Distance"]
        # segment is not a valid string
        else:
            raise ValueError("Invalid segment argument provided ("
                             "expected whole, early, or late)")

    total_time = distance_array.shape[0]
    time_array = np.arange(0, distance_array.shape[0], 1)

    a,b = np.polyfit(time_array,distance_array,  1)

    subplot.plot(time_array, distance_array,  c="red", alpha = 0.5)
    subplot.plot(time_array, a*time_array + b, c="blue", alpha = 0.5, label = "Best fit (slope={:.3f})".format(a))
    subplot.set(xticks = np.arange(0, time_array.shape[0], 60))
    subplot.set(
        ylim = [0, distance_array.max()]
    )

def generate_distance_best_fits(trial):
    df= trial.df.copy()


    fig, ax = plt.subplots(1, 3, figsize=(10, 5), layout="constrained")
    fig.suptitle(
        "Distance to Target over time - Trial {}".format(trial.trial_num),
        fontweight='bold',
        fontsize=figure_title_fontsize)

    distance_plot_best_fit(trial, "whole", ax[0])
    distance_plot_best_fit(trial, "early", ax[1])
    distance_plot_best_fit(trial, "late", ax[2])

    ax[0].set_title("Whole Trial")
    ax[1].set_title("Until min(distance)")
    ax[2].set_title("After min(distance)")

    format_distance_subplot(ax[0], "Time (s)", "Distance (pixels)")
    format_distance_subplot(ax[1], "Time (s)", "Distance (pixels)")
    format_distance_subplot(ax[2], "Time (s)", "Distance (pixels)")

    # ax[0].legend()
    add_subplot_legend(ax[0])
    add_subplot_legend(ax[1])
    add_subplot_legend(ax[2])
    ax[0].legend(fontsize = 10)
    ax[1].legend(fontsize = 10)
    ax[2].legend(fontsize = 10)

    fig.savefig("Plots/Distance Plots/Distance - Trial {}".format(
        trial.trial_num))
    fig.show()
    plt.close(fig)

# returns tuple x,y,distance,frame of frame when closest
def get_closest_point(trial):
    df = trial.df
    frame_closest = df["Distance"].idxmin()
    x_closest = df["X"].iloc[frame_closest]
    y_closest = df["Y"].iloc[frame_closest]
    distance_closest = df["Distance"].iloc[frame_closest]

    return x_closest, y_closest, distance_closest, frame_closest


def print_closest_point_data(trial):
    x_closest, y_closest, distance_closest, frame_closest = get_closest_point(
        trial)
    time_closest = frame_closest * 0.2
    trial_num = trial.trial_num
    delta_x = x_closest - trial.df["X"].iloc[0]
    delta_y = y_closest - trial.df["Y"].iloc[0]

    distance_traveled = math.sqrt(delta_x ** 2 + delta_y ** 2)
    print("CLOSEST POINT DATA - TRIAL {}".format(trial_num))
    print("\tX = {}".format(x_closest))
    print("\tY = {}".format(y_closest))
    print("\tTime = {}".format(time_closest))
    print("\tDisplacement form start = {}".format(distance_traveled))
    print("\tDistance from target = {}".format(distance_closest))

def main():
    initialize_directories()

    trial_1 = Trial(trial_1_data, 1)
    trial_2 = Trial(trial_2_data, 2)

    trajectory_plot(trial_1)
    trajectory_plot(trial_2)

    #generate_bearing_histograms(trial_1, trial_2)
    generate_smooth_bearing_histograms(trial_1, trial_2)
    # data is GATHERED IN generate_bearing_histograms and we save it to a csv by calling save_histogram_data
    save_histogram_data(trial_1)
    save_histogram_data(trial_2)

    # return velocity array and time array

    v_arr_1, t_arr_1 = velocity_plot(trial_1)
    v_arr_2, t_arr_2 = velocity_plot(trial_2)

    save_velocity_data(v_arr_1, t_arr_1,
                       "Velocity data (normalzied) - trial 1")
    save_velocity_data(v_arr_2, t_arr_2,
                       "Velocity data (normalzied) - trial 2")

    print_closest_point_data(trial_1)
    print_closest_point_data(trial_2)



    generate_distance_best_fits(trial_1)
    generate_distance_best_fits(trial_2)

if __name__ == '__main__':
    main()
