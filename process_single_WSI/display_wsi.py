import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, RadioButtons
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\temp_folder\12AG00001-14_MDNF01_HES"),
        help="Path to the directory where the infos are saved",
        required=False,
    )
    return parser.parse_args()


def display_wsi(img, final_mask, coord_thumb, df_tiles, name="WSI display"):
    """Show the WSI with the segmentation mask and the tiles predictions in
    an interactive way.

    Args:
    img (np.ndarray)        : (H, W, 3) image of the WSI
    final_mask (np.ndarray) : (H, W) segmentation mask of the WSI
    coord_thumb (np.ndarray): (N, 4, 2) coordinates of the tiles in the WSI. Each tile is represented by its 4 corners.
                            Only the first corner is used to display the tiles. The coordinates should be in the space
                            as the variable img, ie in the 'thumbnail' space.
    df_tiles (pd.DataFrame) : (N, P) dataframe containing the predictions of the tiles. The first two columns are the
                            coordinates of the tiles (in the WSI space) and the rest are the predictions of the tiles. (P>=3). df_tiles includes the tumor predictions.
    name (str)              : name of the figure
    """

    threshold_sign_default = 0.0
    threshold_tum_default = 0.5
    class_name = df_tiles.columns[2:][0]
    num_bins = 40

    valid_pts = np.argwhere(df_tiles[class_name] > threshold_sign_default).squeeze()
    # tum_pts = np.argwhere(df_tiles["tum_pred"] > threshold_tum).squeeze()
    # valid_pts = np.intersect1d(valid_pts, tum_pts)

    # Setup the layout
    mosaic = """
        FABCE
        .DDD.
        .DDD.
        .DDD."""

    # fig = plt.figure(layout="constrained", num="WSI display")
    fig = plt.figure(num=name, layout="tight")
    axd = fig.subplot_mosaic(mosaic)

    axd["A"].imshow(img)
    axd["A"].set_title("Original image")

    axd["B"].imshow(final_mask, cmap="gray")
    axd["B"].set_title("Segmentation mask")

    sns.histplot(df_tiles[class_name], bins=num_bins, ax=axd["C"], kde=False)
    axd["C"].axvline(threshold_sign_default, color="r")
    axd["C"].set_title(f"{class_name} prediction distribution")

    sns.histplot(df_tiles["tum_pred"], bins=num_bins, ax=axd["E"], kde=False)
    axd["E"].axvline(threshold_tum_default, color="r")
    axd["E"].set_title(f"Tumor prediction distribution")

    # display text on axis F
    axd["F"].text(
        0.5,
        0.5,
        "Click on the sliders to change the thresholds\n"
        "Choose the class to display with the buttons\n"
        "You can enable or disable tumor filtering",
        ha="center",
        va="center",
        fontsize=10,
        transform=axd["F"].transAxes,
        clip_on=False,
    )

    axd["D"].scatter(
        coord_thumb[valid_pts, 0, 0],
        coord_thumb[valid_pts, 0, 1],
        c="b",
        s=5,
    )
    axd["D"].imshow(img)
    axd["D"].set_title(f"Class {class_name} with threshold_sign {threshold_sign_default}")

    for ax in ["A", "B", "D", "F"]:
        axd[ax].set_aspect("equal")
        axd[ax].set_axis_off()

    # Initialize the interactive elements
    # radio buttons to change the class
    ax_classname = plt.axes([0.01, 0.15, 0.25, len(df_tiles.columns[2:]) * 0.025])
    radio = RadioButtons(ax_classname, df_tiles.columns[2:], active=0)

    #  button to choose to display the tumor or not
    ax_tumor = plt.axes([0.01, 0.05, 0.15, 2 * 0.025])
    tumors = RadioButtons(ax_tumor, ["Display tumor tiles only", "Display all"], active=1)

    # slider to change the threshold_sign
    # ax_tresh = plt.axes([0.2, 0.25, 0.0225, 0.63])
    ax_tresh = plt.axes([0.8, 0.05, 0.0225, 0.63])
    slider_tresh = Slider(
        ax_tresh,
        "Signature",
        df_tiles.drop(columns=["x", "y"]).min().min(),
        df_tiles.drop(columns=["x", "y"]).max().max(),
        valinit=threshold_sign_default,
        orientation="vertical",
    )

    # slide to change the threshold_tum
    # ax_tresh_tum = plt.axes([0.15, 0.25, 0.0225, 0.63])
    ax_tresh_tum = plt.axes([0.85, 0.05, 0.0225, 0.63])
    slider_tresh_tum = Slider(
        ax_tresh_tum,
        "Tumors",
        0,
        1,
        valinit=threshold_tum_default,
        orientation="vertical",
    )

    # function to update the plot
    def update(val):
        class_name = radio.value_selected
        slider_tresh.valmin = df_tiles[class_name].min()
        slider_tresh.valmax = df_tiles[class_name].max()
        slider_tresh.ax.set_ylim(slider_tresh.valmin, slider_tresh.valmax)

        threshold_sign = slider_tresh.val
        threshold_tum = slider_tresh_tum.val
        display_tumor = tumors.value_selected == "Display tumor tiles only"
        valid_pts = np.argwhere(df_tiles[class_name] > threshold_sign).squeeze()

        axd["C"].clear()
        if display_tumor:
            df_tiles["tumoral"] = df_tiles["tum_pred"] >= threshold_tum
            tum_pts = np.argwhere(df_tiles["tumoral"]).squeeze()
            valid_pts = np.intersect1d(valid_pts, tum_pts)
            sns.histplot(data=df_tiles, x=class_name, bins=num_bins, ax=axd["C"], hue="tumoral", kde=False)
            axd["C"].axvline(threshold_sign, color="r")
        else:
            sns.histplot(df_tiles[class_name], bins=num_bins, ax=axd["C"], kde=False)
            axd["C"].axvline(threshold_sign, color="r")
        axd["C"].set_title(f"{class_name} prediction distribution")

        axd["D"].clear()
        axd["D"].imshow(img)
        axd["D"].scatter(coord_thumb[valid_pts, 0, 0], coord_thumb[valid_pts, 0, 1], c="b", s=5)
        axd["D"].set_aspect("equal")
        axd["D"].set_axis_off()

        axd["E"].clear()
        sns.histplot(df_tiles["tum_pred"], bins=num_bins, ax=axd["E"], kde=False)
        axd["E"].axvline(threshold_tum, color="r")
        axd["E"].set_title(f"Tumor prediction distribution")

        fig.canvas.draw_idle()

    # register the update function with the slider and radio buttons
    slider_tresh.on_changed(update)
    radio.on_clicked(update)
    slider_tresh_tum.on_changed(update)
    tumors.on_clicked(update)

    plt.show()


if __name__ == "__main__":
    args = parse_args()

    coord_thumb = np.load(args.dir / "coord_thumb.npy")
    final_mask = np.load(args.dir / "final_mask.npy")
    img = np.load(args.dir / "img.npy")
    tiles_coord = np.load(args.dir / "tiles_coord.npy")
    df_tiles = pd.read_csv(args.dir / "tiles_preds.csv")

    display_wsi(img, final_mask, coord_thumb, df_tiles)
