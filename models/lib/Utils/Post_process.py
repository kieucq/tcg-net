import argparse
import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import convolve

def post_processing(
    df: pd.DataFrame,
    test: bool = False
) -> pd.DataFrame:
    # Define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="Path to config file", default='/N/slate/tnn3/TruongChu/PostProcessing/config.yaml')

    # Parse command line arguments
    args = parser.parse_args()

    # Function to read YAML configuration file
    def read_yaml_config(file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    # Load YAML config file
    if args.config:
        config = read_yaml_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)

    kernel = np.array([[0, 1, 0], [1, args.center_weight, 1], [0, 1, 0]])
    if np.sum(kernel) != 0:
        kernel = kernel / np.sum(kernel)
    
    df["ScoreOld"] = df["Score"]
    df["Score"] = np.zeros(len(df))

    for date_time in df["Datetime"].unique():
        df_date_orig = df.loc[df["Datetime"] == date_time]
        df_date = df_date_orig.reset_index() 

        #print(df["Lat"].value_counts())
        #print(df["Lon"].value_counts())
        # print(df_date["Lon"].min())
        # print(df_date["Lon"].max())
        # print(df_date["Lat"].min())
        # print(df_date["Lat"].max())

        np_test = np.zeros((7, 11))
        np_real = np.zeros((7, 11))
        for i in range(len(df_date)):
            lat_lon = df_date.at[i, "Point"].split("_")
            lon_id = int(float(lat_lon[1]) - 100) // 5
            lat_id = int(float(lat_lon[0])) // 5

            np_test[lat_id, lon_id] = df_date.at[i, "ScoreOld"]
            np_real[lat_id, lon_id] = df_date.at[i, "Label"]

        #print(np_test)

        max_np_test = np_test.max()

        np_test_new = (np_test - np_test.min()) / (np_test.max() - np_test.min())

        # Apply padding of 1 around the input array
        padded_input = np.pad(np_test_new, pad_width=1, mode='constant', constant_values=0)

        # Perform convolution
        output_conv = convolve(padded_input, kernel, mode='constant', cval=0.0)
        output_conv = output_conv[1:-1, 1:-1]

        # Thresholding
        output_conv[output_conv < args.threshold] = 0

        output_final = np_test_new + output_conv * args.add_rate

        final_out = (output_final - output_final.min()) / (output_final.max() - output_final.min())

        final_out *= max_np_test
        
        for idx in df_date_orig.index:
            lat_lon = df_date_orig.at[idx, "Point"].split("_")
            lon_id = int(float(lat_lon[1]) - 100) // 5
            lat_id = int(float(lat_lon[0])) // 5

            df.at[idx, "Score"] = final_out[lat_id, lon_id]

        if args.print_image is True:
            plt.imshow(output_conv)
            plt.colorbar(fraction=0.046 * 7 / 11, pad=0.04)
            plt.title(f"Original convolution {date_time}")
            for i in range(output_conv.shape[0]):
                for j in range(output_conv.shape[1]):
                    plt.text(j, i, f"{output_conv[i, j]:.2f}", ha='center', va='center', color='white')
            os.makedirs(f"logfiles/img/{date_time}", exist_ok=True)
            plt.savefig(f"logfiles/img/{date_time}/orig_conv.png")
            plt.clf()
            plt.close()

            # Display the result
            #print("Input Array:\n", np_test)
            #print("Kernel:\n", kernel)
            #print("Padded Input Array:\n", padded_input)
            #print("Output Array after Convolution:\n", output_conv)

            fig, axes = plt.subplots(2, 2, figsize=(24, 14))

            axes[0, 0].set_title("Edited output")
            axes[0, 0].imshow(final_out) #, cmap="gray")
            # Overlay each pixel's value on the image
            for i in range(final_out.shape[0]):
                for j in range(final_out.shape[1]):
                    axes[0, 0].text(j, i, f"{final_out[i, j]:.2f}", ha='center', va='center', color='white')
            axes[0, 1].set_title("Mask convolution")
            axes[0, 1].imshow(output_conv) #, cmap="gray")
            # Overlay each pixel's value on the image
            for i in range(output_conv.shape[0]):
                for j in range(output_conv.shape[1]):
                    axes[0, 1].text(j, i, f"{output_conv[i, j]:.2f}", ha='center', va='center', color='white')
            axes[1, 0].set_title("Original prediction")
            axes[1, 0].imshow(np_test) #, cmap="gray")
            # Overlay each pixel's value on the image
            for i in range(np_test.shape[0]):
                for j in range(np_test.shape[1]):
                    axes[1, 0].text(j, i, f"{np_test[i, j]:.2f}", ha='center', va='center', color='white')
            axes[1, 1].set_title("True label")
            im = axes[1, 1].imshow(np_real) #, cmap="gray")
            # Overlay each pixel's value on the image
            for i in range(np_real.shape[0]):
                for j in range(np_real.shape[1]):
                    axes[1, 1].text(j, i, f"{np_real[i, j]:.2f}", ha='center', va='center', color='white')
                    
            fig.subplots_adjust(right=0.8)
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046 * 7 / 11, pad=0.04)

            fig.suptitle(f"Map of {date_time}", fontsize=30)
            plt.savefig(f"logfiles/img/{date_time}/total.png")
            plt.clf()
            plt.close()
            #print(df_date)

        if test:
            break

    # df.to_csv("./mapEdited.csv", index=False)

    return df

def post_processing_file(
    in_file_path: str,
    out_file_path: str,
    test: bool = False
):
    df = pd.read_csv(in_file_path)

    df_edited = post_processing(df, test=test)

    df_edited.to_csv(out_file_path, index=False)

    print(f"Save file to {out_file_path} successful")


if __name__ == "__main__":
    csv_inp_temp = '/N/slate/tnn3/DucHGA/TC/ModelMerra/Output/DynamicSingleFix/ResNet/Dataset1_{}_v0/map_out.csv'
    csv_out_temp = '/N/slate/tnn3/DucHGA/TC/ModelMerra/Output/DynamicSingleFix/ResNet/Dataset1_{}_v0/map_fix_out.csv'

    for step in range(1, 9):
        print(step)
        inp_path = csv_inp_temp.format(step)
        out_path = csv_out_temp.format(step)
        post_processing_file(in_file_path=inp_path,
                             out_file_path=out_path,
                             test=True)
