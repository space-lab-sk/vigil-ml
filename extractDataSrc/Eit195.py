import os
import pandas as pd
import requests
import glob
import numpy as np
import cv2
from astropy.io import fits
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sunpy.map


class Eit195:
    def __init__(self, start_datetime: str, stop_datetime: str) -> None:
        self.start_date = start_datetime
        self.stop_date = stop_datetime


    def extract_data(self, csv_filename: str, quality_check=True):
        
        df = pd.read_csv(csv_filename, header=None, names=['links'])

        filtered_df = df[df['links'].str.contains('fits')]

        # ------------ extract fits data------------
        self._download_fits_files_from_csv(filtered_df)

        # ------------ transform and load-----------
        # ---------(from fits to png and save)-------
        self._fits_to_png()

        # -------------quality check----------------
        if quality_check:
            self._quality_check()


        # ------------cleanup fits files-----------

        self._cleanup_fits_files()

        
    def _download_fits_files_from_csv(self, df: pd.DataFrame):
        """This code snippet iterates through a DataFrame assumed to contain a 'links' 
        column with download URLs for FITS files.  For each row, it extracts the link, 
        retrieves the filename from the URL, and calls a separate function (self._download_fits_file)
          to handle the download and file saving process.

        Args:
            df (pd.DataFrame): dataframe of links to fits files
        """
        files_count = len(df)

        for index, row in df.iterrows():

            print(f"downloading fits files... {index} / {files_count}")
            link: str = row['links']
            filename = link.split('/')[-1]
            self._download_fits_file(link, filename)

            clear_output(wait=True)


    def _download_fits_file(self, url: str, filename: str):
        """This function downloads a FITS file from a specified URL and saves it with a given filename.
        It constructs the destination path within a designated folder (data_fits/eit).
        The function uses the requests library to download the file in chunks.
        It iterates through content chunks, keeps track of downloaded data size,
        and writes each chunk to the file.

        Args:
            url (str): url for fits file to download
            filename (str): extracted filename to use for fits_file
        """

        destination_fits_folder = "data_fits/eit"

        with open(os.path.join(destination_fits_folder, filename), "wb") as f:
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)


    def _extract_filename(self, original_file_link: str):
        """changes Helioviewer .fits name convention to VSO naming convention. 
        Changin to VSO naming convention will provide ability to synergy with data from Adam's Bachelor Thesis if needed

        Args:
            original_file_link (str): Helioviewer .fits naming convention e.g. SOHO_EIT_195_20020201T000010_L1.fits

        Returns:
            str: VSO naming convention e.g. 20020201_0000_eit195_1024.png
        """

        resolution = "1024"
        filename = original_file_link.split('/')[-1]

        parts = filename.split("_")
        datetime = parts[3].split("T")

        date = datetime[0]
        time = datetime[1][:4]
        instrument = parts[1].lower()
        channel = parts[2]

        # Reformat and return the filename
        return f"{date}_{time}_{instrument}{channel}_{resolution}.png"
    

    def _fits_to_png(self):
        """
        Processes FITS files and saves them as PNG images with specific formatting.

        This function takes FITS files, flips them vertically, and converts them into PNG images
          with a specific colormap and normalization. It iterates through FITS files in a directory,
            processes each one, and saves the resulting image to a designated folder. 

        CONSTANTS:
            source_dir (str): Path to the directory containing FITS files. 
                The path can include wildcards (e.g., "data_fits/eit/*.fits").
            target_dir (str): Path to the directory for saving processed PNGs.
            color_map (str): Name of the Matplotlib colormap to use for image display 
                (e.g., "sohoeit195").

        Returns:
            None
        """

        source_dir="data_fits/eit/*.fits"
        target_dir="data_processed/eit/" 
        color_map="sohoeit195"

        fits_links = glob.glob(source_dir)

        for count, fits_file_link in enumerate(fits_links):
            
            print(f"transforming and asving image... {count + 1}")

            image_data = fits.getdata(fits_file_link)

            # data flip verticaly
            image_data = np.flipud(image_data)

            target_directory = target_dir
            filename = self._extract_filename(fits_file_link)
            filename_to_save = target_directory + filename

            plt.figure(figsize=(13.3, 13.3), dpi=100)
            plt.imshow(image_data, cmap=color_map, 
                norm=LogNorm(np.percentile(image_data, 0.1).round(5), 
                            vmax=(np.percentile(image_data, 99.9)).round(5)))
            plt.axis('off')

            try:
                plt.savefig(filename_to_save, bbox_inches='tight', pad_inches=0, transparent=True)
            except ValueError:
                pass

            plt.close()

            clear_output(wait=True)


    def _quality_check(self):
        """
        Deletes PNG files based on a grayscale intensity threshold in data_processed/eit/ folder.

        1. Reads the PNG image using OpenCV's `cv2.imread` function.
        2. Converts the image to grayscale using `cv2.cvtColor`.
        3. Calculates the average intensity of the first row of pixels 
        (assuming all rows have similar intensity distribution).
        4. Compares the average intensity to a user-defined threshold.
        5. If the average intensity is greater than the threshold, the file 
        is deleted using `os.remove`.

        Returns:
            None
        """

        source_dir="data_processed/eit/*.png" 
        treshold = 80.0

        png_files_links = glob.glob(source_dir)

        for count, png_file_link in enumerate(png_files_links):

            image_data = cv2.imread(png_file_link)
            image_data_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            mean = image_data_gray[0].mean()

            if mean > treshold:
                os.remove(png_file_link)

    
    def _cleanup_fits_files(self):
        """
        Removes all FITS files (*.fits) from a data_fits directory.

        Returns:
            None
        """

        source_dir="data_fits/eit/*.fits"
        fits_files_links = glob.glob(source_dir)

        for fits_file_link in  fits_files_links:
            os.remove(fits_file_link)

