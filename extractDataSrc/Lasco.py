from extractDataSrc.Hapi import get_hapi_data
import os
from IPython.display import clear_output
import numpy as np
import pandas as pd
from datetime import datetime
from hvpy import createScreenshot, DataSource, create_layers


class Lasco:
    def __init__(self, start_datetime: str, stop_datetime: str) -> None:
        self.start = start_datetime
        self.stop = stop_datetime


    def extract_data(self, detector, skip_images=0):

        # ------------extract data-----------------
        server = 'https://api.helioviewer.org/hapi/Helioviewer/hapi'
        # dataset is determined bellow
        dataset = ""
        parameters = 'url'

        scale = None
        hvpy_layer = None
        
        if detector == "c3":
            dataset = "LASCO_C3"
            scale = 56
            hvpy_layer = DataSource.LASCO_C3
        elif detector == "c2":
            dataset = 'LASCO_C2'
            scale = 11.5
            hvpy_layer = DataSource.LASCO_C2
        else:
            raise ValueError("product should be \"c2\" or \"c3\"")
        
        # ------------transform data-----------------
        hapi_data = get_hapi_data(server, dataset, parameters, self.start, self.stop)
        images_count = len(hapi_data)
        target_dir = f"data_processed/lasco/{detector}/"

        df = self._transform_data(hapi_data)

        # ------------load data (save)-----------------
        for index, row in df.iterrows():

            print(f"{index+1} / {images_count}")

            if index<skip_images:
                print("skipping images...")
                continue

            time = str(row['Time'])

            original_datetime = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

            filename = self._extract_filename(time, detector)
            save_path = target_dir+filename

            screenshot_location = createScreenshot(
                date=datetime(original_datetime.year, original_datetime.month, original_datetime.day, original_datetime.hour, original_datetime.minute),
                layers=create_layers([(hvpy_layer, 100)]),
                imageScale=scale,
                x0=0,
                y0=0,
                width=1024,
                height=1024,
                filename=save_path,
                overwrite=True
            )

            clear_output(wait=True)


    def _transform_data(self, data):
        filtered_data = np.array(data, dtype=[('Time', 'S22'), ('url', '<U255')])
        df = pd.DataFrame(data=filtered_data)
        df['Time'] = df['Time'].str.decode('utf-8')
        df['Time'] = pd.to_datetime(df['Time'], format="%Y-%m-%dT%H:%M:%SZ")

        return df
    

    def _extract_filename(self, original_file_link: str, detector: str):
        """changes Helioviewer .fits name convention to VSO naming convention. 
        Changin to VSO naming convention will provide ability to synergy with data from Adam's Bachelor Thesis if needed

        Args:
            original_file_link (str): Helioviewer .fits naming convention e.g. 2001_03_21__00_00_01_942__SOHO_MDI_MDI_magnetogram.jp2

        Returns:
            str: VSO naming convention e.g. 20020201_0000_mdimag_1024
        """

        resolution = "1024"
        instrument = "lasco"
        channel = detector

        filename = original_file_link.split(' ')

        date = filename[0].replace("-", "")
        time = filename[1].replace(":", "")[:4] 

        # Reformat and return the filename
        return f"{date}_{time}_{instrument}{channel}_{resolution}"
