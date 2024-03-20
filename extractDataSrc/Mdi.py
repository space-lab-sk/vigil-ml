from extractDataSrc.Hapi import get_hapi_data
import os
import requests
from IPython.display import clear_output

class Mdi:
    def __init__(self, start_datetime: str, stop_datetime: str) -> None:
        self.start = start_datetime
        self.stop = stop_datetime


    def extract_data(self, product: str):
        
        # ------------extract data-----------------
        server = 'https://api.helioviewer.org/hapi/Helioviewer/hapi'
        # dataset is determined bellow
        dataset = ""
        parameters = 'url'
        

        if product == "mag":
            dataset = "MDI_MAG"
        elif product == "con":
            dataset = "MDI_INT"
        else:
            raise ValueError("product should be \"mag\" for magnetogram or \"con\" for continuum")
        
        hapi_data = get_hapi_data(server, dataset, parameters, self.start, self.stop)
        images_count = len(hapi_data)
        target_dir = f"data_processed/mdi/{product}"

        # ------------load data (save)-----------------
        for count, item in enumerate(hapi_data):

            print(f"{count} / {images_count}")
            filename = self._extract_filename(item[1])

            self._download_file(item[1], filename, target_dir)
            clear_output(wait=True)



    def _download_file(self, url: str, filename: str, target_dir: str):
        """Downloads a file from the specified URL and saves it to the target directory.

        Args:
            url (str): The URL of the file to download.
            filename (str): The filename to use for the downloaded file.
            target_dir (str): The directory where the downloaded file will be saved.

        Raises:
            requests.exceptions.RequestException: If there is an error downloading the file.
        """
        
        with open(os.path.join(target_dir, filename), "wb") as f:
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
            original_file_link (str): Helioviewer .fits naming convention e.g. 2001_03_21__00_00_01_942__SOHO_MDI_MDI_magnetogram.jp2

        Returns:
            str: VSO naming convention e.g. 20020201_0000_mdimag_1024.png
        """

        resolution = "1024"
        filename: str = original_file_link.split('/')[-1]

        parts = filename.split("_")
        date = parts[0] + parts[1] + parts[2]
        time = parts[4] + parts[5] 

        instrument = parts[10].lower()
        channel = parts[12][:3]

        # Reformat and return the filename
        return f"{date}_{time}_{instrument}{channel}_{resolution}.png"

