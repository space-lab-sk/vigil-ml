# vigil-ml

### Data Acquisition

We have collected in-situ measurement and images of 4 extreme events outlined in [this file](extreme_events_list.txt). Data comes from numerous instruments:   
- SOHO/EIT 195 Angstrom  
- SOHO/MDI magnetograms and continuum intensity images  
- SOHO/LASCO C2 and C3 coronagraphs
- SOHO/CELIAS proton speed and proton number density  
- WIND/MFI magnetic field   
    
       
The notebook [extract-data.ipynb](extract-data.ipynb) demonstrates use cases for downloading data. The scripts responsible for downloading the data are located in the **extractDataSrc** folder. These scripts are organized into separate Python files and classes, each handling data downloads for a specific instrument.   
    
To obtain data, we take advantage of [hapiclient](https://pypi.org/project/hapiclient/) and [hvpy](https://pypi.org/project/hvpy/). Hapiclient is used to obtain in-situ measurements and timestamps for extisting image data. Hvpy library is used to collect SOHO/LASCO and SOHO/MDI images. Different approach is required for SOHO/EIT 195 images. We utilize [Virtual Solar Observatory](https://sdac.virtualsolar.org/cgi/search) to search for relevant links to .fits files. Then script processes the links to obtain fits files and creates .png images. We observe that this solution provides higher quality for EIT 195 images compared to hvpy solution.   

