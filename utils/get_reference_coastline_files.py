# Downloads all Sentinel 1 images from PolarView which were used for the latest Antarctic coastline mapping, in .tif format.
# Reference coastline: https://data.bas.ac.uk/collections/e74543c0-4c4e-4b41-aa33-5bb2f67df389/
import time
import requests
import csv

startTime = time.time()


base_URI = 'https://www.polarview.aq/images/104_S1geotiff/'

filenames = []

# Open list of filenames of S1 data used in making the reference coastline
# Add path to csv file below
with open('') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Skip first line of CSV file
    for row in reader:
        filenames.append(row) # Save filenames as array
        
for name in filenames:
    save_name = name[0] + '.tar.gz'
    request_URI = base_URI + save_name
    dl_response = requests.get(request_URI, stream=True)
    if dl_response.status_code != 404:
        # Add destination folder path below
        path = '' + save_name
        with open(path, "wb") as f:
            for chunk in dl_response.iter_content(chunk_size=16*1024):
                f.write(chunk)
        print(save_name, "downloaded.")
    else:
        print(save_name,"could not be downloaded.")
executionTime = (time.time() - startTime)
print("Execution time: " + str(executionTime))
