# Retrieves files from the SciHub API (https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/BatchScripting?redirectedfrom=SciHubUserGuide.8BatchScripting)
import json
import csv
import requests
import xml.etree.ElementTree as ET

username = 'minamyung'
password = 'pF32xMRWrSK858S'


base_URI = "https://scihub.copernicus.eu/dhus/odata/v1/Products"

filenames = []

base_path = 'C:\\Users\myung\Documents\CSC8099\Data\Coastline_images_to_extract\\'


# Open list of filenames of S1 data used in making the reference coastline
with open('C:\\Users\myung\Documents\CSC8099\Data\\filenames.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Skip first line of CSV file
    for row in reader:
        filenames.append(row) # Save filenames as array

# Query database to get the UUID of each file
for name in filenames:
    # Query breakdown: get first result where the name contains the substring x in json format
    request_URI = base_URI + "?$format=json&$top=1&$filter=substringof('" + name + "',Name)"
    response = requests.get(request_URI, stream=True, auth=(username, password))
    
    if response.status_code != 404:
        json = json.loads(response.content)
        product_name = json['d']['results'][0]['Name']
        product_id = json['d']['results'][0]['Id']
        nodes_URI = base_URI + "('" + product_id + "')/Nodes('" + product_name + ".SAFE')/Nodes('measurement')/Nodes"

        node_response = requests.get(nodes_URI, stream=True, auth=(username, password))
        # Get all measurement nodes for the file
        if node_response.status_code != 404:
            root = ET.fromstring(node_response.content)
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                for child in entry.findall('{http://www.w3.org/2005/Atom}content'):
                    src_suffix = child.attrib['src']
                    save_name = src_suffix[7:-10]
                    path = base_path + save_name + ".tif"
                    # Download all found nodes (should only be .tif)
                    dl_request_URI = base_URI + "('" + product_id + "')/Nodes('" + product_name + ".SAFE')/Nodes('measurement')/" + src_suffix
                    dl_response = requests.get(dl_request_URI, stream=True, auth=(username, password))

                    with open(path, "wb") as f:
                        for chunk in dl_response.iter_content(chunk_size=16*1024):
                            f.write(chunk)
                    print(save_name, "downloaded.")
        else:
            print(product_name, "nodes not found")
    else:
        print(product_name, "could not be downloaded.")