# Antarctic coastline mapping
## Description
Source code for dissertation project which aims to automate the digitisation of the Antactic coastline using satellite images (namely Sentinel 1).
Builds on work done by Anastasija Jadrevska.

## Prerequisites
Dependencies:
- gdal=3.3.1
- geos=3.9.1
- numpy=1.21.0
- opencv=4.5.2
- python=3.9.5
- requests=2.25.1
- scipy=1.7.0
See requirements.txt for full list of dependencies (or creating an environment to run from).
Check imports using test_import.py module.

## Files
Main image processing files:
- test_binary.py : main image processing file
- line_to_vector.py : converting to vector data (must run in QGIS)

## Acknowledgements
Many thanks to my supervisor, David Herbert, from Newcastle University, as well as Louise Ireland and Laura Gerrish from the British Antarctic Survey.

## Contact
Manon Myung:
m.myung2@ncl.ac.uk
myung.mina@gmail.com
