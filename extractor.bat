@echo off
@REM Unzips all files in the Coastline_images_to_extract folder into the Coastline_images folder. Delete all files in the first folder after running to avoid duplicates.
FOR %%i IN (C:\Users\myung\Documents\CSC8099\Data\Coastline_images_to_extract\*) DO tar -xvzf %%i -C C:\Users\myung\Documents\CSC8099\Data\Coastline_image
ECHO Done