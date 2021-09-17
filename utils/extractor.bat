@echo off
@REM Unzips all files in the Coastline_images_to_extract folder into the Coastline_images folder. Delete all files in the first folder after running to avoid duplicates.
@REM Add source folder and destination folder paths below
FOR %%i IN (C:\*) DO tar -xvzf %%i -C C:\
ECHO Done
