import os

# Change directory to where your .bmp files are located
os.chdir("./Exp7/")

# Set the amount of padding you want
padding = 6

# Loop through each file in the directory
for file in os.listdir():
    # Check if the file is a .bmp file
    if file.endswith(".bmp"):
        # Extract the original number from the filename
        num = file.replace("Cam2_0_0_", "").split(".")[0]
        # Create the new filename with padding and original number
        new_name = f"Cam2_0_0_{num.zfill(padding)}.bmp"
        # Rename the file
        os.rename(file, new_name)
 
