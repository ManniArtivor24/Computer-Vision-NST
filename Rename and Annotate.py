import os
def rename_files_in_subfolders(root_folder_path):
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(root_folder_path):
        # Sort the files to ensure consistent naming
        files.sort()

        # Identify the folder name (e.g., "train" or "test")
        folder_name = os.path.basename(root)

        # Rename each file in the current directory
        for index, filename in enumerate(files):
            # Get the file extension
            file_extension = os.path.splitext(filename)[1]

            # Create the new filename based on the folder name
            new_filename = f"frame{index + 1:04d}_{folder_name}{file_extension}"

            # Full path to the current and new file
            old_file = os.path.join(root, filename)
            new_file = os.path.join(root, new_filename)

            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} to {new_file}")


rename_files_in_subfolders(r'C:\Users\Admin\Desktop\Personal Projects\Datasets\HIT-UAV dataset\hit-uav\images')
rename_files_in_subfolders(r'C:\Users\Admin\Desktop\Personal Projects\Datasets\HIT-UAV dataset\hit-uav\labels')
