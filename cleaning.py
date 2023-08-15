import os

def rename_images(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    image_files.sort()  # Sort files alphabetically

    for index, old_name in enumerate(image_files):
        extension = os.path.splitext(old_name)[1]
        new_name = f"{index + 1:04d}{extension}"  # Format as 0001, 0002, etc.

        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed '{old_name}' to '{new_name}'")

if __name__ == "__main__":
    folder_path = "/home/aymen/Music/dataset/Images/train/"  # Replace with the path to your folder
    rename_images(folder_path)
