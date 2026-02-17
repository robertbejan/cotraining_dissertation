import os

directory_unlabeled = 'D:/Facultate/Disertatie/mainProject/pythonProject1/organized_labeled_ultrasound_dataset/unlabeled_train'
directory_labeled = 'D:/Facultate/Disertatie/mainProject/pythonProject1/all images'

def get_clean_class(folder_name):
    name = folder_name.lower()
    if "abdomen" in name: return "abdomen"
    if "brain" in name: return "brain"
    if "femur" in name: return "femur"
    if "cervix" in name: return "maternal_cervix"
    if "thorax" in name: return "thorax"
    return "other"


for root_u, _, files_u in os.walk(directory_unlabeled):
    for filename_u in files_u:
        path_img_u = os.path.join(root_u, filename_u)

        found = False
        for root_l, _, files_l in os.walk(directory_labeled):
            if found: break

            for filename_l in files_l:
                if filename_u == filename_l:
                    raw_folder = os.path.basename(root_l)
                    current_class = get_clean_class(raw_folder)

                    new_filename = f"{current_class}_{filename_u}"
                    new_path_u = os.path.join(root_u, new_filename)

                    try:
                        os.rename(path_img_u, new_path_u)
                        print(f"Success: {filename_u} -> {new_filename}")
                        found = True
                        break
                    except FileNotFoundError:
                        print(f"Error: Could not find {path_img_u}. It might have been renamed already.")
                        found = True
                        break