import pandas as pd
import os

import torch.utils.data


def class_sorter():
    xlsx_path = '/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data_filtered.xlsx'

    dataset = pd.read_excel(xlsx_path, sheet_name='FETAL_PLANES_DB_data')
    df = pd.DataFrame(dataset)

    df_plane = df.loc[:, 'Plane']
    df_brain = df.loc[:, 'Brain_plane']

    aux_df_plane = df_plane.values.copy()
    for i in range(len(df_plane.values)):
        if df_plane.values[i] == 'Fetal brain':
            aux_df_plane[i] = df_plane.values[i] + '-' + df_brain.values[i]

    classes = set(aux_df_plane)
    print('Classes are: ', classes)
    df_plane = aux_df_plane.copy()
    df.loc[:, 'Plane'] = df_plane

    root_path = 'D:/Facultate/Disertatie/mainProject/pythonProject1/all images'
    for i in classes:
        directory = i
        path = os.path.join(root_path, directory)
        os.mkdir(path)

    return df


def get_class(df, img_name):
    df_image_name = df.loc[:, 'Image_name']
    df_plane = df.loc[:, 'Plane']
    img_name = img_name[:-4]
    i = list(df_image_name).index(img_name)

    if img_name == df_image_name[i]:
        return df_plane[i]
    else:
        print("Error at get_class: wrong img name")


def folder_sorter(df):
    init_path = '/FETAL_PLANES_ZENODO/Images'
    final_path = 'D:/Facultate/Disertatie/mainProject/pythonProject1/all images'
    for (root, dirs, file) in os.walk(init_path):
        file = sorted(file)
        for f in file:

            img_class = get_class(df, f)
            img_init_path = os.path.join(init_path, f)
            img_final_path = os.path.join(final_path, img_class, f)
            os.rename(img_init_path, img_final_path)
            # torch.utils.data.random_split()

    return print('Sorting done')


def main():
    df = class_sorter()
    folder_sorter(df)


if __name__ == "__main__":
    main()
