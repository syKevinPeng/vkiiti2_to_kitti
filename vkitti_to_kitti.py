import os
import re

import numpy as np
import pandas as pd

VKITTI_DIR = os.path.join("D:/Study/FIRE/KITTI-Orientation/virtual_kitti")
VKITTI_VER = "vkitti_2.0.3"
BBOX_FILE_HEADER = "frame cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio isMoving"
POSE_FILE_HEADER = "frame cameraID trackID alpha width height length world_space_X world_space_Y world_space_Z rotation_world_space_y rotation_world_space_x rotation_world_space_z camera_space_X camera_space_Y camera_space_Z rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z"
INFO_FILE_HEADER = "trackID label model color"
EXCLUDED_DIRECTORY = "15-deg-left 15-deg-right 30-deg-left 30-deg-right"
KITTI_COLUMNS_NAME = [
    "type",
    "truncation_ratio",
    "occupancy_ratio",
    "alpha",
    "left", "top", "right", "bottom",
    "height", "width", "length",
    "camera_space_X", "camera_space_Y", "camera_space_Z",
    "rotation_camera_space_y"
]


# return a list of all vkitti full path of images
def get_all_img_path(img_dir, shuffle=False):
    # get all image path to img_list
    img_path_list = []
    for currentpath, dirs, files in os.walk(img_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRECTORY.split()]
        for file in files:
            img_path_list.append(os.path.join(currentpath, file))
    print(img_path_list)
    if shuffle: np.random.shuffle(img_path_list)
    return img_path_list



# get label take in a full image path as input and output kitti-like label in dataframe
def get_label_from_single_path(img_path: str, label_dir: str, verbose=False):
    re_group = re.search(r'(Scene\d\d)\\(.+)\\frames.+(Camera_\d)\\rgb_(\d+.jpg)', img_path)
    scene_name, category, cam, img_id = re_group.groups()
    frame = str(int(img_id[:-4]))
    cam_id = (cam[-1])
    if verbose: print(
        f'Scene_name: {scene_name}, category: {category}, cam_id: {cam_id}, img_id: {img_id}, frame:{frame}')

    label_file = os.path.join(label_dir, scene_name, category)
    bbox_file = os.path.join(label_file, "bbox.txt")
    pose_file = os.path.join(label_file, "pose.txt")
    info_file = os.path.join(label_file, "info.txt")
    df_bbox = pd.read_csv(bbox_file, sep=" ", names=BBOX_FILE_HEADER.split(), low_memory=False)
    df_pose = pd.read_csv(pose_file, sep=" ", names=POSE_FILE_HEADER.split(), low_memory=False)
    df_info = pd.read_csv(info_file, sep=" ", names=INFO_FILE_HEADER.split(), low_memory=False)
    # drop the first row, which is the description
    df_bbox.drop(0, axis=0, inplace=True)
    df_pose.drop(0, axis=0, inplace=True)
    df_info.drop(0, axis=0, inplace=True)

    # get current frame
    # !!! redundant?
    try:  # use try-except to handle situation that no label is provided
        current_frame_bbox = df_bbox.loc[(df_bbox['frame'] == frame) & (df_bbox['cameraID'] == cam_id)]
        current_frame_pose = df_pose.loc[(df_pose['frame'] == frame) & (df_pose['cameraID'] == cam_id)]
    except:
        return False

    df_info.drop(df_info.columns[[2,3]], axis=1, inplace=True)  # get trackID and its corresponding label
    info_dict = dict(zip(df_info['trackID'], df_info['label']))

    # get the portion that is useful for kitti
    kitti_bbox = current_frame_bbox[[
                                    "frame",
                                    "cameraID",
                                    "trackID",
                                    "truncation_ratio",
                                     "occupancy_ratio",
                                     "left", "right", "top", "bottom"  # 2d bounding box
                                     ]]
    kitti_pose = current_frame_pose[["frame",
                                     "cameraID",
                                     "trackID",
                                     "alpha",
                                     "width", "height", "length",  # dimension
                                     "camera_space_X", "camera_space_Y", "camera_space_Z",  # location in camera coord
                                     "rotation_camera_space_y"]]
    # kitti_label = pd.concat([kitti_bbox, kitti_pose], axis=1)
    kitti_label = kitti_bbox.merge(kitti_pose, how='inner', left_on=["frame", "cameraID", "trackID"],
                                   right_on=["frame", "cameraID","trackID"])

    # add missing "type column"
    kitti_label['type'] = kitti_label["trackID"].replace(info_dict)
    kitti_label = kitti_label.reindex(columns=KITTI_COLUMNS_NAME)

    pd.set_option('display.max_colwidth', None)
    if verbose: print(kitti_label)
    return kitti_label

def get_label_sequentially():
    pass

if __name__ == "__main__":
    # construct directory and check validation
    img_dir = os.path.join(VKITTI_DIR, VKITTI_VER + "_rgb")
    label_dir = os.path.join(VKITTI_DIR, VKITTI_VER + "_textgt.tar")
    if not os.path.isdir(img_dir): raise FileNotFoundError(f'{img_dir} cannot be found')
    if not os.path.isdir(label_dir): raise FileNotFoundError(f'{label_dir} cannot be found')

    img_list = get_all_img_path(img_dir, True)
    dataframe = get_label_from_single_path(img_list[1], label_dir=label_dir, verbose=True)
    if isinstance(dataframe, bool) and dataframe == False:
        print("No label provided")
