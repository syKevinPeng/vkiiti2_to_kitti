import os, re
import pandas as pd

VKITTI_DIR = os.path.join("D:/Study/FIRE/KITTI-Orientation/virtual_kitti")
VKITTI_VER = "vkitti_2.0.3"
BBOX_FILE_HEADER = "frame cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio isMoving"
POSE_FILE_HEADER = "frame cameraID trackID alpha width height length world_space_X world_space_Y world_space_Z rotation_world_space_y rotation_world_space_x rotation_world_space_z camera_space_X camera_space_Y camera_space_Z rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z"

img_dir = os.path.join(VKITTI_DIR, VKITTI_VER+"_rgb")
label_dir =  os.path.join(VKITTI_DIR, VKITTI_VER+"_textgt.tar")

if __name__ == "__main__":
    if not os.path.isdir(img_dir): raise FileNotFoundError(f'{img_dir} cannot be found')
    if not os.path.isdir(label_dir): raise FileNotFoundError(f'{label_dir} cannot be found')
    for currentpath, folders, files in os.walk(img_dir):
        for file in files:
            # --- processing images ---
            print(os.path.join(currentpath, file))
            img_path = os.path.join(currentpath, file)
            re_group = re.search(r'(Scene\d\d)\\(\w+).+(Camera_\d)\\rgb_(\d+.jpg)', img_path)
            scene_name, category, cam_id, img_id = re_group.groups()
            print(f'Scene_name: {scene_name}, category: {category}, cam_id: {cam_id}, img_id: {img_id}')


            # --- processing labels ---
            label_file = os.path.join(label_dir, scene_name, category)
            bbox_file = os.path.join(label_file, "bbox.txt")
            pose_file = os.path.join(label_file, "pose.txt")
            df_bbox = pd.read_csv(bbox_file, sep=" ", names=BBOX_FILE_HEADER.split())
            df_pose = pd.read_csv(pose_file, sep=" ", names=POSE_FILE_HEADER.split())

            print(df_bbox)
            break