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
            re_group = re.search(r'(Scene\d\d)\\(.+)\\frames.+(Camera_\d)\\rgb_(\d+.jpg)', img_path)
            scene_name, category, cam, img_id = re_group.groups()
            frame = str(int(img_id[:-4]))
            cam_id = (cam[-1])
            print(f'Scene_name: {scene_name}, category: {category}, cam_id: {cam_id}, img_id: {img_id}, frame:{frame}')


            # --- processing labels ---
            # Notice: Image after 425 does not have labels
            label_file = os.path.join(label_dir, scene_name, category)
            bbox_file = os.path.join(label_file, "bbox.txt")
            pose_file = os.path.join(label_file, "pose.txt")
            df_bbox = pd.read_csv(bbox_file, sep=" ", names=BBOX_FILE_HEADER.split())
            df_pose = pd.read_csv(pose_file, sep=" ", names=POSE_FILE_HEADER.split())
            # drop the first row, which is the description
            df_bbox.drop(0, axis=0, inplace=True)
            df_pose.drop(0, axis=0, inplace=True)
            # get current frame
            try:
                current_frame_bbox = df_bbox.loc[(df_bbox['frame'] == frame) & (df_bbox['cameraID'] == cam_id)]
                current_frame_pose = df_pose.loc[(df_pose['frame'] == frame) & (df_pose['cameraID'] == cam_id)]

            # get the portion that is useful for kitti
            kitti_bbox = current_frame_bbox[["truncation_ratio",
                                             "occupancy_ratio",
                                             "left", "right", "top", "bottom" # 2d bounding box
                                             ]]
            kitti_pose = current_frame_pose[["alpha",
                                             "width", "height", "length",  # dimension
                                             "camera_space_X", "camera_space_Y", "camera_space_Z", # location in camera coord
                                             "rotation_camera_space_y"]] # rotation Y !!!! make sure IT IS rot-y
            kitti_label = pd.concat([kitti_bbox, kitti_pose], axis=1)
            kitti_columns_name = [
                "truncation_ratio",
                "occupancy_ratio",
                "alpha",
                "left", "top", "right", "bottom",
                "height", "width", "length",
                "camera_space_X", "camera_space_Y", "camera_space_Z",
                "rotation_camera_space_y"
            ]
            kitti_label = kitti_label.reindex(columns=kitti_columns_name)

            pd.set_option('display.max_colwidth', None)
            print(kitti_label)
            exit()