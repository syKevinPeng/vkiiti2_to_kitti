from vkitti_to_kitti import *

# These constant are imported from my projects:
CROP_RESIZE_H, CROP_RESIZE_W = 224, 224
IMG_H, IMG_W = 376, 1242
TRAIN_CLASSES = ['Car', 'Pedestrian', 'Cyclist']
KITTI_CLASSES = ['Cyclist', 'Tram', 'Person_sitting', 'Truck', 'Pedestrian', 'Van', 'Car', 'Misc', 'DontCare']
DIFFICULTY = ['easy', 'moderate', 'hard']
MIN_BBOX_HEIGHT = [40, 25, 25]
MAX_OCCLUSION = [0, 1, 2]
MAX_TRUNCATION = [0.15, 0.3, 0.5]
NUMPY_TYPE = np.float32
VIEW_ANGLE_TOTAL_X = 1.4835298642
VIEW_ANGLE_TOTAL_Y = 0.55850536064

def get_all_objs_from_kitti_dir(label_dir, image_dir, difficulty='hard'):
    # store all objects
    all_objs = []

    image_dir = get_all_img_path(image_dir, True)
    for path_to_img in tqdm.tqdm(image_dir):
        label_file = get_label(path_to_img, label_dir=label_dir)
        objs = [{
            'image_file': path_to_img,
            'class_name': type,
            'class_id': 0, # !!! double check this
            'truncated': float(truncation_ratio),
            'occluded': float(occupancy_ratio),
            'alpha': float(alpha),
            'xmin': int(left),  #!!!! double check this
            'ymin': int(bottom),
            'xmax': int(right),
            'ymax': int(top),
            'height': float(height),
            'width': float(width),
            'length': float(length),
            'dims': np.asarray([height, width, length]), # !!! same as width height length?
            'loc_x': float(camera_space_X),
            'loc_y': float(camera_space_Y),
            'loc_z': float(camera_space_Z),
            'rot-y': float(rotation_camera_space_y),
            'line': None,
            'vier_angle': ((float(left) + float(right)) / 2)/CROP_RESIZE_W * VIEW_ANGLE_TOTAL_X - (VIEW_ANGLE_TOTAL_X / 2)
        }for type, truncation_ratio, occupancy_ratio, alpha, left, top, right, bottom,height, width, length,
        camera_space_X, camera_space_Y, camera_space_Z, rotation_camera_space_y in label_file.to_numpy()]
        all_objs.append(objs)
    return all_objs

if __name__ == "__main__":
    # construct directory and check validation
    img_dir = os.path.join(VKITTI_DIR, VKITTI_VER + "_rgb")
    label_dir = os.path.join(VKITTI_DIR, VKITTI_VER + "_textgt.tar")
    if not os.path.isdir(img_dir): raise FileNotFoundError(f'{img_dir} cannot be found')
    if not os.path.isdir(label_dir): raise FileNotFoundError(f'{label_dir} cannot be found')
    get_all_objs_from_kitti_dir(label_dir, img_dir)