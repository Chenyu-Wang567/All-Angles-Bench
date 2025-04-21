import os
import shutil
from pathlib import Path
import cv2

def extract_frames_from_video(video_path, output_folder, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    interval = total_frames // num_frames
    
    output_folder.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = output_folder / f"frame{i + 1}_{Path(video_path).stem}.jpg"
        cv2.imwrite(str(frame_filename), frame)
    
    cap.release()

def process_subfolders(root_path, folder_list, output_root):
    for subfolder in Path(root_path).iterdir():
        if subfolder.is_dir():
            folder_name = subfolder.name
            if folder_name not in folder_list:
                continue
            
            target_folder = subfolder / "frame_aligned_videos/downscaled/448"
            if target_folder.exists() and target_folder.is_dir():
                for video_file in target_folder.glob("cam*.mp4"):
                    print(f"Processing video: {video_file}")
                    new_subfolder = Path(output_root) / subfolder.name
                    new_subfolder.mkdir(parents=True, exist_ok=True)
                    
                    extract_frames_from_video(video_file, new_subfolder)

def move_and_rename_frames_from_file(file_path,destination_root, source_root):
    with open(file_path, 'r') as f:
        folder_list = [line.strip() for line in f.readlines()]

    for folder in folder_list:
        folder_path = Path(os.path.join(source_root,folder))
        if folder_path.exists() and folder_path.is_dir():
            target_folder = Path(destination_root) / folder_path.name
            target_folder.mkdir(parents=True, exist_ok=True)
            
            for frame_file in folder_path.glob("frame2_*.jpg"):
                new_filename = frame_file.name.replace("frame2_", "")
                
                destination_file = target_folder / new_filename
                
                shutil.move(str(frame_file), str(destination_file))
                print(f"Moved and renamed {frame_file} to {destination_file}")
        else:
            print(f"Folder {folder} does not exist or is not a directory.")

if __name__ == "__main__":
    folder_list_file = "./folder_list.txt"
    
    with open(folder_list_file, 'r') as f:
        folder_list = [line.strip() for line in f.readlines()]
    
    input_path = "./datasets/ego4d/ego-exo4d/takes"
    output_root = "./extracted_frames_huggingface"
    destination_root = "./egoexo4dscenes"

    process_subfolders(input_path, folder_list, output_root)
    print("Processing frames extraction completed!")
    
    move_and_rename_frames_from_file(folder_list_file, destination_root, output_root)
    print("Files moved and renamed successfully!")
