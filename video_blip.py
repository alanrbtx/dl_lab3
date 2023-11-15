from transformers import pipeline
import os
import pathlib
import cv2
import time
import argparse
import requests

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

parser = argparse.ArgumentParser(description='My example explanation')
parser.add_argument(
    '--video_path',
    type=str,
    default=2,
    help='provide an integer (default: 2)'
)
my_namespace = parser.parse_args()

# Open the video file
video = cv2.VideoCapture(my_namespace.video_path)

# Set the desired frequency (in seconds)
save_frequency = 1

# Loop through all the frames
is_frame_available, frame = video.read()
frame_count = 0

while is_frame_available:
    # Get the current timestamp of the frame
    timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert from milliseconds to seconds
    
    # Save the frame as an image file with the timestamp in the filename
    if frame_count % int(save_frequency * video.get(cv2.CAP_PROP_FPS)) == 0:
        cv2.imwrite(f'video_frames/frame$${timestamp:.1f}$$.jpg', frame)
        print(f"Video processing: {timestamp:.1f}")
    
    
    # Read the next frame
    is_frame_available, frame = video.read()
    frame_count += 1

    # Add a delay to regulate the frequency
    time.sleep(1 / video.get(cv2.CAP_PROP_FPS))

# Release the video file
video.release()

image_path = os.listdir("video_frames")
sorted_images = sorted(image_path, key=lambda x: float(x.split('$$')[1]))

descriptions = []
for idx, image in enumerate(sorted_images):
    print("Processing by BLIP: video_frames/" + image)
    descriptions.append("time code " + sorted_images[idx].split("$$")[1] + ": " + pipe("video_frames/" + image)[0].get("generated_text"))


descriptions = descriptions[:-1]
joined_descriptions = ".\n".join(descriptions)
prompt_descriptions = ". ".join(descriptions)

print("""
FRAME-BY-FRAME DESCRIPTION:\n""", joined_descriptions)

def get_response(prompt):
    res = requests.get(url="http://10.207.0.31:5005/get_answer", json={"prompt": f"{prompt}"})
    answer = res.json()["result"][2:]
    return answer

prompt = "This is a text description of consecutive frames with timecodes from the video. The video shows only one person. Based on these descriptions, try to briefly tell what is happening on the video and answer, what happening in this video? Коротко тветь на русском"
res = get_response(prompt=(prompt + prompt_descriptions))
print("""
MODEL RESPONSE: """, res)