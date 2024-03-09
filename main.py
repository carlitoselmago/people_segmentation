import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2
import time

capture = cv2.VideoCapture(2)

# Set the window named "frame" to fullscreen
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

segment_video = instanceSegmentation()
target_classes = segment_video.select_target_classes(person = True)

segment_video.load_model("pointrend_resnet50.pkl", confidence=0.96, detection_speed="rapid")
segment_video.process_camera(capture, show_bboxes=True, segment_target_classes = target_classes,frames_per_second=15, check_fps=True, show_frames=True,frame_name="frame",output_video_name="output"+str(time.time())+".mp4")
