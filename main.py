import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2

capture = cv2.VideoCapture(0)

segment_video = instanceSegmentation()
target_classes = segment_video.select_target_classes(person = True)

segment_video.load_model("pointrend_resnet50.pkl", confidence=0.96, detection_speed="rapid")
segment_video.process_camera(capture, show_bboxes=True, segment_target_classes = target_classes,frames_per_second=15, check_fps=True, show_frames=True,frame_name="frame",output_video_name="output.mp4")
