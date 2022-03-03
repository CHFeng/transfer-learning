import os
import time
import cv2
# for parse command line
from absl import app, flags
from absl.flags import FLAGS

# 間隔幾個frame擷取一次圖片
CUT_UNIT = 5

flags.DEFINE_string("video", "0", "path to input video or set to 0 for webcam")
flags.DEFINE_string("dest_path", "cutOutput", "path to output pictures")
pwd = os.path.abspath(os.getcwd())


def main(_argv):
    vid = cv2.VideoCapture(FLAGS.video)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    totalFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # create output folder
    if not os.path.exists(FLAGS.dest_path):
        os.mkdir(FLAGS.dest_path)
    # fetch total files of ouput folder
    prevFileNames = os.listdir(FLAGS.dest_path)
    prevFiles = len(os.listdir(FLAGS.dest_path))
    lastFileName = prevFileNames[prevFiles - 1]
    lastImagesCount = int(lastFileName.split('_')[1].split('.')[0])
    print("FPS:{} Total Frames:{} Output Folder:{} Last Count:{}".format(fps, totalFrames, FLAGS.dest_path, lastImagesCount))

    cutCount = 0
    for idx in range(totalFrames):
        vid.grab()
        if idx % (fps * CUT_UNIT) == 0:
            return_value, frame = vid.retrieve()
            filePath = os.path.join(FLAGS.dest_path, "cut_{:0>4d}.jpg".format(cutCount + lastImagesCount))
            cv2.imwrite(filePath, frame)
            cutCount += 1

    print("Total Fetch:{}".format(cutCount))


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass