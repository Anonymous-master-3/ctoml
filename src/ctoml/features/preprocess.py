
import re
from pathlib import Path
import numpy as np
from PIL import Image


def read_frames(path):
    path = Path(path)
    if path.is_dir():
        files = [p for p in path.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
        files.sort(key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name)])
        for p in files:
            with Image.open(p) as im:
                yield np.asarray(im.convert('RGB'))
    else:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError('Cannot decode video: ' + str(path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()


def crop_padded(frame, box):
    x0, y0, x1, y1 = map(int, box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Invalid ROI box')
    h, w = frame.shape[:2]
    out = np.zeros((y1-y0, x1-x0, 3), dtype=np.uint8)
    sx, sy, ex, ey = max(0,x0), max(0,y0), min(w,x1), min(h,y1)
    if ex > sx and ey > sy:
        out[sy-y0:ey-y0, sx-x0:ex-x0] = frame[sy:ey, sx:ex]
    return out


def mouth_crop(frame, landmarks):
    landmarks = np.asarray(landmarks)
    if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
        raise ValueError('Expected 68 finite XY landmarks')
    x, y = np.rint(landmarks[48:68].mean(axis=0)).astype(int)
    return crop_padded(frame, (x-50, y-30, x+50, y+30))


def face_roi(frame, face_box, size=112):





    x0,y0,x1,y1 = map(float, face_box)
    w,h = x1-x0,y1-y0
    if w <= 0 or h <= 0:
        raise ValueError('Invalid face box')
    patch = crop_padded(frame, (int(x0-1.5*w),int(y0-1.5*h),int(x1+1.5*w),int(y1+1.5*h)))
    h,w = patch.shape[:2]

    if h < w:
        d = (w-h)//2
        patch = np.pad(patch, ((d,d+1),(0,0),(0,0)))
    elif w < h:
        d = (h-w)//2
        patch = np.pad(patch, ((0,0),(d,d+1),(0,0)))
    return np.asarray(Image.fromarray(patch).resize((size,size), Image.Resampling.BILINEAR))


class DlibMouthCropper:
    def __init__(self, predictor_path):
        if not Path(predictor_path).is_file():
            raise FileNotFoundError('Provide local Dlib 68-landmark predictor: ' + str(predictor_path))
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))

    def __call__(self, frame):
        faces = self.detector(frame, 1)
        if len(faces) != 1:
            raise ValueError('GRID expects exactly one detected face; found ' + str(len(faces)))
        shape = self.predictor(frame, faces[0])
        return mouth_crop(frame, [(shape.part(i).x, shape.part(i).y) for i in range(68)])
