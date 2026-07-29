"""Unified inference for both small and large images with TensorRT acceleration.
Usage: python predict.py --img path/to/image.jpg
       python predict.py --dir path/to/folder/     (batch predict all images)
"""
import cv2, torch, numpy as np, time, argparse, os, sys, glob
from pathlib import Path
from ultralytics import YOLO

ENGINE_PATH = r"E:/deeplearning/yolo11m+dinov3/best.engine"
PT_PATH = r"E:/deeplearning/yolo11m+dinov3/best.pt"

MAX_DIRECT_SIZE = 2048   # Images <= 2048px: direct inference
WINDOW = 640
STRIDE = 480
CONF = 0.2

def get_model():
    if os.path.exists(ENGINE_PATH):
        print(f"[Model] TensorRT FP16: {ENGINE_PATH}")
        return YOLO(ENGINE_PATH, task='detect'), 'tensorrt'
    else:
        print(f"[Model] PyTorch FP32: {PT_PATH}")
        return YOLO(PT_PATH), 'pytorch'

def predict_direct(model, img, backend):
    """Direct inference for small images."""
    r = model.predict(img, imgsz=640, conf=CONF, iou=0.5, device=0, verbose=False)
    return r[0].boxes

def predict_large(model, img, backend):
    """Sliding window for large images."""
    h, w = img.shape[:2]
    patches, offsets = [], []
    for y0 in range(0, h, STRIDE):
        for x0 in range(0, w, STRIDE):
            x1, y1 = min(x0+WINDOW, w), min(y0+WINDOW, h)
            patches.append(img[y0:y1, x0:x1])
            offsets.append((x0, y0))

    all_boxes, all_scores, all_cls = [], [], []
    for i in range(0, len(patches)):
        batch_data = patches[i]
        r = model.predict(batch_data, imgsz=640, conf=CONF, iou=0.5, device=0, verbose=False)
        res = r[0] if isinstance(r, list) else r
        ox, oy = offsets[i]
        if res.boxes is not None:
            for box, score, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                       res.boxes.conf.cpu().numpy(),
                                       res.boxes.cls.cpu().numpy()):
                all_boxes.append([box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy])
                all_scores.append(float(score))
                all_cls.append(int(cls))

    if all_boxes:
        keep = cv2.dnn.NMSBoxes(all_boxes, all_scores, CONF, 0.5)
        keep = keep.flatten() if hasattr(keep, 'flatten') else np.array(keep).flatten()
    else:
        keep = []
    return all_boxes, all_scores, all_cls, keep

def draw_and_save(img, all_boxes, all_scores, all_cls, keep, out_path, model, elapsed):
    for i in keep:
        x1, y1, x2, y2 = map(int, all_boxes[i])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(img, f'{model.names[all_cls[i]]} {all_scores[i]:.2f}',
                    (x1, max(y1-5,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imwrite(str(out_path), img)
    print(f'  -> {len(keep)} detections | {elapsed:.1f}s | {out_path}')

def process_image(img_path, model, backend, out_dir=None):
    img = cv2.imread(img_path)
    if img is None:
        print(f'[SKIP] Cannot read: {img_path}')
        return
    h, w = img.shape[:2]
    stem = Path(img_path).stem
    out_path = (Path(out_dir) / f'{stem}_pred.jpg') if out_dir else Path(img_path).parent / f'{stem}_pred.jpg'

    t0 = time.time()
    if max(h, w) <= MAX_DIRECT_SIZE:
        print(f'[Direct] {w}x{h} -> ', end='', flush=True)
        boxes = predict_direct(model, img, backend)
        elapsed = time.time() - t0
        all_boxes = boxes.xyxy.cpu().numpy().tolist() if boxes is not None else []
        all_scores = boxes.conf.cpu().numpy().tolist() if boxes is not None else []
        all_cls = boxes.cls.cpu().numpy().astype(int).tolist() if boxes is not None else []
        keep = list(range(len(all_boxes)))
    else:
        print(f'[Large]  {w}x{h} -> ', end='', flush=True)
        all_boxes, all_scores, all_cls, keep = predict_large(model, img, backend)
        elapsed = time.time() - t0

    draw_and_save(img, all_boxes, all_scores, all_cls, keep, out_path, model, elapsed)
    return elapsed, len(keep)

def process_directory(dir_path, model, backend):
    exts = ('*.jpg','*.jpeg','*.png','*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(dir_path, ext)))
    files = sorted(files)
    if not files:
        print(f'No images found in {dir_path}')
        return

    out_dir = Path(dir_path) / 'predictions'
    out_dir.mkdir(exist_ok=True)
    print(f'\nProcessing {len(files)} images...\n')

    total_time, total_det = 0, 0
    for f in files:
        result = process_image(f, model, backend, out_dir)
        if result:
            t, d = result
            total_time += t
            total_det += d

    print(f'\n{"="*50}')
    print(f'Total: {len(files)} images, {total_time:.1f}s, {total_det} detections')
    print(f'Avg: {total_time/len(files):.2f}s/img')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help='Single image path')
    parser.add_argument('--dir', help='Directory of images')
    parser.add_argument('--conf', type=float, default=CONF)
    args = parser.parse_args()

    CONF = args.conf
    model, backend = get_model()

    if args.dir:
        process_directory(args.dir, model, backend)
    elif args.img:
        process_image(args.img, model, backend)
    else:
        # Default: run on the cascade image
        default_img = r"C:\Users\zzz\Downloads\652DDE13070AAD6D16C5A5D19A4192FD_cascade_result.jpg"
        print(f"No args, using default image")
        process_image(default_img, model, backend)