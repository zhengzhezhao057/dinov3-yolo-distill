"""Large image sliding window inference with TensorRT/PyTorch support.
Usage: python large_inference.py --img path/to/image.jpg
       python large_inference.py                   (uses default image)
"""
import cv2, torch, numpy as np, time, argparse, os, sys
from pathlib import Path
from ultralytics import YOLO

ENGINE_PATH = r"E:/deeplearning/yolo11m+dinov3/best.engine"
PT_PATH = r"E:/deeplearning/yolo11m+dinov3/best.pt"
DEFAULT_IMG = r"C:\Users\zzz\Downloads\652DDE13070AAD6D16C5A5D19A4192FD_cascade_result.jpg"

def predict_large(img_path, model_path, window=640, stride=480, conf=0.2):
    model = YOLO(model_path, task='detect')
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    patches, offsets = [], []
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1, y1 = min(x0+window, w), min(y0+window, h)
            patches.append(img[y0:y1, x0:x1])
            offsets.append((x0, y0))

    engine_mode = model_path.endswith('.engine')
    use_batch = 1 if engine_mode else 64
    backend = "TensorRT FP16" if engine_mode else "PyTorch FP32"
    print(f'{backend} | {w}x{h} | {len(patches)} patches | Batch={use_batch}')

    t0 = time.time()
    all_boxes, all_scores, all_cls = [], [], []
    for i in range(0, len(patches), use_batch):
        batch_data = patches[i:i+use_batch]
        if engine_mode:
            batch_data = batch_data[0] if len(batch_data) == 1 else batch_data
        r = model.predict(batch_data, imgsz=640, conf=conf, iou=0.5, device=0, verbose=False)
        if engine_mode and not isinstance(r, list):
            r = [r]
        for j, res in enumerate(r):
            ox, oy = offsets[i+j]
            if res.boxes is not None:
                for box, score, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                           res.boxes.conf.cpu().numpy(),
                                           res.boxes.cls.cpu().numpy()):
                    all_boxes.append([box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy])
                    all_scores.append(float(score))
                    all_cls.append(int(cls))

    t_infer = time.time() - t0

    if all_boxes:
        keep = cv2.dnn.NMSBoxes(all_boxes, all_scores, conf, 0.5)
        keep = keep.flatten() if hasattr(keep, 'flatten') else np.array(keep).flatten()
    else:
        keep = []

    for i in keep:
        x1, y1, x2, y2 = map(int, all_boxes[i])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(img, f'{model.names[all_cls[i]]} {all_scores[i]:.2f}',
                    (x1, max(y1-5,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    out = Path(img_path).parent / f'{Path(img_path).stem}_pred.jpg'
    cv2.imwrite(str(out), img)

    print(f'Time: {t_infer:.1f}s ({len(patches)/t_infer:.1f} p/s) | {len(keep)} detections')
    print(f'Saved: {out}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default=DEFAULT_IMG)
    parser.add_argument('--mode', choices=['auto','tensorrt','pytorch'], default='auto')
    parser.add_argument('--window', type=int, default=640)
    parser.add_argument('--stride', type=int, default=480)
    parser.add_argument('--conf', type=float, default=0.2)
    args = parser.parse_args()

    if args.mode == 'pytorch' or not os.path.exists(ENGINE_PATH):
        model_path = PT_PATH
    else:
        model_path = ENGINE_PATH

    print(f"Model: {model_path}")
    predict_large(args.img, model_path, args.window, args.stride, args.conf)
