import os, time
import cv2 as cv

def print_image_information(image):
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    print(f"height: {h}")
    print(f"width: {w}")
    print(f"channels: {channels}")
    print(f"size: {image.size}")
    print(f"data type: {image.dtype}")

def read_image_and_print(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Ma akhriyi karo sawirka: {image_path}")
    print_image_information(img)

def _measure_fps(cap, probe_frames=60):
    fps = cap.get(cv.CAP_PROP_FPS)
    if not fps or fps == 0:
        start = time.time(); frames = 0
        while frames < probe_frames:
            ok, _ = cap.read()
            if not ok: break
            frames += 1
        elapsed = time.time() - start
        fps = frames / elapsed if elapsed > 0 else 0.0
    return fps

def save_camera_info_txt(output_path: str, device_index: int = 0):
    cap = cv.VideoCapture(device_index, cv.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError("Kamaraddu ma furmin. Isku day --device 1 iwm.")
    fps = _measure_fps(cap)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"fps: {fps:.2f}\n")
        f.write(f"height: {int(height)}\n")
        f.write(f"width: {int(width)}\n")
    print(f"Wrote: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="lena.png")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "camera_outputs.txt"))
    a = parser.parse_args()
    # IV
    read_image_and_print(a.image)
    # V
    save_camera_info_txt(a.out, a.device)

if __name__ == "__main__":
    main()
