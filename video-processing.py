import cv2
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass

@dataclass
class Config:
    video_path: str = "6-1.mp4"
    output_csv: str = "mask.csv"
    start_frame: int = 98
    enable_visualization: bool = True
    search_radius: int = 80
    retry_half_size: int = 70
    max_retries: int = 3
    max_fallbacks: int = 3
    auto_manual_reselect: bool = True
    target_area: float = 400.0
    ellipse_min_area: float = 100.0
    ellipse_max_area: float = 2200.0
    min_contour_points: int = 5
    line_min_len: float = 30.0
    line_max_len: float = 120.0
    line_min_area: float = 40.0
    major_jump_threshold: float = 7
    ellipse_axis_ratio_threshold: float = 8.0
    distance_weight: float = 0.6
    shape_weight: float = 0.4
    blur_kernel: tuple = (3, 3)
    adaptive_block_size: int = 31
    adaptive_c: int = 5
    close_kernel_size: int = 25
    open_kernel_size: int = 5
    gray_weight: float = 0.9
    sat_weight: float = 0.1
    mask_binary_threshold: int = 100
    max_display_width: int = 1600
    max_display_height: int = 1200
    vis_scale: float = 2.5
    contour_thickness: int = 1
    box_thickness: int = 2
    ellipse_thickness: int = 1
    line_thickness: int = 1
    center_radius: int = 3
    marker_size: int = 20
    text_font_scale: float = 0.7
    text_thickness: int = 2

def normalize_angle_180(angle):
    return float(angle) % 180.0

def resize_display(frame, cfg):
    h, w = frame.shape[:2]
    max_w = int(getattr(cfg, "max_display_width", 1600))
    max_h = int(getattr(cfg, "max_display_height", 720))
    if max_w <= 0 or max_h <= 0:
        return frame, 1.0
    scale_w = max_w / float(w)
    scale_h = max_h / float(h)
    scale = min(scale_w, scale_h, 1.0)
    if scale >= 1.0:
        return frame, 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    display = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return display, scale

def select_roi(frame, cfg, window_name, message):
    display, scale = resize_display(frame, cfg)
    print(message)
    roi_small = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    if roi_small == (0, 0, 0, 0):
        return None
    x, y, w, h = [int(v / scale) for v in roi_small]
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h

def get_mask(channel, k_close, k_open, cfg):
    blur = cv2.GaussianBlur(channel, cfg.blur_kernel, 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, cfg.adaptive_block_size, cfg.adaptive_c)
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)
    return cv2.morphologyEx(close, cv2.MORPH_OPEN, k_open)

def make_mask(crop, k_close, k_open, cfg):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    gray_mask = get_mask(gray, k_close, k_open, cfg)
    sat_mask = get_mask(sat, k_close, k_open, cfg)
    fused = cv2.addWeighted(gray_mask, cfg.gray_weight, sat_mask, cfg.sat_weight, 0)
    _, mask = cv2.threshold(fused, cfg.mask_binary_threshold, 255, cv2.THRESH_BINARY)
    return mask

def find_best_ellipse(contours, cfg, rel_cx, rel_cy, search_radius):
    best_ellipse, best_score = None, -1.0
    search_radius = max(float(search_radius), 1e-6)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (cfg.ellipse_min_area <= area <= cfg.ellipse_max_area):
            continue
        if len(cnt) < cfg.min_contour_points:
            continue
        try:
            (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        if axis_a >= axis_b:
            major, minor, final_angle = float(axis_a), float(axis_b), float(angle)
        else:
            major, minor, final_angle = float(axis_b), float(axis_a), float(angle) + 90.0
        final_angle = normalize_angle_180(final_angle)
        dist = np.hypot(cx - rel_cx, cy - rel_cy)
        score = ((1.0 - min(dist / search_radius, 1.0)) * cfg.distance_weight + (1.0 - abs(area - cfg.target_area) / max(cfg.target_area, 1e-6)) * cfg.shape_weight)
        if score > best_score:
            best_ellipse = ((cx, cy), (major, minor), final_angle)
            best_score = score
    return best_ellipse

def find_best_line(contours, cfg, target_length, rel_cx, rel_cy, search_radius):
    best_line, best_score = None, -1.0
    search_radius = max(float(search_radius), 1.0)
    if target_length <= 0:
        target_length = (cfg.line_min_len + cfg.line_max_len) / 2.0
    target_length = float(target_length)
    for cnt in contours:
        if len(cnt) < cfg.min_contour_points:
            continue
        if cv2.contourArea(cnt) < cfg.line_min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), _, _ = rect
        box = np.array(cv2.boxPoints(rect), dtype=np.float32)
        edge1 = np.linalg.norm(box[0] - box[1])
        edge2 = np.linalg.norm(box[1] - box[2])
        if edge1 >= edge2:
            length, direction = float(edge1), box[1] - box[0]
        else:
            length, direction = float(edge2), box[2] - box[1]
        if not (cfg.line_min_len <= length <= cfg.line_max_len):
            continue
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction = direction / norm
        dist = np.hypot(cx - rel_cx, cy - rel_cy)
        length_penalty = abs(length - target_length) / max(target_length, 1e-6)
        score = ((1.0 - min(dist / search_radius, 1.0)) * cfg.distance_weight + (1.0 - min(length_penalty, 1.0)) * cfg.shape_weight)
        if score > best_score:
            center = np.array([cx, cy], dtype=np.float32)
            half_len = length / 2.0
            pt1 = center - direction * half_len
            pt2 = center + direction * half_len
            angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
            best_line = ((float(cx), float(cy)), length, angle_deg, pt1, pt2)
            best_score = score
    return best_line

def create_tracker():
    legacy = getattr(cv2, "legacy", None)
    creators = [getattr(legacy, "TrackerCSRT_create", None), getattr(legacy, "TrackerKCF_create", None), getattr(cv2, "TrackerCSRT_create", None), getattr(cv2, "TrackerKCF_create", None)]
    for creator in creators:
        if creator is not None:
            return creator()
    raise RuntimeError("current OpenCV version does not support CSRT/KCF Tracker。")

def get_search_crop(frame, cx, cy, radius):
    h, w = frame.shape[:2]
    x1 = max(0, int(cx - radius))
    y1 = max(0, int(cy - radius))
    x2 = min(w, int(cx + radius))
    y2 = min(h, int(cy + radius))
    return frame[y1:y2, x1:x2], x1, y1

def crop_and_prepare(frame, cx, cy, radius, cfg, k_close, k_open):
    crop, x1, y1 = get_search_crop(frame, cx, cy, radius)
    if crop.size == 0:
        return None
    mask = make_mask(crop, k_close, k_open, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {"mask": mask, "contours": contours, "offset": (x1, y1), "rel_center": (cx - x1, cy - y1), "search_radius": float(radius)}

def make_blank_view(frame, cfg):
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    radius = max(10, int(cfg.search_radius))
    crop, x1, y1 = get_search_crop(frame, cx, cy, radius)
    if crop.size == 0:
        mask = np.zeros((20, 20), dtype=np.uint8)
        return {"mask": mask, "contours": [], "offset": (0, 0), "rel_center": (10.0, 10.0), "search_radius": float(radius), "tracker_box": (0, 0, 1, 1)}
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    return {"mask": mask, "contours": [], "offset": (x1, y1), "rel_center": (cx - x1, cy - y1), "search_radius": float(radius), "tracker_box": (0, 0, 1, 1)}

def copy_with_frame(data, frame_idx):
    out = data.copy()
    out["frame"] = frame_idx
    return out

def get_target_line_length(prev_major, cfg):
    if prev_major <= 0:
        return (cfg.line_min_len + cfg.line_max_len) / 2.0
    return float(np.clip(prev_major, cfg.line_min_len, cfg.line_max_len))

def make_ellipse_data(best_ellipse, offset, frame_idx):
    (cx, cy), (major_axis, minor_axis), angle = best_ellipse
    ox, oy = offset
    return {"frame": frame_idx, "center_x": float(cx) + ox, "center_y": float(cy) + oy, "major_axis": float(major_axis), "minor_axis": float(minor_axis), "angle": float(angle)}

def make_line_data(best_line, offset, frame_idx):
    (cx, cy), length, angle, pt1, pt2 = best_line
    ox, oy = offset
    data = {"frame": frame_idx, "center_x": float(cx) + ox, "center_y": float(cy) + oy, "major_axis": float(length), "minor_axis": 0.0, "angle": float(angle)}
    endpoints = ((int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])))
    return data, endpoints

def accept_ellipse_with_possible_line(best_ellipse, view, prev_data, cfg, frame_idx):
    (cx, cy), (major_axis, minor_axis), angle = best_ellipse
    prev_major = float(prev_data.get("major_axis", 0.0))
    major_axis = float(major_axis)
    minor_axis = float(minor_axis)
    axis_ratio = major_axis / max(minor_axis, 1e-6)
    axis_ratio_threshold = float(getattr(cfg, "ellipse_axis_ratio_threshold", 10.0))
    major_jump = abs(major_axis - prev_major) > cfg.major_jump_threshold
    too_flat = axis_ratio > axis_ratio_threshold
    if not major_jump and not too_flat:
        data = make_ellipse_data(best_ellipse, view["offset"], frame_idx)
        return data, "ellipse", None, True
    if major_jump:
        print(f"frame {frame_idx}: major axis jump {prev_major:.2f} -> {major_axis:.2f}, try line fitting")
    if too_flat:
        print(f"frame {frame_idx}: ratia exceed {axis_ratio:.2f} > {axis_ratio_threshold:.2f}, try line fitting")
    target_len = get_target_line_length(prev_major, cfg)
    rel_cx, rel_cy = view["rel_center"]
    best_line = find_best_line(view["contours"], cfg, target_len, rel_cx, rel_cy, view["search_radius"])
    if best_line is not None:
        data, endpoints = make_line_data(best_line, view["offset"], frame_idx)
        print(f"frame {frame_idx}: line fitting seccess, length={data['major_axis']:.2f}")
        return data, "line", endpoints, True
    return None, "fallback", None, False

def manual_reselect(frame, prev_data, frame_idx, cfg, message):
    new_roi = select_roi(frame, cfg, "Re-select Coin", message)
    if new_roi is None:
        return None, None
    tracker = create_tracker()
    tracker.init(frame, new_roi)
    current_data = copy_with_frame(prev_data, frame_idx)
    return tracker, current_data

def process_frame(frame, tracker, prev_data, frame_idx, cfg, k_close, k_open):
    success, bbox = tracker.update(frame)
    if success:
        tx, ty, tw, th = map(int, bbox)
    else:
        tw = int(max(float(prev_data["major_axis"]), 1.0))
        th = int(max(float(prev_data["minor_axis"]), 1.0))
        tx = int(prev_data["center_x"] - tw / 2.0)
        ty = int(prev_data["center_y"] - th / 2.0)
    tcx, tcy = tx + tw / 2.0, ty + th / 2.0
    view = crop_and_prepare(frame, tcx, tcy, cfg.search_radius, cfg, k_close, k_open)
    if view is None:
        print(f"frame {frame_idx}: fail in search area, use previous data。")
        return copy_with_frame(prev_data, frame_idx), "fallback", None, 1, None
    view["tracker_box"] = (tx, ty, tw, th)
    rel_cx, rel_cy = view["rel_center"]
    best_ellipse = find_best_ellipse(view["contours"], cfg, rel_cx, rel_cy, view["search_radius"])
    if best_ellipse is not None:
        current_data, mode, endpoints, accepted = accept_ellipse_with_possible_line(best_ellipse, view, prev_data, cfg, frame_idx)
        if accepted:
            return current_data, mode, endpoints, 0, view
        print(f"frame {frame_idx}: success in ellipse fitting but fail in line fitting, use previous data。")
        return copy_with_frame(prev_data, frame_idx), "fallback", None, 1, view
    print(f"frame {frame_idx}: first-time ellipse fitting fail, begin re-research")
    retry_view = crop_and_prepare(frame, prev_data["center_x"], prev_data["center_y"], cfg.retry_half_size, cfg, k_close, k_open)
    if retry_view is None:
        print(f"frame {frame_idx}: fail in re-search area, use previous data")
        return copy_with_frame(prev_data, frame_idx), "fallback", None, 1, view
    view = retry_view
    view["tracker_box"] = (tx, ty, tw, th)
    rel_cx, rel_cy = view["rel_center"]
    retry_ellipse = find_best_ellipse(view["contours"], cfg, rel_cx, rel_cy, view["search_radius"])
    if retry_ellipse is not None:
        current_data, mode, endpoints, accepted = accept_ellipse_with_possible_line(retry_ellipse, view, prev_data, cfg, frame_idx)
        if accepted:
            print(f"frame {frame_idx}: accept retry ellipse/line result。")
            return current_data, mode, endpoints, 0, view
        print(f"frame {frame_idx}: success in retry ellipse fitting but fail in line fitting,use previous data")
        return copy_with_frame(prev_data, frame_idx), "fallback", None, 2, view
    print(f"frame {frame_idx}: fail in retry ellipse fitting, use previous data")
    return copy_with_frame(prev_data, frame_idx), "fallback", None, 2, view

def draw_visualization(cfg, view, current_data, mode, line_endpoints_local, frame_idx, consecutive_retries):
    vis_img = cv2.cvtColor(view["mask"], cv2.COLOR_GRAY2BGR)
    h, w = vis_img.shape[:2]
    off_x, off_y = view["offset"]
    if len(view["contours"]) > 0:
        cv2.drawContours(vis_img, view["contours"], -1, (255, 0, 255), cfg.contour_thickness)
    cv2.rectangle(vis_img, (0, 0), (w - 1, h - 1), (0, 255, 0), cfg.box_thickness)
    tx, ty, tw, th = view["tracker_box"]
    cv2.rectangle(vis_img, (int(tx - off_x), int(ty - off_y)), (int(tx - off_x + tw), int(ty - off_y + th)), (0, 0, 255), cfg.box_thickness)
    tcx, tcy = tx + tw / 2.0, ty + th / 2.0
    cv2.drawMarker(vis_img, (int(tcx - off_x), int(tcy - off_y)), (0, 255, 255), cv2.MARKER_CROSS, cfg.marker_size, cfg.box_thickness)
    center_local = (int(current_data["center_x"] - off_x), int(current_data["center_y"] - off_y))
    if mode == "line" and line_endpoints_local is not None:
        pt1, pt2 = line_endpoints_local
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), cfg.line_thickness)
        cv2.circle(vis_img, center_local, cfg.center_radius, (0, 255, 255), -1)
    elif float(current_data["minor_axis"]) > 0:
        axes = (int(current_data["major_axis"] / 2.0), int(current_data["minor_axis"] / 2.0))
        color = (0, 255, 0) if mode == "ellipse" else (0, 255, 255)
        cv2.ellipse(vis_img, center_local, axes, current_data["angle"], 0, 360, color, cfg.ellipse_thickness)
    else:
        length = float(current_data["major_axis"])
        rad = np.deg2rad(float(current_data["angle"]))
        dx = np.cos(rad) * length / 2.0
        dy = np.sin(rad) * length / 2.0
        pt1 = (int(center_local[0] - dx), int(center_local[1] - dy))
        pt2 = (int(center_local[0] + dx), int(center_local[1] + dy))
        color = (0, 255, 0) if mode == "line" else (0, 255, 255)
        cv2.line(vis_img, pt1, pt2, color, cfg.line_thickness)
        cv2.circle(vis_img, center_local, cfg.center_radius, (0, 255, 255), -1)
    vis_frame = cv2.resize(vis_img, (int(w * cfg.vis_scale), int(h * cfg.vis_scale)), interpolation=cv2.INTER_NEAREST)
    status_text = mode.upper()
    if consecutive_retries > 0:
        status_text += f" Retry({consecutive_retries}/{cfg.max_retries})"
    cv2.putText(vis_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, cfg.text_font_scale * 1.3, (0, 255, 255), cfg.text_thickness)
    cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, cfg.text_font_scale * 1.3, (255, 255, 255), cfg.text_thickness)
    cv2.putText(vis_frame, "Click Next | K Re-select | Q Quit", (10, int(h * cfg.vis_scale) - 20), cv2.FONT_HERSHEY_SIMPLEX, cfg.text_font_scale, (200, 200, 200), cfg.text_thickness)
    cv2.imshow("Coin Tracking (Mask)", vis_frame)

def wait_for_next_frame(mouse_state):
    mouse_state["clicked"] = False
    while True:
        key = cv2.waitKey(1)
        if key != -1:
            c = key & 0xFF
            if c in (ord("q"), ord("Q")):
                return "quit"
            if c in (ord("k"), ord("K")):
                return "reselect"
        if mouse_state["clicked"]:
            return "next"

def track_coin_ellipse_advanced(cfg):
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"fail to open the video: {cfg.video_path}")
    if cfg.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError(f"cannot read {cfg.start_frame} frame。")
    roi = select_roi(first_frame, cfg, "Select Coin", f"select the coin in frame {cfg.start_frame}，press ENTER to confirm")
    if roi is None:
        raise ValueError("No area has been selected, or the selected area is invalid.")
    tracker = create_tracker()
    tracker.init(first_frame, roi)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel_size, cfg.close_kernel_size))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel_size, cfg.open_kernel_size))
    x, y, w, h = roi
    init_major = float(max(w, h))
    init_minor = float(min(w, h))
    init_angle = 0.0 if w >= h else 90.0
    prev_data = {"frame": cfg.start_frame, "center_x": x + w / 2.0, "center_y": y + h / 2.0, "major_axis": init_major, "minor_axis": init_minor, "angle": init_angle}
    results = []
    frame_idx = cfg.start_frame
    consecutive_retries = 0
    mouse_state = {"clicked": False}
    if cfg.enable_visualization:
        cv2.namedWindow("Coin Tracking (Mask)")
        def mouse_callback(event, mx, my, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_state["clicked"] = True
        cv2.setMouseCallback("Coin Tracking (Mask)", mouse_callback)
    print("start to process the video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        prev_data_snapshot = prev_data.copy()
        current_data, mode, line_endpoints_local, retry_delta, view = process_frame(frame, tracker, prev_data_snapshot, frame_idx, cfg, k_close, k_open)
        if retry_delta == 0:
            consecutive_retries = 0
        else:
            consecutive_retries += retry_delta
        if cfg.auto_manual_reselect and consecutive_retries >= cfg.max_retries:
            print(f"【Warning】The number of consecutive failures has reached {consecutive_retries}, triggering automatic/manual reselection...")
            new_tracker, manual_data = manual_reselect(frame, prev_data_snapshot, frame_idx, cfg, "Consecutive failures trigger manual reselection: Please reselect the coin in the current frame and press ENTER to confirm.")
            if new_tracker is None:
                print("No area was selected, or the selected area is invalid; the program terminates.")
                break
            tracker = new_tracker
            current_data = manual_data
            mode = "manual"
            line_endpoints_local = None
            consecutive_retries = 0
        if view is None:
            view = make_blank_view(frame, cfg)
        results.append(current_data)
        prev_data = current_data.copy()
        if cfg.enable_visualization:
            draw_visualization(cfg, view, current_data, mode, line_endpoints_local, frame_idx, consecutive_retries)
            action = wait_for_next_frame(mouse_state)
            if action == "quit":
                break
            elif action == "reselect":
                print("The K key was detected, triggering manual reselection of the current frame.")
                new_tracker, manual_data = manual_reselect(frame, prev_data_snapshot, frame_idx, cfg, "Manual reselection with the K key: Please reselect the coin in the current frame and press ENTER to confirm.")
                if new_tracker is not None:
                    tracker = new_tracker
                    current_data = manual_data
                    mode = "manual"
                    line_endpoints_local = None
                    consecutive_retries = 0
                    results[-1] = current_data
                    prev_data = current_data.copy()
                    print(f"frame {frame_idx}: Manual reselection with the K key was successful; the final data from the previous frame will be used.")
                    draw_visualization(cfg, view, current_data, mode, line_endpoints_local, frame_idx, consecutive_retries)
                    cv2.waitKey(1)
                else:
                    print("Invalid manual selection, retain the original result for the current frame.")
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processing has reached the {frame_idx}th frame...")
    cap.release()
    if cfg.enable_visualization:
        cv2.destroyAllWindows()
    pd.DataFrame(results).to_csv(cfg.output_csv, index=False)
    print(f"processing complete! Result stores at: {cfg.output_csv}")

if __name__ == "__main__":
    cfg = Config()
    if not os.path.exists(cfg.video_path):
        print(f"Error: cannot find the video '{cfg.video_path}'。")
    else:
        track_coin_ellipse_advanced(cfg)