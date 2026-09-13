import cv2
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import math
import sys
import os

warnings.filterwarnings('ignore')

# Section 1: Class Definition & Core Infrastructure
class WaterTank3DReconstructor:
    """Main pipeline class for 3D spatial and rotational reconstruction."""
    
    # --- Subsection 1.1: Class Initialization ---
    def __init__(self):
        """Initialize state variables, calibration parameters, and user configurations."""
        self.water_surface_lines = []
        self.bottom_lines = []
        self.scale_factor = None
        self.tank_length = None
        self.tank_width = None
        self.tank_height = None
        self.coin_diameter = None
        self.drawing_mode = 'water_surface'
        self.current_line_index = 0
        self.temp_line = []
        self.image = None
        self.original_image = None
        self.csv_data = None
        self.video_path = None
        
        self.y_pixel2fov = None
        self.y_fov_zero = None
        self.x_pixel2fov = None
        self.x_fov_zero = None
        self.scale_prod = None



# Section 2: Function Parameter Calculation & Calibration
    # --- Subsection 2.1: Image Interaction & User Selection Helpers ---
    def extract_first_frame(self, video_path):
        """Extract the initial calibration frame from video source."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {video_path}")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Failed to read the initial frame from video.")
        return frame

    def draw_line_callback(self, event, x, y, flags, param):
        """Mouse callback handler to record GUI reference points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_line.append((x, y))
            if len(self.temp_line) == 2:
                cv2.line(self.image, self.temp_line[0], self.temp_line[1], (0, 255, 0), 2)
                if self.drawing_mode == 'water_surface':
                    self.water_surface_lines.append(self.temp_line.copy())
                else:
                    self.bottom_lines.append(self.temp_line.copy())
                self.temp_line = []
                self.current_line_index += 1
                self.update_display()

    def update_display(self):
        """Refresh line overlays and render visual instructions."""
        line_names = ['Near edge', 'Far edge', 'Centerline']
        target_surface = "water surface" if self.drawing_mode == 'water_surface' else "bottom surface"
        
        if self.current_line_index < 3:
            text = f"Draw {target_surface} {line_names[self.current_line_index]} (2 clicks)"
        else:
            text = "Surface done. Press 'n' to switch mode or 'q' to finish"
        
        self.image = self.original_image.copy()
        for line in self.water_surface_lines:
            cv2.line(self.image, line[0], line[1], (0, 255, 0), 2)
        if self.drawing_mode == 'bottom':
            for line in self.bottom_lines:
                cv2.line(self.image, line[0], line[1], (0, 255, 0), 2)
                
        cv2.putText(self.image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Water Tank Setup', self.image)

    def get_user_input_visual(self, video_path):
        """Launch interactive window to record reference lines and tank measurements."""
        self.video_path = video_path
        raw_image = self.extract_first_frame(video_path)
        img_h, img_w = raw_image.shape[:2]
        
        max_display_width = 700
        self.scale_factor = max_display_width / img_w if img_w > max_display_width else 1.0
        display_w = int(img_w * self.scale_factor)
        display_h = int(img_h * self.scale_factor)
        
        self.original_image = cv2.resize(raw_image, (display_w, display_h), interpolation=cv2.INTER_AREA) if self.scale_factor != 1.0 else raw_image
        self.image = self.original_image.copy()
        
        cv2.imshow('Water Tank Setup', self.image)
        cv2.resizeWindow('Water Tank Setup', display_w, display_h)
        cv2.setMouseCallback('Water Tank Setup', self.draw_line_callback)
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n') and self.drawing_mode == 'water_surface' and self.current_line_index == 3:
                self.drawing_mode = 'bottom'
                self.current_line_index = 0
                self.update_display()
            elif key == ord('q') and self.drawing_mode == 'bottom' and self.current_line_index == 3:
                break
            elif key == ord('r'):
                if self.drawing_mode == 'water_surface':
                    self.water_surface_lines = []
                else:
                    self.bottom_lines = []
                self.current_line_index = 0
                self.temp_line = []
                self.update_display()
                
        self.water_surface_lines = (np.array(self.water_surface_lines) / self.scale_factor).tolist()
        self.bottom_lines = (np.array(self.bottom_lines) / self.scale_factor).tolist()
        cv2.destroyAllWindows()
        
        self.tank_length = float(input("Tank length (cm): "))
        self.tank_width = float(input("Tank width (cm): "))
        self.tank_height = float(input("Water depth (cm): "))
        return True

    # --- Subsection 2.2: Field of View and Scale Solvers ---
    def calculate_line_length(self, line):
        """Compute Euclidean distance for a 2D pixel segment."""
        p1, p2 = line
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def calculate_view_angles(self):
        """Calculate field-of-view mapping ratios and origin offsets."""
        water_mid = self.calculate_line_length(self.water_surface_lines[2])
        bottom_mid = self.calculate_line_length(self.bottom_lines[2])
        
        water_y_span = abs((self.water_surface_lines[1][0][1] + self.water_surface_lines[1][1][1]) / 2 - 
                           (self.water_surface_lines[0][0][1] + self.water_surface_lines[0][1][1]) / 2)
        bottom_y_span = abs((self.bottom_lines[1][0][1] + self.bottom_lines[1][1][1]) / 2 - 
                            (self.bottom_lines[0][0][1] + self.bottom_lines[0][1][1]) / 2)
        
        water_fov = np.arcsin((water_y_span / self.tank_length) / (water_mid / self.tank_width))
        bottom_fov = -np.arctan((bottom_y_span / self.tank_length) / (bottom_mid / self.tank_width))
        
        water_y_mid = (self.water_surface_lines[2][0][1] + self.water_surface_lines[2][1][1]) / 2
        bottom_y_mid = (self.bottom_lines[2][0][1] + self.bottom_lines[2][1][1]) / 2
        
        left_mid = self.calculate_line_length([self.water_surface_lines[2][0], self.bottom_lines[2][0]])
        right_mid = self.calculate_line_length([self.water_surface_lines[2][1], self.bottom_lines[2][1]])
        
        left_x_span = abs((self.water_surface_lines[1][0][0] + self.bottom_lines[1][0][0]) / 2 - 
                          (self.water_surface_lines[0][0][0] + self.bottom_lines[0][0][0]) / 2)
        right_x_span = abs((self.water_surface_lines[1][1][0] + self.bottom_lines[1][1][0]) / 2 - 
                           (self.water_surface_lines[0][1][0] + self.bottom_lines[0][1][0]) / 2)
        
        left_fov = np.arcsin((left_x_span / self.tank_length) / (left_mid / self.tank_height))
        right_fov = -np.arctan((right_x_span / self.tank_length) / (right_mid / self.tank_height))
        
        left_x_mid = (self.water_surface_lines[2][0][0] + self.bottom_lines[2][0][0]) / 2
        right_x_mid = (self.water_surface_lines[2][1][0] + self.bottom_lines[2][1][0]) / 2
        
        y_pixel2fov = (water_fov - bottom_fov) / (water_y_mid - bottom_y_mid)
        y_fov_zero = water_y_mid - water_fov / y_pixel2fov
        x_pixel2fov = (right_fov - left_fov) / (right_x_mid - left_x_mid)
        x_fov_zero = right_x_mid - right_fov / x_pixel2fov
        
        return y_pixel2fov, y_fov_zero, x_pixel2fov, x_fov_zero

    def calculating_scale_parameters(self, y_pixel2fov, y_fov_zero, x_pixel2fov, x_fov_zero):
        """Derive combined camera perspective scaling factor across refractive boundary."""
        water_near_angle = abs(x_pixel2fov * (self.water_surface_lines[0][1][0] - self.water_surface_lines[0][0][0]))
        water_far_angle = abs(x_pixel2fov * (self.water_surface_lines[1][1][0] - self.water_surface_lines[1][0][0]))
        
        water_near_scale = self.calculate_line_length(self.water_surface_lines[0]) / (
                self.tank_width / (2 * np.sin(water_near_angle / 2)) * water_near_angle)
        water_far_scale = self.calculate_line_length(self.water_surface_lines[1]) / (
                self.tank_width / (2 * np.sin(water_far_angle / 2)) * water_far_angle)
        
        water_near_y = (self.water_surface_lines[0][0][1] + self.water_surface_lines[0][1][1]) / 2
        water_far_y = (self.water_surface_lines[1][0][1] + self.water_surface_lines[1][1][1]) / 2
        
        water_near_fov = y_pixel2fov * (y_fov_zero - water_near_y)
        water_far_fov = y_pixel2fov * (y_fov_zero - water_far_y)
        
        distance_to_vat1 = self.tank_length * np.cos(water_near_fov) * water_far_scale / (
                water_near_scale * np.cos(water_far_fov) - water_far_scale * np.cos(water_near_fov))
                
        bottom_near_angle = abs(x_pixel2fov * (self.bottom_lines[0][1][0] - self.bottom_lines[0][0][0]))
        bottom_far_angle = abs(x_pixel2fov * (self.bottom_lines[1][1][0] - self.bottom_lines[1][0][0]))
        
        bottom_near_scale = self.calculate_line_length(self.bottom_lines[0]) / (
                self.tank_width / (2 * np.sin(bottom_near_angle / 2)) * bottom_near_angle)
        bottom_far_scale = self.calculate_line_length(self.bottom_lines[1]) / (
                self.tank_width / (2 * np.sin(bottom_far_angle / 2)) * bottom_far_angle)
                
        bottom_near_y = (self.bottom_lines[0][0][1] + self.bottom_lines[0][1][1]) / 2
        bottom_far_y = (self.bottom_lines[1][0][1] + self.bottom_lines[1][1][1]) / 2
        
        bottom_near_fov = y_pixel2fov * (y_fov_zero - bottom_near_y)
        bottom_far_fov = y_pixel2fov * (y_fov_zero - bottom_far_y)
        
        distance_to_vat2 = self.tank_length * np.cos(bottom_near_fov) * bottom_far_scale / (
                bottom_near_scale * np.cos(bottom_far_fov) - bottom_far_scale * np.cos(bottom_near_fov))
                
        distance_to_vat = (distance_to_vat1 + distance_to_vat2) / 2
        
        water_scale_prod = (distance_to_vat / np.cos(water_near_fov)) * water_near_scale
        bottom_scale_prod = (distance_to_vat / np.cos(bottom_near_fov)) * bottom_near_scale
        
        return (water_scale_prod + bottom_scale_prod) / 2

    # --- Subsection 2.3: System Calibration Pipeline ---
    def calibrate_camera(self):
        """Execute parameter calculation sequence to store intrinsic optical settings."""
        self.y_pixel2fov, self.y_fov_zero, self.x_pixel2fov, self.x_fov_zero = self.calculate_view_angles()
        self.scale_prod = self.calculating_scale_parameters(
            self.y_pixel2fov, self.y_fov_zero, self.x_pixel2fov, self.x_fov_zero)


# Section 3: Translational Degree of Freedom (3DoF) Reconstruction
    # --- Subsection 3.1: 2D-to-3D Coordinate Projection Solver ---
    def reconstruct_position(self, scale_prod, y_pixel2fov, y_fov_zero,
                             x_pixel2fov, x_fov_zero, x_pixel, y_pixel, major_axis):
        """Project pixel observations to absolute 3D Cartesian coordinates (X, Y, Z)."""
        x_fov = x_pixel2fov * (x_fov_zero - x_pixel)
        y_fov = y_pixel2fov * (y_pixel - y_fov_zero)
        coin_scale = major_axis / self.coin_diameter
        distance = scale_prod / coin_scale
        
        y = distance * np.sin(y_fov)
        distance_xz = distance * np.cos(y_fov)
        x = distance_xz * np.sin(x_fov)
        z = distance_xz * np.cos(x_fov)
        return x, y, z

    # --- Subsection 3.2: Full Trajectory Spatial Assembly ---
    def reconstruct_trajectory(self, csv_file_path):
        """Construct full continuous 3D spatial motion trajectory combined with rotation data."""
        self.csv_data = pd.read_csv(csv_file_path)
        
        orientation_df = self.reconstruct_orientation_pipeline()
        angle_map = {}
        if orientation_df is not None and len(orientation_df) > 0:
            for _, arow in orientation_df.iterrows():
                angle_map[int(arow['frame'])] = (
                    arow['tilt_angle_deg'],
                    arow['in_plane_angle_deg']
                )
        
        results = []
        for idx, row in self.csv_data.iterrows():
            if pd.isna(row['center_x']):
                results.append({
                    'frame': row['frame'],
                    'x_cm': np.nan,
                    'y_cm': np.nan,
                    'z_cm': np.nan,
                    'tilt_angle_deg': np.nan,
                    'in_plane_angle_deg': np.nan
                })
                continue
            x_pos, y_pos, z_pos = self.reconstruct_position(
                self.scale_prod, self.y_pixel2fov, self.y_fov_zero,
                self.x_pixel2fov, self.x_fov_zero,
                row['center_x'], row['center_y'], row['major_axis']
            )
            frame_id = int(row['frame']) if 'frame' in row.index else idx
            tilt, in_plane = angle_map.get(frame_id, (np.nan, np.nan))
            results.append({
                'frame': frame_id,
                'x_cm': x_pos,
                'y_cm': y_pos,
                'z_cm': z_pos,
                'tilt_angle_deg': tilt,
                'in_plane_angle_deg': in_plane
            })
        return pd.DataFrame(results)


# Section 4: Rotational Degree of Freedom (3DoF) Reconstruction

    # --- Subsection 4.1: Ellipse Geometry Candidate Functions ---
    @staticmethod
    def _orient_normalize(v):
        """Normalize a 3D vector to unit length and enforce micro-zero thresholding."""
        x, y, z = v
        n = math.sqrt(x * x + y * y + z * z)
        if n < 1e-12:
            return (0.0, 0.0, 1.0)
        x /= n
        y /= n
        z /= n
        if abs(x) < 1e-12: x = 0.0
        if abs(y) < 1e-12: y = 0.0
        if abs(z) < 1e-12: z = 0.0
        return (x, y, z)

    @staticmethod
    def _orient_unique_vectors(vectors, decimals=8):
        """Remove duplicate candidate normal vectors within target decimal precision."""
        seen = set()
        out = []
        for v in vectors:
            v = WaterTank3DReconstructor._orient_normalize(v)
            key = (round(v[0], decimals), round(v[1], decimals), round(v[2], decimals))
            if key not in seen:
                seen.add(key)
                out.append(v)
        return out

    @staticmethod
    def _orient_compute_candidates(minor_axis, major_axis, angle_deg):
        """Derive candidate unit normal vectors from 2D ellipse projections."""
        a = max(abs(minor_axis), abs(major_axis))
        b = min(abs(minor_axis), abs(major_axis))
        if a <= 1e-12:
            return [], None, None
        r = max(0.0, min(1.0, b / a))
        theta = math.asin(r)
        theta_deg = math.degrees(theta)
        ang = float(angle_deg) % 180.0
        beta_deg = ang + 90.0 if ang <= 90.0 else ang - 90.0
        beta = math.radians(beta_deg)
        ux, uy = -math.cos(beta), math.sin(beta)
        if abs(ux) < 1e-12: ux = 0.0
        if abs(uy) < 1e-12: uy = 0.0
        norm_xy = math.hypot(ux, uy)
        if norm_xy < 1e-12:
            ux, uy = 0.0, 1.0
        else:
            ux /= norm_xy
            uy /= norm_xy
        cos_t = math.cos(theta)
        cos2 = cos_t * cos_t
        if cos2 < 1e-16:
            return [WaterTank3DReconstructor._orient_normalize((0.0, 0.0, 1.0))], theta_deg, r
        z_abs = math.sqrt(max(0.0, (1.0 - cos2) / cos2))
        if z_abs < 1e-12:
            return [
                WaterTank3DReconstructor._orient_normalize((ux, uy, 0.0)),
                WaterTank3DReconstructor._orient_normalize((-ux, -uy, 0.0))
            ], theta_deg, r
        candidates = []
        for s_xy in (1.0, -1.0):
            for s_z in (1.0, -1.0):
                if s_z < 0:
                    continue
                candidates.append(WaterTank3DReconstructor._orient_normalize((s_xy * ux, s_xy * uy, s_z * z_abs)))
        return candidates, theta_deg, r

    # --- Subsection 4.2: Candidate Selection & Tracking Helpers ---
    @staticmethod
    def _select_resize_for_display(img, max_side=1280):
        """Resize visual canvas to fit within comfortable rendering dimensions."""
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side <= max_side:
            return img.copy()
        scale = max_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _select_normalize(v):
        """Convert input array to floating point unit vector."""
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return None if n < 1e-12 else v / n

    @staticmethod
    def _select_angle_between(a, b, undirected=False):
        """Calculate angular discrepancy between two 3D orientation vectors."""
        a = WaterTank3DReconstructor._select_normalize(a)
        b = WaterTank3DReconstructor._select_normalize(b)
        if a is None or b is None:
            return math.pi
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        ang = math.acos(dot)
        return min(ang, math.pi - ang) if undirected else ang

    @staticmethod
    def _select_load_video_frames(video_path, frame_indices):
        """Extract frame buffers corresponding to input tracking frame indices."""
        needed = sorted({int(i) for i in frame_indices if int(i) >= 0})
        result = {}
        if not needed:
            return result
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return result
        for idx in needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                result[idx] = frame
        cap.release()
        return result

    @staticmethod
    def _select_draw_candidates(img, cand_list, frame_id):
        """Draw normal vector directions and manual controls on candidate image."""
        canvas = WaterTank3DReconstructor._select_resize_for_display(img)
        h, w = canvas.shape[:2]
        ui_scale = max(0.45, min(1.2, max(h, w) / 1280.0))
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
                  (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        cv2.putText(canvas, f"Frame {frame_id}: press 0-{len(cand_list) - 1}",
                    (10, int(28 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * ui_scale, (255, 255, 255), 2, cv2.LINE_AA)
        cx, cy = w // 2, h // 2
        radius = int(min(w, h) * 0.35)
        cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), -1)
        for i, (cid, vec) in enumerate(cand_list):
            color = colors[i % len(colors)]
            label = f"{i}: ({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})" if cid is None else f"{i}: id={cid}, ({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})"
            cv2.putText(canvas, label, (10, int((60 + 26 * i) * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui_scale, color, 2, cv2.LINE_AA)
            if np.all(np.isfinite(vec)):
                ex = int(cx + vec[0] * radius)
                ey = int(cy - vec[1] * radius)
                cv2.arrowedLine(canvas, (cx, cy), (ex, ey), color, 2, tipLength=0.08)
                cv2.putText(canvas, str(i), (ex + 6, ey - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui_scale, color, 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _select_terminal_choice(frame_id, cand_list):
        """Fallback CLI selection interface for vector candidates."""
        n = len(cand_list)
        for i, (cid, vec) in enumerate(cand_list):
            label = f"id={cid}, " if cid is not None else ""
            print(f"  {i}: {label}nx={vec[0]:.6f}, ny={vec[1]:.6f}, nz={vec[2]:.6f}")
        while True:
            s = input(f"Frame {frame_id}: Select candidate [0-{n - 1}] (or 'q' to quit): ").strip()
            if s.lower() in ["q", "quit", "exit"]:
                sys.exit(0)
            if s.isdigit() and 0 <= int(s) < n:
                return int(s)

    @staticmethod
    def _select_manual_choice(img, cand_list, frame_id, allow_display=True):
        """Manual interactive choice handler for initial frame disambiguation."""
        n = len(cand_list)
        if n == 0 or n == 1:
            return 0
        if allow_display and img is not None:
            try:
                canvas = WaterTank3DReconstructor._select_draw_candidates(img, cand_list, frame_id)
                win_name = "Select Normal Vector"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.imshow(win_name, canvas)
                cv2.resizeWindow(win_name, canvas.shape[1], canvas.shape[0])
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        sys.exit(0)
                    if ord("0") <= key < ord("0") + n:
                        cv2.destroyAllWindows()
                        return key - ord("0")
                    if key in (ord("t"), 13, 10, 27):
                        cv2.destroyAllWindows()
                        break
            except Exception:
                return WaterTank3DReconstructor._select_terminal_choice(frame_id, cand_list)
        return WaterTank3DReconstructor._select_terminal_choice(frame_id, cand_list)

    # --- Subsection 4.3: Vector Transformation Utilities ---
    @staticmethod
    def _angle_from_normal(nx, ny, nz, tol=1e-12):
        """Convert 3D vector coordinates to Euler tilt and in-plane azimuth angles."""
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm <= tol:
            return np.nan, np.nan
        rho = math.hypot(nx, ny)
        if rho <= tol:
            azimuth = 0.0
            angle_with_projection = math.pi / 2
        else:
            azimuth = math.atan2(ny, nx) % (2 * math.pi)
            angle_with_projection = math.atan2(nz, rho)
        return math.degrees(angle_with_projection), math.degrees(azimuth)

    # --- Subsection 4.4: Orientation Tracking Pipeline Steps ---
    def _step1_compute_candidates(self):
        """Extract geometric orientation candidates from dataset across frames."""
        candidates = {}
        for _, row in self.csv_data.iterrows():
            try:
                frame_id = int(row['frame'])
                minor = float(row['minor_axis'])
                major = float(row['major_axis'])
                angle = float(row['angle'])
            except (KeyError, ValueError, TypeError):
                continue
            if not all(math.isfinite(v) for v in (minor, major, angle)):
                continue
            cands, _, _ = self._orient_compute_candidates(minor, major, angle)
            if not cands:
                continue
            cands = self._orient_unique_vectors(cands)
            if frame_id not in candidates:
                candidates[frame_id] = []
            for cid, n in enumerate(cands):
                candidates[frame_id].append((cid, np.array(n, dtype=float)))
        return candidates

    def _step2_select_best_and_convert(self, candidates, undirected=False):
        """Track continuous vectors using angular distance optimization between frames."""
        frame_ids = sorted(candidates.keys())
        if not frame_ids:
            return pd.DataFrame(columns=['frame', 'tilt_angle_deg', 'in_plane_angle_deg'])
            
        manual_frames = frame_ids[:1]
        frames_img = self._select_load_video_frames(self.video_path, manual_frames) if self.video_path else {}
            
        results = []
        prev_vec = None
        prev_in_plane = None
        
        for fid in manual_frames:
            cand_list = candidates[fid]
            img = frames_img.get(fid, None)
            choice = self._select_manual_choice(img=img, cand_list=cand_list, frame_id=fid, allow_display=True)
            _, vec = cand_list[choice]
            prev_vec = vec
            tilt, in_plane = self._angle_from_normal(vec[0], vec[1], vec[2])
            prev_in_plane = in_plane
            results.append({'frame': fid, 'tilt_angle_deg': tilt, 'in_plane_angle_deg': in_plane})
            
        for fid in frame_ids[1:]:
            cand_list = candidates[fid]
            best_j = None
            best_angle = float('inf')
            for j, (_, cand_vec) in enumerate(cand_list):
                ang = self._select_angle_between(prev_vec, cand_vec, undirected=undirected)
                if ang < best_angle:
                    best_angle = ang
                    best_j = j
            if best_j is None:
                continue
            _, vec = cand_list[best_j]
            prev_vec = vec
            tilt, in_plane = self._angle_from_normal(vec[0], vec[1], vec[2])
            if prev_in_plane is not None:
                diff = in_plane - prev_in_plane
                while diff > 60.0:
                    in_plane -= 360.0
                    diff -= 360.0
                while diff < -60.0:
                    in_plane += 360.0
                    diff += 360.0
            prev_in_plane = in_plane
            results.append({'frame': fid, 'tilt_angle_deg': tilt, 'in_plane_angle_deg': in_plane})
                
        return pd.DataFrame(results) if results else pd.DataFrame(columns=['frame', 'tilt_angle_deg', 'in_plane_angle_deg'])

    def reconstruct_orientation_pipeline(self):
        """Execute candidates extraction and orientation trajectory solving."""
        candidates = self._step1_compute_candidates()
        if not candidates:
            return pd.DataFrame(columns=['frame', 'tilt_angle_deg', 'in_plane_angle_deg'])
        return self._step2_select_best_and_convert(candidates, undirected=False)


# Section 5: Data Output & Visualization Routines

    # --- Subsection 5.1: File Export Helpers ---
    def save_results(self, result_df, output_path):
        """Export tabular reconstructed output data to CSV file format."""
        result_df.to_csv(output_path, index=False)

    # --- Subsection 5.2: Plotting and Graphics Generation ---
    def visualize_results(self, result_df, save_path=None):
        """Render 3D trajectories, orthogonal projections, and orientation dynamics."""
        valid_data = result_df.dropna(subset=['x_cm', 'y_cm', 'z_cm']).sort_values('frame', ascending=True)
        if len(valid_data) == 0:
            return
            
        has_angle = ('tilt_angle_deg' in valid_data.columns and valid_data['tilt_angle_deg'].notna().any())
        n_cols = 4 if has_angle else 3
        fig = plt.figure(figsize=(5 * n_cols, 5))
        
        ax1 = fig.add_subplot(1, n_cols, 1, projection='3d')
        ax1.plot(valid_data['x_cm'], valid_data['y_cm'], valid_data['z_cm'])
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.set_zlabel('Z (cm)')
        ax1.set_title('3D Trajectory')
        
        ax2 = fig.add_subplot(1, n_cols, 2)
        ax2.plot(valid_data['x_cm'], valid_data['y_cm'])
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_title('Front View')
        ax2.grid(True)
        
        ax3 = fig.add_subplot(1, n_cols, 3)
        ax3.plot(valid_data['x_cm'], valid_data['z_cm'])
        ax3.set_xlabel('X (cm)')
        ax3.set_ylabel('Z (cm)')
        ax3.set_title('Top View')
        ax3.grid(True)
        
        if has_angle:
            ax4 = fig.add_subplot(1, n_cols, 4)
            ax4.plot(valid_data['frame'], valid_data['tilt_angle_deg'], label='Tilt Angle')
            ax4.plot(valid_data['frame'], valid_data['in_plane_angle_deg'], label='In-plane Angle')
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Angle (deg)')
            ax4.set_title('Angle Dynamics')
            ax4.legend()
            ax4.grid(True)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Section 6: Main Execution Entrypoint

# --- Subsection 6.1: Program Main Function ---
def main():
    """Main CLI execution flow managing calibration, input parsing, processing, and output generation."""
    reconstructor = WaterTank3DReconstructor()
    
    video_path = input("Enter path to calibration video: ").strip().strip('"').strip("'")
    if not os.path.exists(video_path):
        return
        
    reconstructor.get_user_input_visual(video_path)
    reconstructor.calibrate_camera()
    
    print("Enter CSV paths to process (empty line to submit):")
    csv_paths = []
    while True:
        p = input().strip().strip('"').strip("'")
        if not p:
            break
        if os.path.exists(p):
            csv_paths.append(p)
            
    if not csv_paths:
        return
        
    all_results = []
    for csv_path in csv_paths:
        filename = os.path.basename(csv_path)
        out_name = f"{os.path.splitext(filename)[0][:4]}recons.csv"
        out_dir = os.path.dirname(csv_path)
        output_path = os.path.join(out_dir, out_name) if out_dir else out_name
        
        while True:
            try:
                coin_d = float(input(f"Enter target diameter (cm) for [{filename}]: ").strip())
                if coin_d > 0:
                    reconstructor.coin_diameter = coin_d
                    break
            except ValueError:
                pass
                
        result_df = reconstructor.reconstruct_trajectory(csv_path)
        reconstructor.save_results(result_df, output_path)
        all_results.append((csv_path, result_df))
        
    if input("Display visual output plots? (y/n): ").strip().lower() == 'y':
        for csv_path, res_df in all_results:
            filename = os.path.basename(csv_path)
            image_name = f"{os.path.splitext(filename)[0][:4]}.png"
            out_dir = os.path.dirname(csv_path)
            save_image_path = os.path.join(out_dir, image_name) if out_dir else image_name
            reconstructor.visualize_results(res_df, save_path=save_image_path)

# --- Subsection 6.2: Script Entry Call ---
if __name__ == "__main__":
    main()