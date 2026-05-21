import cv2
import numpy as np
import subprocess
import os


def open_ffmpeg_writer(output_path, width, height, fps):
    """Открывает ffmpeg процесс как pipe для записи сырых BGR кадров."""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        output_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def write_frame(proc, frame):
    proc.stdin.write(frame.tobytes())


def close_ffmpeg_writer(proc):
    proc.stdin.close()
    proc.wait()


def add_audio(video_path, audio_source_path, output_path):
    subprocess.run([
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_source_path,
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ], check=True)


def get_video_dimensions(path):
    probe = subprocess.run([
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0', path
    ], capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(','))
    return w, h


def convert_webm_to_mp4(src, dst):
    w, h = get_video_dimensions(src)
    w_even = w - (w % 2)
    h_even = h - (h % 2)
    print(f"Source size: {w}x{h} -> encoding as {w_even}x{h_even}")
    subprocess.run([
        'ffmpeg', '-y', '-i', src,
        '-vf', f'scale={w_even}:{h_even}',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        dst
    ], check=True)
    if not os.path.exists(dst) or os.path.getsize(dst) < 1000:
        raise RuntimeError(f"Conversion failed: {dst} is missing or empty")


def concatenate_videos(path1, path2, opath, pc_skip_frames=100):
    if not opath.endswith('.mp4'):
        opath += '.mp4'

    nosound_path = opath.replace('.mp4', '_nosound.mp4')

    # --- Конвертация webm -> mp4 ---
    if path1.endswith('.webm'):
        path1_mp4 = path1.replace('.webm', '_converted.mp4')
        if not os.path.exists(path1_mp4) or os.path.getsize(path1_mp4) < 1000:
            print("Converting webm to mp4...")
            convert_webm_to_mp4(path1, path1_mp4)
        else:
            print(f"Using cached: {path1_mp4}")
        path1 = path1_mp4

    # --- Открываем видео ---
    cap_pc = cv2.VideoCapture(path1)
    cap_ph = cv2.VideoCapture(path2)

    if not cap_pc.isOpened() or not cap_ph.isOpened():
        print("Error: Could not open input files")
        return

    fps_pc = cap_pc.get(cv2.CAP_PROP_FPS)
    fps_ph = cap_ph.get(cv2.CAP_PROP_FPS)
    out_fps = min(fps_pc, fps_ph)
    frame_duration = 1.0 / out_fps

    print(f"FPS pc: {fps_pc:.3f}, FPS ph: {fps_ph:.3f}, out_fps: {out_fps:.3f}")
    print(f"Skipping {pc_skip_frames} frames ({pc_skip_frames/fps_pc:.2f}s) from pc")

    # --- Seek и первый кадр ---
    cap_pc.set(cv2.CAP_PROP_POS_FRAMES, pc_skip_frames)
    ret_pc, frame_pc = cap_pc.read()
    ret_ph, frame_ph = cap_ph.read()

    if not ret_pc or not ret_ph:
        print("Error: Could not read first frames")
        return

    pc_time = cap_pc.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    ph_time = cap_ph.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    if pc_time < 0.001:
        pc_time = pc_skip_frames / fps_pc
        print(f"Warning: POS_MSEC unreliable, using calculated pc_time={pc_time:.3f}s")

    print(f"Start times: pc={pc_time:.3f}s, ph={ph_time:.3f}s")

    # --- Размеры ---
    h_pc, w_pc = frame_pc.shape[:2]
    h_ph, w_ph = frame_ph.shape[:2]
    new_h_ph = int(h_ph * w_pc / w_ph)
    new_h_ph -= new_h_ph % 2
    out_w = w_pc - (w_pc % 2)
    out_h = (h_pc + new_h_ph)
    out_h -= out_h % 2

    print(f"Output size: {out_w}x{out_h} @ {out_fps:.3f} fps")

    # --- Открываем ffmpeg writer для nosound ---
    ffmpeg_proc = open_ffmpeg_writer(nosound_path, out_w, out_h, out_fps)

    def resize_ph(f):
        return cv2.resize(f, (out_w, new_h_ph))

    frame_num = 0

    try:
        while True:
            next_pc_time = pc_time + frame_duration
            next_ph_time = ph_time + frame_duration

            while cap_pc.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 < next_pc_time:
                ret_pc, frame_pc = cap_pc.read()
                if not ret_pc:
                    break
            if not ret_pc:
                break

            while cap_ph.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 < next_ph_time:
                ret_ph, frame_ph = cap_ph.read()
                if not ret_ph:
                    break
            if not ret_ph:
                break

            pc_time = next_pc_time
            ph_time = next_ph_time

            frame_ph_resized = resize_ph(frame_ph)
            frame_pc_crop = frame_pc[:h_pc, :out_w]
            concat = np.concatenate([frame_pc_crop, frame_ph_resized], axis=0)

            if concat.shape != (out_h, out_w, 3):
                print(f"Frame {frame_num} size mismatch: {concat.shape}, expected ({out_h}, {out_w}, 3)")
                break

            write_frame(ffmpeg_proc, concat)
            frame_num += 1

            if frame_num % 100 == 0:
                print(f"  [{frame_num}] pc={pc_time:.2f}s ph={ph_time:.2f}s")

    finally:
        cap_pc.release()
        cap_ph.release()
        close_ffmpeg_writer(ffmpeg_proc)  # ffmpeg сам корректно закроет файл

    print(f"\nWritten {frame_num} frames to {nosound_path}")

    # --- Добавляем звук ---
    print("Adding audio...")
    add_audio(nosound_path, path2, opath)

    print(f"\nDone!")
    print(f"  Without audio: {nosound_path}")
    print(f"  With audio:    {opath}")
    print(f"  Duration:      {frame_num / out_fps:.1f}s ({frame_num} frames @ {out_fps:.2f} fps)")


if __name__ == '__main__':
    concatenate_videos("fin_vid1/pc.webm", "fin_vid1/ph.mp4", "fin_vid1/conc", 228)