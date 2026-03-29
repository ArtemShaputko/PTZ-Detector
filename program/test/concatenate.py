import cv2
import numpy as np
import subprocess

def add_audio(video_path, audio_source_path, output_path):
    subprocess.run([
        'ffmpeg', '-y',
        '-i', video_path,        # наше склеенное видео (без звука)
        '-i', audio_source_path, # источник звука
        '-map', '0:v',           # видео из первого
        '-map', '1:a',           # аудио из второго
        '-c:v', 'copy',          # видео не перекодируем
        '-c:a', 'aac',           # аудио в aac
        '-shortest',             # обрезать по короткому
        output_path
    ], check=True)

def concatenate_videos(path1, path2, opath, pc_skip_s=100):
    cap_pc = cv2.VideoCapture(path1)
    if not cap_pc.isOpened():
        print("Error: Could not open", path1)
        return

    cap_ph = cv2.VideoCapture(path2)
    if not cap_ph.isOpened():
        print("Error: Could not open", path2)
        return

    fps_pc = cap_pc.get(cv2.CAP_PROP_FPS)
    fps_ph = cap_ph.get(cv2.CAP_PROP_FPS)
    print(f"FPS pc: {fps_pc}, FPS ph: {fps_ph}")

    out_fps = min(fps_pc, fps_ph)

    # Skip first pc_skip_s seconds of pc video
    if pc_skip_s > 0.001:
        skip_frames = int(pc_skip_s * fps_pc)
    cap_pc.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)  # просто seek, не читать в цикле
    pc_idx = 0  # <-- FIX 3: начинаем счётчик с пропущенного

    # Read first frames to get dimensions
    ret_pc, frame_pc = cap_pc.read()
    ret_ph, frame_ph = cap_ph.read()
    if not ret_pc or not ret_ph:
        print("Error: Could not read first frame.")
        return

    h_pc, w_pc = frame_pc.shape[:2]
    h_ph, w_ph = frame_ph.shape[:2]

    # Resize ph to match pc height if needed
    if h_ph != h_pc:
        new_w_ph = int(w_ph * h_pc / h_ph)
        frame_ph = cv2.resize(frame_ph, (new_w_ph, h_pc))

    out_w = w_pc
    out_h = h_pc + frame_ph.shape[0]

    out = cv2.VideoWriter(opath, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (out_w, out_h))

    pc_interval = fps_pc / out_fps
    ph_interval = fps_ph / out_fps
    pc_pos = 0.0
    ph_pos = 0.0
    ph_idx = 0
    frame_num = 0

    while True:
        target_pc = int(pc_pos) # учитываем смещение
        target_ph = int(ph_pos)

        while pc_idx <= target_pc:
            ret_pc, frame_pc = cap_pc.read()
            if not ret_pc:
                break
            pc_idx += 1
        if not ret_pc:
            break

        while ph_idx <= target_ph:
            ret_ph, frame_ph = cap_ph.read()
            if not ret_ph:
                break
            ph_idx += 1
        if not ret_ph:
            break

        concat = np.concatenate([frame_pc, frame_ph], axis=0)  # FIX 1
        out.write(concat)

        pc_pos += pc_interval
        ph_pos += ph_interval
        frame_num += 1

        if frame_num % 100 == 0:
            print(f"Processed {frame_num} frames...")

    cap_pc.release()
    cap_ph.release()
    out.release()
    
    # Временный файл без звука -> финальный со звуком
    tmp_path = opath + '_nosound.mp4'
    import os
    os.rename(opath, tmp_path)
    add_audio(tmp_path, path2, opath)
    os.remove(tmp_path)

    print(f"Done. Saved to {opath} ({frame_num} frames at {out_fps:.2f} FPS)")

if __name__ == '__main__':
    concatenate_videos("pc.mp4", "ph.mp4", "conc.mp4", pc_skip_s=15)