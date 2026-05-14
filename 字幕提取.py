import subprocess
import json
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE = r"C:\ffmpeg\bin\ffprobe.exe"


def get_subtitle_streams(video_path):
    cmd = [
        FFPROBE,
        "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index:stream_tags=language",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout).get("streams", [])


def extract_subtitle(video_path, stream_index, output_file):
    cmd = [
        FFMPEG,
        "-i", video_path,
        "-map", f"0:{stream_index}",
        output_file
    ]

    subprocess.run(cmd)


def main():
    video = "ceshi1.mp4"

    streams = get_subtitle_streams(video)

    if not streams:
        print("没有找到字幕轨道！")
        return

    print("找到字幕流：")
    for s in streams:
        idx = s["index"]
        lang = s.get("tags", {}).get("language", "unknown")
        print(f"索引: {idx}, 语言: {lang}")

    # 默认提取第一个字幕
    first_index = streams[0]["index"]
    extract_subtitle(video, first_index, "output.srt")


if __name__ == "__main__":
    main()