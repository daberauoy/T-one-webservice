import subprocess
import os
import logging
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

def split(stereo: str):
    print("Received:", stereo)
    format = stereo.split('.')[-1]
    print("Format:", format)
    command = [
        'ffmpeg',
        '-i', stereo,
        '-map_channel', '0.0.0', f'left.{format}',
        '-map_channel', '0.0.1', f'right.{format}'
    ]
    command_mono = [
        'ffmpeg',
        '-i', stereo,
        '-map_channel', '0.0.0', f'left.{format}',
        '-map_channel', '0.0.0', f'right.{format}'
    ]
    print("Executing command:")
    [print(i, end=" ") for i in command]
    print()

    filenames = [f"left.{format}", f"right.{format}"]
    for filename in filenames:
        if os.path.exists(filename):
            logger.warning(f"Found interfering file '{filename}', deleting..")
            os.remove(filename)

    if is_stereo(stereo):
        subprocess.run(command)
    else:
        subprocess.run(command_mono)
        logger.warning("Audio is mono, when it should not be")

    return format


def delete_files(names):
    for file in names:
        file_path = Path(file)
        if file_path.exists():
            os.unlink(file_path)
            logger.info(f"File {file} deleted")
        else:
            logger.warning(f"File {file} does not exist")


def is_stereo(file_path):
    """
    Проверяет, является ли аудиофайл стерео
    """
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=channels',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            channels = int(result.stdout.strip())
            return channels == 2
        return False

    except Exception as e:
        print(f"Ошибка при проверке каналов: {e}")
        return False