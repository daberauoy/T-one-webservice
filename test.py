# import requests
# import os
# from pathlib import Path
#
#
# def send_audio_files_to_asr():
#     base_url = "http://localhost:7999/asr"
#     recordings_folder = "recordings_stereo"
#
#     # Check if recordings folder exists
#     if not os.path.exists(recordings_folder):
#         print(f"Error: Folder '{recordings_folder}' does not exist")
#         return
#
#     # Get all files in the recordings folder
#     audio_files = [f for f in os.listdir(recordings_folder) if os.path.isfile(os.path.join(recordings_folder, f))]
#
#     if not audio_files:
#         print(f"No files found in '{recordings_folder}' folder")
#         return
#
#     print(f"Found {len(audio_files)} files in '{recordings_folder}':")
#     for file in audio_files:
#         print(f"  - {file}")
#
#     print("\nSending requests...")
#
#     for filename in audio_files:
#         file_path = os.path.join(recordings_folder, filename)
#
#         try:
#             # Prepare the form-data
#             files = {
#                 'audio_file': (filename, open(file_path, 'rb'), 'audio/mpeg')
#             }
#
#             # Send POST request with split=true
#             url = f"{base_url}?split=true"
#             print(f"\nSending: {filename}")
#
#             response = requests.post(url, files=files)
#
#             # Check response
#             if response.status_code == 200:
#                 print(f"✓ Success: {filename}")
#                 # Print the response content if needed
#                 # print(f"Response: {response.json()}")
#             else:
#                 print(f"✗ Failed: {filename} - Status: {response.status_code}")
#                 print(f"Error: {response.text}")
#
#         except Exception as e:
#             print(f"✗ Error processing {filename}: {str(e)}")
#         finally:
#             # Ensure file is closed
#             if 'files' in locals() and 'audio_file' in files:
#                 files['audio_file'][1].close()
#
#
# if __name__ == "__main__":
#     send_audio_files_to_asr()

import datetime
from datetime import datetime as dt

# print(f"{datetime.datetime.now().date()} at {datetime.datetime.now().time().hour}")
print(f"{dt.now().date()} at {dt.now().time().hour}:{dt.now().time().minute}")
# print(f"{datetime.date.year.now()} at {datetime.time.hour}")