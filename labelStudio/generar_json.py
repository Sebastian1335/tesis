import os
import json

base_dir = r'D:\la-u\ciclo 2025-1\Seminario\DATASET'
base_url = 'http://localhost:8000'

tasks = []
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith('.mp4'):
            abs_path = os.path.abspath(os.path.join(root, file)).replace(os.sep, '/')
            rel_path = os.path.relpath(abs_path, base_dir).replace(os.sep, '/')
            video_url = f"{base_url}/{rel_path}"
            tasks.append({
                "data": {
                    "video": video_url,
                    "full_path": abs_path
                }
            })

with open('import_videos.json', 'w', encoding='utf-8') as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)

print(f"Se generó import_videos.json con {len(tasks)} videos")
