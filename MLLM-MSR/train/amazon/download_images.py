"""
Download item images from Amazon CDN using image_urls.json.
Multi-threaded, with retry and proxy support.
"""
import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'amazon', 'dpo_ready')
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'amazon', 'images')
NUM_THREADS = 16
TIMEOUT = 10
MAX_RETRIES = 2


def download_image(args):
    filename, url, image_dir = args
    filepath = os.path.join(image_dir, filename)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return True  # already downloaded

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 200 and len(resp.content) > 100:
                with open(filepath, 'wb') as f:
                    f.write(resp.content)
                return True
        except Exception:
            pass
    return False


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, 'image_urls.json'), 'r') as f:
        image_urls = json.load(f)

    print(f"Total images to download: {len(image_urls)}")

    # Check already downloaded
    existing = set(os.listdir(IMAGE_DIR))
    to_download = [(fname, url, IMAGE_DIR)
                   for fname, url in image_urls.items()
                   if fname not in existing]
    print(f"Already downloaded: {len(existing)}, Remaining: {len(to_download)}")

    if not to_download:
        print("All images already downloaded!")
        return

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(download_image, args): args[0] for args in to_download}
        pbar = tqdm(total=len(to_download), desc="Downloading images")
        for future in as_completed(futures):
            if future.result():
                success += 1
            else:
                failed += 1
            pbar.update(1)
        pbar.close()

    print(f"\nDone! Success: {success}, Failed: {failed}, Total: {success + failed}")


if __name__ == '__main__':
    main()
