import requests
import io
import re
from PIL import Image
from typing import List, Tuple
import os

import io
import re
import requests
from PIL import Image
from typing import List, Any
from smolagents import tool

@tool
def fetch_images_from_github(repo_owner: str, repo_name: str, folder_path: str, branch: str = "main") -> list:
    """
    Downloads images from a GitHub repository folder and returns them as a list of PIL Image objects.

    Args:
        repo_owner: The GitHub username or organization (e.g., 'grishakalinin2014-alt').
        repo_name: The repository name (e.g., 'UAV-Agent').
        folder_path: Path to the folder inside the repository (e.g., 'images/blank').
        branch: The branch name. Defaults to 'main'.

    Returns:
        list: A list of PIL.Image objects sorted numerically by their filenames (Max 5).
    """
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}?ref={branch}"
    
    print(f"📡 GitHub API Request: {api_url}")
    response = requests.get(api_url)
    response.raise_for_status()
    
    files = response.json()
    image_files = [f for f in files if f["name"].lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    def extract_number(name):
        match = re.match(r"(\d+)", name)
        return int(match.group(1)) if match else float('inf')

    image_files.sort(key=lambda x: extract_number(x["name"]))
    
    # 🔥 Ограничиваем список первыми 5 элементами
    image_files = image_files[:5]
    
    loaded_images = []
    for file_info in image_files:
        print(f"📥 Downloading: {file_info['name']}...")
        img_response = requests.get(file_info["download_url"])
        img_response.raise_for_status()
        
        img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
        loaded_images.append(img)
        
    print(f"✅ Successfully loaded {len(loaded_images)} images from GitHub.")
    return loaded_images


def load_local_images(folder_path: str) -> List[Tuple[Image.Image, str]]:
    """Загружает все изображения из локальной папки и сортирует их по числовому значению в имени."""
    images = []
    
    # 1. Просто получаем список всех файлов
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    # 2. Загружаем все изображения
    for filename in files:
        path = os.path.join(folder_path, filename)
        img = Image.open(path).convert("RGB")
        images.append((img, filename))
        print(f"Загружено локально: {filename}")
        
    # 3. Сортируем список кортежей (img, filename) ПОСЛЕ загрузки
    # Извлекаем число из строки '1_square.jpg' или '1.jpg'
    images.sort(key=lambda item: int(''.join(filter(str.isdigit, item[1])) or 0))
    
    print(f"✅ Сортировка завершена. Всего изображений: {len(images)}")
    return images