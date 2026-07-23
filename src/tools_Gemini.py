import base64
import io
import requests
import re
import json
import matplotlib.pyplot as plt
import concurrent.futures
import requests
import json
import config
import numpy as np
import imageio
import time
import cv2
from google.oauth2 import service_account
from google.genai import Client
from google import genai
from google.genai import types
from smolagents import tool
from smolagents.tools import tool
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict
from io import BytesIO
from IPython.display import display, Image as IPImage
from typing import Dict, Tuple, Optional


@tool
def describe_satellite_image_Gemini(image: Image.Image) -> str:
    """
    Analyze one image using Gemini 2.0 Flash via Service Account.

    Args:
        image (Image.Image): A PIL.Image.Image object to analyze.

    Returns:
        str: The description of the image or an error message.
    """
    try:
        # 1. Загрузка учетных данных из файла

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": config.GOOGLE_API_KEY
        }

        

        data = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Describe this image in very detailed manner. Name each object you observe and its approximate location."},
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": img_base64  # Здесь должна быть только сама строка base64
                                }
                            }
                        ]
                    }
                ]
            }

        # 3. Подготовка изображения
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')

        # 4. Запрос к модели
        response = requests.post(config.GOOGLE_URL, headers=headers, json=data)


        return response.text if response and response.text else "Error: Empty response."

    except Exception as e:
        print(f"DEBUG: Auth/API Error: {e}")
        return f"Error: {e}"



@tool
def pixelpoint_objects_Gemini(image: Image.Image, objects: str) -> List[dict]:
    """
    Detects objects and returns JSON list of coordinates with automatic retries.

    Args:
        image (Image.Image): The PIL image object to analyze.
        objects (str): The list of the objects that are needed to be detected.

    Returns:
        List[dict]: A list of detected object labels and their coordinates.
    """
    # 1. Подготовка изображения
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = f"""
    This is the satellite image. Detect the key points of all the {objects}.
    {objects} is defined by a distinct rectangular shape, usually darker or lighter than the surrounding grass, with clear edges.
    Ignore linear objects like roads or fences. Return maximum 15 points of {objects}.
    Provide only the answer in json. Return them STRICTLY in the form:
        ```json
    [
    {{"point_2d": [x1, y1], "label": {objects}}},
    {{"point_2d": [x2, y2], "label": {objects}}}
    ]
    ```
    """

    # prompt = f"""
    # detect red point and return it coordinates
    # Return ONLY valid JSON: [{{ "point_2d": [x, y], "label": "{objects}" }}]
    # """

    data = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inlineData": {"mimeType": "image/jpeg", "data": img_base64}}
        ]}]
    }

    headers = {"Content-Type": "application/json"}
    url = f"{config.GOOGLE_URL}?key={config.GOOGLE_API_KEY}"
    
    # --- ЛОГИКА ПОВТОРОВ ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=90)
            
            # Если всё успешно
            if response.status_code == 200:
                raw_content = response.json()['candidates'][0]['content']['parts'][0]['text']
                json_str = re.sub(r'```json\s*|\s*```', '', raw_content).strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    objects_found = re.findall(r'\{[^{}]*"point_2d"\s*:\s*\[\s*\d+\s*,\s*\d+\s*\]\s*,\s*"label"\s*:\s*"[^"]+"\s*\}', json_str)
                    return json.loads('[\n' + ',\n'.join(objects_found) + '\n]') if objects_found else []

            # Если ошибка 503 (перегрузка) или 429 (лимиты), ждем
            print(f"⚠️ Попытка {attempt+1} вернула ошибку {response.status_code}. Ждем 20 секунд...")
            time.sleep(20)

        except Exception as e:
            print(f"DEBUG: Попытка {attempt+1} не удалась: {e}")
            time.sleep(20)
            
    return []


@tool
def detect_and_display_Gemini(frames_dict: Dict[str, Tuple[Image.Image, Tuple[int, int]]], target_object: str) -> Optional[Tuple[int, int]]:
    """

    Detect {target_object} in UAV frames and return coordinates of first {target_object} detected, otherwise None.


    Args:

    frames_dict (dict): Mapping of frame names to (PIL.Image, (x, y)) tuples.
    target_object (str): The object to search for (e.g., 'fire', 'car', 'person').


    Returns:

    Optional[Tuple[int, int]]: Coordinates (x, y) if {target_object} is detected, else None.

    """ 
    headers = {"Content-Type": "application/json"}
    url = f"{config.GOOGLE_URL_DETECT}?key={config.GOOGLE_API_KEY}"

    print(f"🔍 Starting {target_object} detection on every fourth frame...\n")

    request_count = 0

    for idx, (frame_name, (img, coords)) in enumerate(frames_dict.items()):
        if idx % 4 != 0:
            continue
        

        if request_count > 0 and request_count % 10 == 0:
            print(f"⏳ Достигнут лимит в 10 запросов. Ожидание 60 секунд для обхода Rate Limit...")
            # time.sleep(60)

        request_count += 1


        # 1. Корректное сжатие в JPEG (Gemini лучше работает с ним)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = f"Do you see any {target_object}? Answer ONLY with 'YES' or 'NO'."

        data = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": img_base64}}
            ]}]
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=90)
            if response.status_code != 200:
                print(f"❌ Error with frame {frame_name}: {response.status_code}")
                continue

            # 2. Правильный парсинг для Gemini API
            resp_json = response.json()
            description = resp_json['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # plt.imshow(img)
            # plt.title(f"Fire detected in {frame_name}")
            # plt.axis('off')
            # plt.show() 
            
            print(f"🧠 Model response for {frame_name}: {description}")

            if "YES" in description.upper():
                print(f"\n🎯 {target_object} detected at coordinates {coords}!")
                
                # Безопасно сохраняем в лог файл, никакого plt.show()!
                import os
                try:
                    os.makedirs("uav_logs", exist_ok=True)
                    plt.figure()
                    plt.imshow(img)
                    plt.title(f"{target_object} detected in {frame_name}")
                    plt.axis('off')
                    plt.savefig(f"uav_logs/{target_object}_{frame_name}.png", bbox_inches='tight')
                    plt.close()
                except Exception as visual_err:
                    print(f"Skipping plot save: {visual_err}")
                
                return coords # Мгновенный выход из функции

        except Exception as e:
            print(f"DEBUG: Error parsing response for {frame_name}: {e}")
            continue

    print(f"\n✅ No {target_object} detected.")
    return None