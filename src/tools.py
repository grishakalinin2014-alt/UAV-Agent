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


model = config.MODEL_VL
path = config.OPENROUTER_BASE_URL


def get_headers():
    return {
        "Authorization": f"Bearer {config.FIREWORKS_API_KEY}",
        "HTTP-Referer": "https://colab.research.google.com",
        "X-Title": "UAV_AGENT",
        "Content-Type": "application/json"
    }

@tool
def describe_satellite_image(image: Image.Image) -> str:
    """
    Analyze one image using Qwen3.0-VL through Fireworks API.

    Args:
        image: A PIL.Image.Image object to analyze.

    Returns:
        str: The description of the image.
    """
    api_key = config.FIREWORKS_API_KEY
    if not api_key:
        return "Error: No API key provided in request."

    # Convert to RGB if needed
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Prepare content for the API
    content = [
        {"type": "text", "text": '''
        Describe this image in very detailed manner.
        Name each object you observe and its approximate location.
        Example: "I see 6 buildings in the upper part of the image. A road crosses from left to right."
        '''},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
    ]


    payload = {
        "model": model,
        "max_tokens": 1000,
        "temperature": 0.5,
        "top_p": 1,
        "top_k": 40,
        "messages": [{"role": "user", "content": content}]
    }

    headers = {
        "Accept": "application/json",
        "X-Title": "Satellite Analysis App",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(
        path,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        return f"Error {response.status_code}: {response.text}"
    
    # Попытка парсинга
    try:
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"JSON Parse Error: {e}. Raw response: {response.text[:500]}"
    else:
        return f"Error {response.status_code}: {response.text}"
    



@tool
def pixelpoint_objects(image: Image.Image, objects: str) -> List[dict]:
    """
    Pixelpoint the image using Qwen2.5-VL through API. Includes auto-fix for cut-off JSON responses.

    Args:
        image (PIL.Image.Image): PIL image to analyze.
        objects (str): The list of the objects that are needed to be pixelpointed.

    Returns:
        list: Dictionary of detected object labels and their coordinates.
    """
    api_key = config.FIREWORKS_API_KEY

    if not api_key:
        return "❌ Error: No API key found in user data."

    model_name = config.MODEL_VL
    max_tokens = 500
    #temperature = 0.3

    # Prepare prompt content
    content = [{
        "type": "text",
        "text": f'''1. 1. Return a JSON list of {objects}: [{{"point_2d": [x, y], "label": "building"}}]. 
            Output strictly valid JSON. Do not include any other text.
2. ABSOLUTELY NO LINES, NO PATHS, NO CONNECTING ARROWS.
3. If an object is not a house, do not include it.
4. Output ONLY the JSON block. Do not write any conversational text.
'''
    }]

    # Convert images to base64 and attach

    if image.mode != "RGB":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{img_base64}"
        }
    })

    # Send request
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 40,
        "messages": [{"role": "user", "content": content}]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.post(
        path,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    # 2. Проверяем, не пустой ли ответ
    if not response.text.strip():
        raise Exception("API returned an empty response!")

    # 3. Безопасное чтение JSON
    try:
        data = response.json()
        raw_content = data['choices'][0]['message']['content']
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise Exception(f"Failed to parse API response. Raw response: {response.text[:500]}")

    if response.status_code == 200:
      raw_content = response.json()['choices'][0]['message']['content']
      #print(raw_content)

      # Extract JSON from markdown
      match = re.search(r'```json\s*(.*?)\s*```', raw_content, re.DOTALL)
      json_str = match.group(1) if match else raw_content.strip()

      # First attempt at direct JSON parse
      try:
          return json.loads(json_str)
      except json.JSONDecodeError:
          print("⚠️ JSON parsing failed. Attempting to recover...")

          # Extract all valid object-like entries using regex
          pattern = re.compile(r'\{[^{}]*"point_2d"\s*:\s*\[\s*\d+\s*,\s*\d+\s*\]\s*,\s*"label"\s*:\s*"[^"]+"\s*\}')
          objects = pattern.findall(json_str)

          if not objects:
              raise ValueError("❌ Could not recover any valid object entries from the malformed JSON.")

          repaired = '[\n' + ',\n'.join(objects) + '\n]'
          try:
              return json.loads(repaired)
          except Exception as e:
              raise ValueError(f"❌ Final JSON repair failed. First 1000 chars:\n{raw_content[:1000]}\n\nError: {e}")



@tool
def visualize_keypoints_from_image(image: Image.Image, keypoints: list) -> str:
    """
    Visualizes the objects on the image and displays them.
    Usage: visualize_keypoints_from_image(images = loaded_images, keypoints = keypoints)

    Args:
        image (PIL.Image.Image): A PIL image object.
        keypoints (list): A list of keypoints, where each item is a dict with:
            - point_2d: a list of [x, y] coordinates.
            - label: a string representing the label for the point.

    Returns:
        str: Confirmation message after displaying image.
    """
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    


    for idx, kp in enumerate(keypoints):
        x, y = map(int, kp["point_2d"])
        label = f"{idx + 1}. {kp['label']}"
        x = int(x * w / 1000) 
        y = int(y * h / 1000)

        if idx == 0:
            # Start point: solid red circle
            draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill="red")
        else:
            # Other points: red ring
            draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], outline="red", width=2)

        draw.text((x + 8, y - 8), label, fill="blue", font=font)

        # Draw arrow to next point if not the last
        if idx < len(keypoints) - 1:
            next_x, next_y = map(int, keypoints[idx + 1]["point_2d"])
            next_x = int(next_x * w / 1000) 
            next_y = int(next_y * h / 1000)
            draw.line((x, y, next_x, next_y), fill="#00BFFF", width=1)

            # Draw arrowhead
            dx, dy = next_x - x, next_y - y
            length = max((dx**2 + dy**2) ** 0.5, 1)
            ux, uy = dx / length, dy / length
            arrow_size = 8
            left_x = next_x - arrow_size * (ux + uy / 2)
            left_y = next_y - arrow_size * (uy - ux / 2)
            right_x = next_x - arrow_size * (ux - uy / 2)
            right_y = next_y - arrow_size * (uy + ux / 2)
            draw.polygon([(next_x, next_y), (left_x, left_y), (right_x, right_y)], fill="#00BFFF")

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    return f"Displayed image with {len(keypoints)} keypoints and bright arrows between them."
@tool


def uav_simulation(image: Image.Image, labeled_points: list) -> Dict[str, Tuple[Image.Image, Tuple[int, int]]]:
    """
    Simulates the flight of UAV along the provided coordinates, displays a flight GIF, 
    and returns cropped frames around each position.

    Args:
        image (PIL.Image.Image): Source baseline image.
        labeled_points (list): A list of dictionaries containing coordinate information and labels.

    Returns:
        Dict[str, Tuple[Image.Image, Tuple[int, int]]]:
            Dictionary mapping frame names to (cropped image, center coordinates).
    """
    crop_size: int = 60
    steps_per_segment: int = 10
    width, height = image.size

    frame_images = []
    crop_dict = {}
    frame_counter = 1

    # --- ЖЕСТКАЯ ЗАЩИТА ОТ ЛЮБЫХ ФОРМАТОВ ---
    normalized_points = []
    for kp in labeled_points:
        if not isinstance(kp, dict):
            continue
            
        # Проверяем вообще все ключи, где могут лежать координаты
        coord = None
        for key in ['point_2d', 'point', 'coordinates', 'location']:
            if key in kp:
                coord = kp[key]
                break
                
        if coord:
            normalized_points.append({
                'point_2d': coord,
                'label': kp.get('label', 'target')
            })
            
    if len(normalized_points) < 2:
        raise ValueError(f"Недостаточно валидных точек для симуляции. Найдено точек: {len(normalized_points)}")

    # Симуляция по нормализованному списку
    for i in range(len(normalized_points) - 1):
        start = np.array(normalized_points[i]['point_2d'], dtype=np.float32)
        end = np.array(normalized_points[i + 1]['point_2d'], dtype=np.float32)

        interpolated_positions = (np.linspace(0, 1, steps_per_segment)[:, None] * end +
                                  (1 - np.linspace(0, 1, steps_per_segment)[:, None]) * start).astype(int)

        for pos in interpolated_positions:
            x, y = pos
            left = max(x - crop_size, 0)
            upper = max(y - crop_size, 0)
            right = min(x + crop_size, width)
            lower = min(y + crop_size, height)

            if right <= left or lower <= upper:
                continue

            crop = image.crop((left, upper, right, lower))
            crop = crop.resize((crop_size * 2, crop_size * 2))

            draw = ImageDraw.Draw(crop)
            draw.text((5, 5), f"Frame :{frame_counter}", fill='yellow')

            frame_name = f"frame_{frame_counter:03d}"
            crop_dict[frame_name] = (crop, (int(x), int(y)))
            frame_images.append(crop)
            frame_counter += 1

    if not crop_dict:
        raise ValueError("No valid frames could be generated. Check the labeled points and crop size.")

    try:
        gif_buffer = BytesIO()
        imageio.mimsave(gif_buffer, [np.array(frame) for frame in frame_images], duration=0.1, format='GIF')
        gif_buffer.seek(0)
        display(IPImage(data=gif_buffer.read(), format='png'))
    except Exception as e:
        print(f"Показ GIF пропущен или запущен вне Jupyter: {e}")

    return crop_dict

@tool
def detect_and_display(frames_dict: Dict[str, Tuple[Image.Image, Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    """
    Detect fire in UAV frames and return coordinates of first fire detected, otherwise None.

    Args:
        frames_dict (dict): Mapping of frame names to (PIL.Image, (x, y)) tuples.

    Returns:
        Optional[Tuple[int, int]]: Coordinates (x, y) if fire is detected, else None.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.FIREWORKS_API_KEY}" # Используем ключ из конфига
    }

    print("🔍 Starting fire detection on every third frame...\n")

    for idx, (frame_name, (img, coords)) in enumerate(frames_dict.items()):
        if idx % 4 != 0:
            continue

        print(f"🖼️ Checking {frame_name} — Center: {coords}")
        # plt.imshow(img)
        # plt.title(f"Frame: {frame_name}")
        # plt.axis('off')
        # plt.show() # Это откроет окно с картинкой или отобразит её в консоли
        
        # Convert image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data_url = f"data:image/png;base64,{img_base64}"

        # Prepare API payload
        payload = {
            "model": config.MODEL_VL,
            "max_tokens": 500,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do you observe any fire? Say only YES or NO"},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ]
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"❌ Error with frame {frame_name}: {response.text}")
            continue

        description = response.json()['choices'][0]['message']['content']
        print(f"🧠 Model response for {frame_name}: {description}")

        if "YES" in description.upper():
            print(f"\n🔥 Fire detected at coordinates {coords}!")
            plt.imshow(img)
            plt.title(f"Frame: {frame_name}")
            plt.axis('off')
            plt.show() # Это откроет окно с картинкой или отобразит её в консоли
            return coords

    print(f"\n✅ No fire detected in any of the frames.")
    return None


@tool
def resolve_objects_from_query(user_query: str) -> str:
    """
    Analyzes the user's text query, identifies the scenario, and returns 
    a formatted string of objects to find for the Gemini prompt.

    Args:
        user_query: The raw string query from the user (e.g., 'Hogweed thickets are growing').

    Returns:
        str: A comma-separated string of objects (e.g., '"buildings", "warehouses", "houses"').
    """
    query_lower = user_query.lower()

    # Проверяем ключевые слова и сразу отдаем нужную строку объектов
    if "fire" in query_lower:
        return '"buildings", "warehouses", "houses"'
        
    elif "oil" in query_lower or "spill" in query_lower:
        return '"place for oil spill", "place where maybe oil spill is", "open space in water"'
        
    elif "car" in query_lower or "accident" in query_lower:
        return '"car accident", "crossroads", "road junction", "road intersection"'
        
    elif "hogweed" in query_lower:
        return '"hogweed thickets", "open space", "field", "meadow"'
        
    elif "festival" in query_lower or "unauthorized" in query_lower:
        return '"place for unauthorized festival", "open space"'

    # Вариант по умолчанию, если ничего не подошло
    return '"open space"'

@tool
def get_folder_by_query(user_query: str) -> str:
    """
    Analyzes the user's text query to determine the incident type, 
    the corresponding data folder, and the specific target object to detect.

    Args:
        user_query: The raw string query from the user.

    Returns:
        str: A JSON string with 'folder', 'target_object', and 'incident_type' keys.
             Example: '{"folder": "fire", "target_object": "fire", "incident_type": "fire"}'
    """
    query_lower = user_query.lower()
    
    # Карта соответствий: Ключевые слова -> (Папка, Объект поиска, Тип инцидента)
    mapping = {
        "fire": ("fire", "fire", "fire"),
        "oil": ("oil_spill", "oil spill", "oil_spill"),
        "spill": ("oil_spill", "oil spill", "oil_spill"),
        "car": ("cars_accident", "cars accident", "cars_accident"),
        "accident": ("cars_accident", "cars accident", "cars_accident"),
        "hogweed": ("hogweed_thickets", "hogweed thickets", "hogweed_thickets"),
        "festival": ("unauthorized_festival", "unauthorized festival", "unauthorized_festival"),
        "unauthorized": ("unauthorized_festival", "unauthorized festival", "unauthorized_festival")
    }

    # Ищем совпадение
    for keyword, (folder, target_obj, incident_type) in mapping.items():
        if keyword in query_lower:
            return json.dumps({
                "folder": folder, 
                "target_object": target_obj, 
                "incident_type": incident_type
            })

    # Дефолтный вариант, если тема не распознана
    return json.dumps({
        "folder": "default", 
        "target_object": "unknown", 
        "incident_type": "unknown"
    })