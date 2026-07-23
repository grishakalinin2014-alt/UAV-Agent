import os
import config
import time
import csv
import litellm
import urllib.request
import httpx
import litellm
import matplotlib
from smolagents import LiteLLMModel
from src.data_loader import fetch_images_from_github, load_local_images
from src.tools_Gemini import (
    describe_satellite_image_Gemini,
    pixelpoint_objects_Gemini,
    detect_and_display_Gemini
)
from src.tools import (
    describe_satellite_image,
    pixelpoint_objects, 
    visualize_keypoints_from_image, 
    uav_simulation, 
    detect_and_display,
    resolve_objects_from_query
)
from src.agent_factory import create_uav_agent, create_airspace_manager

urllib.request.getproxies = lambda: {}

for key in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(key, None)
os.environ['no_proxy'] = '*'

matplotlib.use('Agg')  # Отключает GUI-движок Tkinter, оставляя чистую генерацию картинок в памяти
# litellm._turn_on_debug()  # Включает детальный вывод сетевых запросов и ошибок API

COMMANDS = {
    "fire": "Analyze the area for fire. Locate the source and report the extent.",
    "oil_spill": "Detect any signs of an oil spill in the water bodies or on the ground.",
    "car_accident": "A car accident has occurred. The police are on the scene.",
    "hogweed_thickets": "Hogweed thickets are growing in the area.",
    "unauthorized_festival": "Unauthorized festival is taking place in the area.",
}


def main():

    # base_path = "data"
    
    # print("Загрузка данных из локальной папки...")
    
    # Загружаем изображения, если папки существуют
    # load_local_images должна возвращать список кортежей (image, filename)
    # blank_images = load_local_images(os.path.join(base_path, "default"))
    # fire_images = load_local_images(os.path.join(base_path, "fire"))
    # hogweed_images = load_local_images(os.path.join(base_path, "hogweed_thickets"))
    # oil_spill_images = load_local_images(os.path.join(base_path, "oil_spill"))
    # car_images = load_local_images(os.path.join(base_path, "cars_accident"))
    # unauthorized_festival_images = load_local_images(os.path.join(base_path, "unauthorized_festival"))

    # if not blank_images or not fire_images:
    #     print("Ошибка: Изображения не найдены в папках data/default или data/fire!")
    #     return
    #     # Загружаем обе папки: "blank" (база) и "fire" (цель)

    # print("Загрузка данных с GitHub...")
        # Передаем аргументы явно, сопоставляя ключи из config с ожидаемыми именами
    # Объявляем переменные настроек внутри функции
    # blank_settings = config.REPO_SETTINGS["default"]
    # fire_settings = config.REPO_SETTINGS["fire"]


    # blank_images = fetch_images_from_github(
    #     repo_owner=blank_settings["owner"],
    #     repo_name=blank_settings["repo"],
    #     folder_path=blank_settings["folder"],
    #     branch=blank_settings["branch"]
    # )
    
    # fire_images = fetch_images_from_github(
    #     repo_owner=fire_settings["owner"],
    #     repo_name=fire_settings["repo"],
    #     folder_path=fire_settings["folder"],
    #     branch=fire_settings["branch"]
    # )
    # print("Загрузка данных с GitHub...")

    # if not blank_images or not fire_images:
    #     print("Ошибка: изображения не были загружены!")
    #     return

    # Берем по одному изображению для теста
    # base_image = blank_images[0][0]    # Чистая зона
    # target_image = fire_images[0][0] 
    # target_image = hogweed_images[0][0]   
    # target_image = unauthorized_festival_images[0][0]   
    # target_image = oil_spill_images[0][0]   
    # target_image = car_images[0][0]   

    # print(f"--- ШАГ 1: Анализ базовой обстановки ---")
    # description = describe_satellite_image_Gemini(base_image)
    # print(f"Описание модели: {description}\n")

    # print(f"--- ШАГ 2: Детекция объектов на целевом изображении ---")
    # objects_to_find = "place for unauthorized festival", "open space"
    # objects_to_find = "buildings", "warehousesq", "houses"
    # objects_to_find = "place for oil spill", "place where maybe oil spill is", "open space in water"
    # objects_to_find = "car accident", "crossroads", "road junction", "road intersection"
    # objects_to_find = "hogweed thickets", "open space", "field", "meadow"
    # objects_to_find = resolve_objects_from_query(input("Введите запрос: "))
    # keypoints = pixelpoint_objects_Gemini(image=base_image, objects=objects_to_find)
    # print(f"Найдено объектов: {len(keypoints)}, {keypoints}")


    # keypoints = [{'point': [222, 573], 'label': "('House', 'Warehouse')"}]

    # keypoints = [
    #     {'point': [518, 496], 'label': 'Warehouse'},
    #     {'point': [468, 503], 'label': 'Warehouse'},
    #     {'point': [965, 723], 'label': 'House'}, 
    #     {'point': [409, 836], 'label': 'House'}, 
    #     {'point': [428, 829], 'label': 'House'}, 
    #     {'point': [872, 781], 'label': 'House'}, 
    #     {'point': [574, 576], 'label': 'House'}, 
    #     {'point': [617, 706], 'label': 'House'}, 
    #     {'point': [674, 629], 'label': 'House'}, 
    #     {'point': [456, 730], 'label': 'House'}]
    # target_obj = "fire"
    # target_obj = "hogweed thickets"
    # target_obj = "unauthorized festival"
    # target_obj = "oil spill"
    # target_obj = "cars accident"
    # print(f"--- ШАГ 3: Визуализация ---")
    # visualize_keypoints_from_image(image=base_image, keypoints=keypoints)

    # print(f"--- ШАГ 4: Симуляция полета БПЛА ---")
    # frames_dict = uav_simulation(image=target_image, labeled_points=keypoints)

    # print(f"--- ШАГ 5: Финальный анализ кадров на наличие {target_obj}  ---")
    # fire_coords = detect_and_display_Gemini(frames_dict=frames_dict, target_object=target_obj)

    # if fire_coords:
    #     print(f"\n✅ Миссия завершена: {target_obj} обнаружен в координатах {fire_coords}")
    # else:
    #     print(f"\n✅ Миссия завершена: {target_obj} не обнаружен.")




    # Инициализация базовой модели через GeminiModel(Gemini 2.5 Flash)
    model = LiteLLMModel(
        model_id="gemini/gemini-3.5-flash",  
        api_key=config.GOOGLE_API_KEY,
    )

    # Создаем двух агентов
    uav_agent = create_uav_agent(model)
    airspace_manager_agent = create_airspace_manager(model, uav_agent)

    # Выбор сценария
    print("Доступные сценарии:", list(COMMANDS.keys()))
    selected_key = input("Введите ключ сценария (или нажмите Enter для тестирования 'fire'): ").strip()
    if not selected_key or selected_key not in COMMANDS:
        selected_key = "fire"
        
    user_request = COMMANDS[selected_key]
    print(f"\n📝 Выбран запрос: '{user_request}'")

    repo_owner = "grishakalinin2014-alt"
    repo_name = "UAV-Agent"
    branch = "main"

    results = []

    # Цикл бенчмарка из ваших примеров (5 итераций экспериментов)
    for i in range(0, 5):
        print(f"\n=== 🏁 ЗАПУСК ЭКСПЕРИМЕНТА №{i} ===")
        start_time = time.time()

        # Формируем динамическую инструкцию для Менеджера
        # Формируем полностью динамическую инструкцию для Менеджера
        task_prompt = f"""
        You are an intelligent Airspace Manager. Your job is to analyze the user's request, download maps and target images dynamically from GitHub, find infrastructure coordinates on a baseline map, and coordinate a UAV verification flight.

        User Request: "{user_request}"
        Current Experiment Index (Image Index to process): {i}
        
        GitHub Repository Configuration:
        - Owner: "{repo_owner}"
        - Repo Name: "{repo_name}"
        - Branch: "{branch}"
        - Baseline Folder: "images/blank"

        Follow this execution plan precisely in your generated Python code:
        1. Download BASELINE images from GitHub using the 'fetch_images_from_github' tool with owner, repo, branch, and folder="images/blank".
           Select the {i}-th image as your clean reference map: baseline_image = baseline_images[{i}].
        
        2. Call 'get_folder_by_query' with the User Request. Use the 'json' module to parse the resulting JSON string:
           import json
           incident_data = json.loads(get_folder_by_query("{user_request}"))
           folder = incident_data["folder"]
           target_object = incident_data["target_object"]
           incident_type = incident_data["incident_type"]
           CRITICAL: Pass the user request string strictly as a positional argument or using the exact parameter name 'user_query', like this: get_folder_by_query(user_query="{user_request}").
        
        3. Determine the TARGET GitHub folder path. Map the extracted folder name strictly to the repository structure:
           - If folder is "fire", the target GitHub folder path is "images/fire_buildings".
           - If folder is "hogweed_thickets", the target GitHub folder path is "images/hogweed_thickets".
           - For any other folder, use the exact "images/" + folder_name structure.
        
        4. Call 'resolve_objects_from_query' with the User Request to get the types of static infrastructure objects to inspect (e.g., crossroads, buildings).
        5. Pass 'baseline_image' and the infrastructure objects string to 'pixelpoint_objects_Gemini' to detect coordinates on the clean map.
        6. Visualize these detected keypoints ON THE BASELINE IMAGE using 'visualize_keypoints_from_image(image=baseline_image, keypoints=...)'.
        
       7. MANDATORY UAV DELEGATION (CRITICAL): 
           You MUST ALWAYS execute the dynamic flight verification workflow for EVERY single incident type, including static ones (such as hogweed thickets, infrastructure damage, trash piles, etc.). NEVER skip this step or complete the task based only on the baseline map.
           
           Call 'uav_agent' by providing a text prompt containing:
           - The exact target folder path on GitHub (e.g., "images/fire_buildings" or the folder mapped in Step 3).
           - The target image index to process (which is {i}).
           - The 'target_object' name to search for (e.g., extracted "target_object" from Step 2).
           - The generated coordinates array (keypoints) from Step 5.
           
           Instruct the 'uav_agent' to precisely write and run this Python code structure:
           a) Download the target images from GitHub using 'fetch_images_from_github'.
           b) Select the target_image = target_images[{i}].
           c) Generate the frames dictionary: frames_dict = uav_simulation(image=target_image, labeled_points=keypoints).
           d) Pass the 'frames_dict' entirely into 'detect_and_display_Gemini' without ANY manual 'for' loops over individual frames.
           e) Immediately return the resulting coordinates or status to you.

        8. FINAL ANSWER FORMULATION:
           Formulate your final answer ONLY after the uav_agent completes the dynamic flight verification and returns the frame-by-frame detection results.
        Provide the final coordinates or a verification summary as your final answer.
        """

        try:
            # Запуск цепочки через главного агента
            experiment_outcome = airspace_manager_agent.run(task_prompt)
        except Exception as e:
            experiment_outcome = f"Execution failed with error: {str(e)}"
            print(f"❌ Ошибка на итерации {i}: {e}")

        elapsed_time = time.time() - start_time
        print(f"⏱️ Эксперимент {i} завершен за {elapsed_time:.2f} сек.")

        results.append({
            'Experiment Number': i,
            'Scenario': selected_key,
            'Request Text': user_request,
            'Elapsed Time': f"{elapsed_time:.2f}",
            'Outcome': experiment_outcome
        })

    # Сохраняем логи в CSV-файл по завершении итераций
    csv_file_path = f'experiment_results_{selected_key}.csv'
    fieldnames = ['Experiment Number', 'Scenario', 'Request Text', 'Elapsed Time', 'Outcome']

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)

    print(f"\n📊 Все эксперименты завершены! Данные сохранены в {csv_file_path}")



if __name__ == "__main__":
    main()