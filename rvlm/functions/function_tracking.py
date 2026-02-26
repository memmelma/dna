import time
import numpy as np
from rvlm.functions.function_helpers import call_gemini, _parse_json, _get_tracker
from rvlm.requests.helpers import get_obj_labels_prompt, get_obj_points_prompt, get_obj_paths_prompt
from rvlm.utils.conversion import gemini_to_image_frame

_label_cache: dict = {}
_point_cache: dict = {}
_reference_cache: dict = {}

async def video_to_dense_reward(video, language_instruction):
    start_time = time.time()
    tracker = _get_tracker()

    img = video[0]
    H, W = img.shape[:2]
    
    if language_instruction not in _label_cache:
        prompt = get_obj_labels_prompt(language_instruction)
        res = await call_gemini(prompt, img_input=img, thinking_level="LOW")
        # labels = get_obj_labels(language_instruction, img, temperature=0.0, thinking_budget=0)
        # print("labels", res.text)
        _label_cache[language_instruction] = res.text
    else:
        print("using cached labels!")
    labels = _label_cache[language_instruction]
    # print("labels", labels)

    print("WARNING: removing gripper and robot labels!")
    labels = [l for l in labels if (("gripper" not in l) and ("robot" not in l))]
    
    if language_instruction not in _point_cache:
        prompt = get_obj_points_prompt(language_instruction, labels)
        res = await call_gemini(prompt, img_input=img, thinking_level="LOW")
        # points_g = get_obj_points_from_labels(language_instruction, img, labels, temperature=0.0, thinking_budget=0)
        # print("points_g", res.text)
        _point_cache[language_instruction] = _parse_json(res.text)
    else:
        print("using cached points!")
    points_g = _point_cache[language_instruction]
    points_i = gemini_to_image_frame(points_g, W, H)
    # print("points_i", points_i)

    if language_instruction not in _reference_cache:
        prompt = get_obj_paths_prompt(language_instruction, points_g, 7)
        res = await call_gemini(prompt, img_input=img, thinking_level="MEDIUM")
        # predicted_path_g = get_obj_paths_from_points(
        #     task=language_instruction, img=img, points=points_g,
        #     n_points=7, temperature=0.2, thinking_budget=-1,
        # )
        # print("predicted_path_g", res.text)
        _reference_cache[language_instruction] = _parse_json(res.text)
    else:
        print("using cached predicted path!")
    predicted_path_g = _reference_cache[language_instruction]
    predicted_path_i = gemini_to_image_frame(predicted_path_g, W, H)

    tracked_path_i = tracker.track_paths(video, points_i)

    per_obj_rewards, per_obj_alignments = tracker.compute_reward(predicted_path_i, tracked_path_i, return_alignments=True)

    per_obj_rewards_arr = np.stack(list(per_obj_rewards.values()), axis=0)  # (num_objects, T)
    mean_rewards = np.mean(per_obj_rewards_arr, axis=0)
    end_time = time.time()
    print(f"Time taken for video_to_dense_reward: {end_time - start_time} seconds")
    return tracked_path_i, predicted_path_i, per_obj_rewards, per_obj_alignments, mean_rewards