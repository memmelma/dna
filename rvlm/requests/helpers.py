import json
import textwrap
import asyncio
import numpy as np
from rvlm.requests.gemini_utils import create_config, img_to_mime, call_gemini_robotics_er, call_gemini_robotics_er_async

def get_obj_labels_prompt(task: str):
    prompt = textwrap.dedent("""\
    List the object parts AND (multiple) descriptors (e.g., top, bottom, handle, center, left, right, blue, green, transparent, ...) that must move to complete the task: {{TASK}}
    Only list minimal set and exclude unnecessary. Include the robot's gripper.
    
    The answer should follow the JSON format:
    [{"label": <object_name>_<descriptors>}, ...]
    """)
    prompt = prompt.replace("{{TASK}}", task)
    return prompt

def get_obj_points_prompt(task: str, obj_labels: list[dict]):
 
    prompt = textwrap.dedent("""\
    Point EXACTLY to the following object parts: {{LABELS}}
    All object parts exist in the image.

    The answer should follow the JSON format:
    [{"label": <object_label>, "point": [<point>]}, ...]

    The points are in [y, x] format normalized to 0-1000.""")

    prompt = prompt.replace("{{TASK}}", task)
    prompt = prompt.replace("{{LABELS}}", json.dumps(obj_labels))

    return prompt

def get_obj_paths_prompt(task: str, points: list[dict], n_points: int):
    
    prompt = textwrap.dedent("""\
        List {{N}} points for the trajectory each object in {{OBJECTS}} must follow to complete {{TASK}}. Mimic how the objects should move when grasped by a robot, i.e., consider collision avoidance, goal reaching, smooth arching motions, etc.

        You are given the following initial points {{POINTS}}

        The points should be labeled by order of the trajectory, from '0' (start
        point at left hand) to <n> (final point).

        Smooth example motion:
        [{"label": <object_label>, "points": [[185 104], [84, 74], [46, 54], [34, 30], [26, 51]]}]

        The answer should follow the JSON format:
        [{"label": <object_label>, "points": [<point_0>, <point_1>, ...]}, ...]

        The points are in [y, x] format normalized to 0-1000.""")

    prompt = prompt.replace("{{TASK}}", task)
    prompt = prompt.replace("{{OBJECTS}}", json.dumps([p["label"] for p in points]))
    prompt = prompt.replace("{{POINTS}}", json.dumps(points))
    prompt = prompt.replace("{{N}}", str(n_points))

    return prompt


def request_gemini(img: np.ndarray, prompt: str, temperature: float = 0.2, thinking_budget: int = -1):
    config = create_config(temperature=temperature, thinking_budget=thinking_budget)
    mime = img_to_mime(img)
    json_output = call_gemini_robotics_er(mime, prompt, config)
    return json.loads(json_output)

async def request_gemini_async(img: np.ndarray, prompt: str, temperature: float = 0.2, thinking_budget: int = -1):
    config = create_config(temperature=temperature, thinking_budget=thinking_budget)
    mime = img_to_mime(img)
    json_output = await call_gemini_robotics_er_async(mime, prompt, config)
    return json.loads(json_output)

def request_gemini_batch(imgs: list[np.ndarray], prompts: list[str], temperature: float = 0.2, thinking_budget: int = -1):
    async def request_async(imgs, prompts):
        return await asyncio.gather(*[request_gemini_async(img=img, prompt=prompt, temperature=temperature, thinking_budget=thinking_budget) for img, prompt in zip(imgs, prompts)])
    return asyncio.run(request_async(imgs, prompts))


def get_obj_labels(task: str, img: np.ndarray, temperature: float = 0.2, thinking_budget: int = -1):
    prompt = get_obj_labels_prompt(task)
    return request_gemini(img=img, prompt=prompt, temperature=temperature, thinking_budget=thinking_budget)

def get_obj_labels_batch(task: str, imgs: list[np.ndarray], temperature: float = 0.2, thinking_budget: int = -1):
    prompts = [get_obj_labels_prompt(task) for _ in range(len(imgs))]
    return request_gemini_batch(imgs=imgs, prompts=prompts, temperature=temperature, thinking_budget=thinking_budget)


def get_obj_points_from_labels(task: str, img: np.ndarray, labels: list[dict], temperature: float = 0.2, thinking_budget: int = 100):
    prompt = get_obj_points_prompt(task, labels)
    return request_gemini(img=img, prompt=prompt, temperature=temperature, thinking_budget=thinking_budget)

def get_obj_points_from_labels_batch(task: str, imgs: list[np.ndarray], labels: list[dict], temperature: float = 0.2, thinking_budget: int = 100):
    prompts = [get_obj_points_prompt(task, labels) for _ in range(len(imgs))]
    return request_gemini_batch(imgs=imgs, prompts=prompts, temperature=temperature, thinking_budget=thinking_budget)


def get_obj_paths_from_points(task: str, img: np.ndarray, points: list[dict], n_points: int = 7, temperature: float = 0.2, thinking_budget: int = -1):
    prompt = get_obj_paths_prompt(task, points, n_points)
    return request_gemini(img=img, prompt=prompt, temperature=temperature, thinking_budget=thinking_budget)

def get_obj_paths_from_points_batch(task: str, imgs: list[np.ndarray], points: list[dict], n_points: int = 7, temperature: float = 0.2, thinking_budget: int = -1):
    prompts = [get_obj_paths_prompt(task, points, n_points) for _ in range(len(imgs))]
    return request_gemini_batch(imgs=imgs, prompts=prompts, temperature=temperature, thinking_budget=thinking_budget)


def postprocess_obj_paths(obj_paths: list[dict], H: int, W: int):
    obj_paths_dict = {}
    for p in obj_paths:

        point = p.get("point", None)
        points = p.get("points", None)

        if points:
            obj_path = np.array(points) 
                    # y, x -> x, y
            obj_path = obj_path[:, ::-1]
        elif point:
            obj_path = np.array(point)
            obj_path = obj_path[::-1]
        else:
            raise ValueError(f"No point or points found for {p['label']}")

        # convert to pixel coordinates [0, 1000] -> [0, H], [0, W]
        obj_path = obj_path / 1000.0 * np.array([H, W])
        # convert to int
        obj_path = obj_path.astype(np.int32)                
        obj_paths_dict[p["label"]] = obj_path

    return obj_paths_dict