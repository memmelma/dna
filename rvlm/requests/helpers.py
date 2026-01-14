import json
import textwrap
import numpy as np
from rvlm.requests.gemini_utils import create_config, img_to_mime, call_gemini_robotics_er

def get_obj_labels_prompt(task: str):
    prompt = textwrap.dedent("""\
    List the object parts and descriptors (e.g., top, bottom, handle, ...) relevant to the task: {{TASK}}

    The answer should follow the JSON format:
    [{"label": <object_name>_<descriptor>}, ...]
    """)
    prompt = prompt.replace("{{TASK}}", task)
    return prompt

def get_obj_points_prompt(task: str, obj_labels: list[dict]):
 
    prompt = textwrap.dedent("""\
    Point to object parts {{LABELS}}:

    The answer should follow the JSON format:
    [{"label": <object_name>_<label_0>, "point": [<point>]}, ...]

    The points are in [y, x] format normalized to 0-1000.""")

    prompt = prompt.replace("{{TASK}}", task)
    prompt = prompt.replace("{{LABELS}}", json.dumps(obj_labels))

    return prompt

def get_obj_paths_prompt(task: str, points: list[dict], n_points: int):
    
    prompt = textwrap.dedent("""\
        List {{N}} points for the trajectory each object in {{OBJECTS}} must follow to complete {{TASK}}.
        Ensure the objects don't collide on their way and end up at the goal.

        You are given the following initial points {{POINTS}}

        The points should be labeled by order of the trajectory, from '0' (start
        point at left hand) to <n> (final point).

        The answer should follow the JSON format:
        [{"label": <object_name>_<label_0>, "points": [<point_0>, <point_1>, ...]}, ...]

        The points are in [y, x] format normalized to 0-1000.""")

    prompt = prompt.replace("{{TASK}}", task)
    prompt = prompt.replace("{{OBJECTS}}", json.dumps([p["label"] for p in points]))
    prompt = prompt.replace("{{POINTS}}", json.dumps(points))
    prompt = prompt.replace("{{N}}", str(n_points))

    return prompt

def get_obj_labels(task: str, img: np.ndarray, temperature: float = 0.2, thinking_budget: int = -1):
    
    prompt = get_obj_labels_prompt(task)
    config = create_config(temperature=temperature, thinking_budget=thinking_budget)
    
    mime = img_to_mime(img)
    json_output = call_gemini_robotics_er(mime, prompt, config)

    return json.loads(json_output)

def get_obj_points_from_labels(task: str, img: np.ndarray, labels: list[dict], temperature: float = 0.2, thinking_budget: int = 100):
    
    prompt = get_obj_points_prompt(task, labels)
    config = create_config(temperature=temperature, thinking_budget=thinking_budget)
    
    mime = img_to_mime(img)
    json_output = call_gemini_robotics_er(mime, prompt, config)
    return json.loads(json_output)

def get_obj_paths_from_points(task: str, img: np.ndarray, points: list[dict], n_points: int = 7, temperature: float = 0.2, thinking_budget: int = -1):

    prompt = get_obj_paths_prompt(task, points, n_points)
    config = create_config(temperature=temperature, thinking_budget=thinking_budget)
    
    mime = img_to_mime(img)
    json_output = call_gemini_robotics_er(mime, prompt, config)
    return json.loads(json_output)

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