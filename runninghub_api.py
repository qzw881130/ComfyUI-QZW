"""Run a GPT Image 2.5 ComfyUI workflow on RunningHub."""

import copy
import hashlib
import json
import logging
import re
import time

import requests

from .image_api import ImageAPIError, ImageClient, encode_images, safe_response


LOG = logging.getLogger("ComfyUI-QZW")
DEFAULT_WORKFLOW_ID = "2117652716037173249"
IMAGE_CLASS = "RH_RhartImageG25FlareImageToImage"
STABLE_IMAGE_CLASS = "RH_RhartImageG25OfficialTokenFlareEdit"


def workflow_layout(graph):
    if not isinstance(graph, dict):
        raise ValueError("RunningHub 未返回有效的 API 格式工作流。")
    targets = [str(k) for k, v in graph.items() if isinstance(v, dict) and v.get("class_type") == IMAGE_CLASS]
    if len(targets) != 1:
        raise ValueError(f"工作流需要恰好一个 {IMAGE_CLASS} 节点，当前找到 {len(targets)} 个。")
    target = targets[0]
    slots = []
    for field, link in graph[target].get("inputs", {}).items():
        if not re.fullmatch(r"image\d+", field):
            continue
        if (not isinstance(link, list) or len(link) != 2
                or graph.get(str(link[0]), {}).get("class_type") != "LoadImage"):
            raise ValueError(f"节点 {target}.{field} 需要连接到 LoadImage。")
        slots.append((field, str(link[0])))
    slots.sort(key=lambda slot: int(slot[0][5:]))
    if len({node_id for _, node_id in slots}) != len(slots):
        raise ValueError("每个参考图输入需要独立的 LoadImage 节点，以免不同图片互相覆盖。")
    outputs = [str(k) for k, v in graph.items() if isinstance(v, dict) and v.get("class_type") == "SaveImage"
               and v.get("inputs", {}).get("images") == [target, 0]]
    if len(outputs) != 1:
        raise ValueError("工作流需要一个直接连接 GPT 图片输出的 SaveImage 节点。")
    return target, slots, outputs[0]


def build_workflow(graph, file_names, prompt, aspect_ratio, resolution, seed, channel="economy"):
    target, slots, output = workflow_layout(graph)
    if channel not in ("economy", "stable"):
        raise ValueError("channel 应为 economy 或 stable。")
    max_images = 16 if channel == "stable" else 10
    if not 1 <= len(file_names) <= max_images:
        hint = "；超过 10 张请选择 channel=stable（稳定版按 token 计费）" if channel == "economy" else ""
        raise ValueError(f"RunningHub {channel} 支持 1–{max_images} 张参考图，当前输入 {len(file_names)} 张{hint}。")
    if not str(prompt).strip():
        raise ValueError("prompt 不能为空。")
    workflow = copy.deepcopy(graph)
    inputs = workflow[target]["inputs"]
    for field, _ in slots:
        del inputs[field]
    # The saved template may have fewer LoadImage nodes than the model supports.
    next_id = max((int(key) for key in workflow if str(key).isdigit()), default=0) + 1
    used_fields = {field for field, _ in slots}
    for index in range(1, max_images + 1):
        if len(slots) >= len(file_names):
            break
        field = f"image{index}"
        if field in used_fields:
            continue
        load_id = str(next_id)
        next_id += 1
        workflow[load_id] = {"class_type": "LoadImage", "inputs": {"image": ""}}
        slots.append((field, load_id))
    slots.sort(key=lambda slot: int(slot[0][5:]))
    for (field, load_id), name in zip(slots, file_names):
        workflow[load_id]["inputs"]["image"] = name
        inputs[field] = [load_id, 0]
    prompt_link = inputs.get("prompt")
    if (isinstance(prompt_link, list) and len(prompt_link) == 2
            and workflow.get(str(prompt_link[0]), {}).get("class_type") == "CR Prompt Text"):
        workflow[str(prompt_link[0])]["inputs"]["prompt"] = str(prompt)
    else:
        inputs["prompt"] = str(prompt)
    inputs.update(aspectRatio=aspect_ratio, resolution=resolution, seed=int(seed), skip_error=False)
    if channel == "stable":
        workflow[target]["class_type"] = STABLE_IMAGE_CLASS
        inputs.update(quality="high", background="auto", outputFormat="png")
        if aspect_ratio == "auto":
            del inputs["aspectRatio"]

    # Submit only the selected output and its dependencies, excluding unused example images.
    active = set()
    pending = [output]
    while pending:
        node_id = pending.pop()
        if node_id in active:
            continue
        active.add(node_id)
        for value in workflow[node_id].get("inputs", {}).values():
            if (isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)
                    and str(value[0]) in workflow):
                pending.append(str(value[0]))
    workflow = {k: v for k, v in workflow.items() if k in active}
    # RunningHub resets seeds unless explicitly overridden through nodeInfoList.
    overrides = [{"nodeId": target, "fieldName": "seed", "fieldValue": int(seed)}]
    return workflow, overrides, {"image_node": target, "output_node": output,
                                 "reference_nodes": [load_id for _, load_id in slots[:len(file_names)]]}


class RunningHubClient(ImageClient):
    def post(self, path, **kwargs):
        self.check_interrupt()
        try:
            response = self.session.request("POST", self.config.base_url + path,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=(30, 300), allow_redirects=False, **kwargs)
        except requests.RequestException as exc:
            detail = safe_response(str(exc), self.config.api_key)
            detail = re.sub(r"(https?://)[^/\s@]+@", r"\1[redacted]@", detail)[:1200]
            hint = ("未自动重发；提交结果未知，请先检查 RunningHub 任务记录。" if path == "/task/openapi/create"
                    else "本次请求未完成；未重新提交生图任务。")
            raise ImageAPIError(f"RunningHub {path}: {type(exc).__name__}: {detail}；{hint}") from None
        try:
            body = response.json()
        except ValueError:
            raise ImageAPIError(f"RunningHub {path}: HTTP {response.status_code}，未返回 JSON。") from None
        if not isinstance(body, dict):
            raise ImageAPIError(f"RunningHub {path}: 响应不是 JSON 对象。")
        if not 200 <= response.status_code < 300 or body.get("code", 0) not in (0, "0"):
            message = safe_response(body.get("message") or body.get("msg") or body.get("errorMessage") or body,
                                    self.config.api_key)
            raise ImageAPIError(f"RunningHub {path}: HTTP {response.status_code}, code={body.get('code')}: {str(message)[:1200]}")
        return body

    def fetch_workflow(self, workflow_id):
        body = self.post("/api/openapi/getJsonApiFormat",
                         json={"apiKey": self.config.api_key, "workflowId": str(workflow_id)})
        raw = (body.get("data") or {}).get("prompt")
        try:
            graph = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            raise ImageAPIError("RunningHub 工作流 JSON 无法解析。") from None
        workflow_layout(graph)
        return graph

    def upload(self, file):
        self.check_interrupt()
        name, raw, size = file
        digest = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        if self.upload_cache is not None and not self.force_upload:
            cached = self.upload_cache.get(digest)
            if cached:
                self.upload_stats["cache_hits"] += 1
                LOG.info("[QZW RunningHub] upload cache hit: %s (%dx%d)", name, *size)
                return cached
        started_at = time.time()
        try:
            body = self.post("/openapi/v2/media/upload/binary", files={"file": (name, raw, "image/png")})
        except ImageAPIError as exc:
            raise ImageAPIError(f"参考图 {name}（{size[0]}×{size[1]}）：{exc} 尚未提交生图任务。") from None
        data = body.get("data") or {}
        file_name = data.get("fileName")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ImageAPIError("RunningHub 上传响应缺少 fileName，无法填入 LoadImage。")
        self.upload_stats["uploaded"] += 1
        if self.upload_cache is not None:
            self.upload_cache.put(digest, data.get("download_url") or "", data.get("expires_in", 86400),
                                  started_at, file_name=file_name)
        LOG.info("[QZW RunningHub] uploaded: %s (%dx%d)", name, *size)
        return file_name

    def poll(self, task_id, interval, max_time):
        deadline, failures = time.monotonic() + max_time, 0
        while time.monotonic() < deadline:
            self.check_interrupt()
            try:
                body = self.post("/openapi/v2/query", json={"taskId": task_id})
            except ImageAPIError as exc:
                failures += 1
                if failures >= 5:
                    raise ImageAPIError(f"任务 {task_id} 查询失败；未重新提交任务。{exc}") from None
            else:
                failures = 0
                status = str(body.get("status", "")).upper()
                if status in ("FAILED", "FAILURE", "ERROR", "CANCEL", "CANCELLED", "CANCELED"):
                    detail = body.get("errorMessage") or body.get("failedReason") or body.get("errorCode")
                    raise ImageAPIError(f"RunningHub 任务 {task_id} {status}: {safe_response(detail, self.config.api_key)}")
                if status == "SUCCESS":
                    if not body.get("results"):
                        raise ImageAPIError(f"RunningHub 任务 {task_id} 已完成，但没有输出文件。")
                    return body
                if body.get("errorCode") not in (None, "", 0, "0"):
                    raise ImageAPIError(f"RunningHub 任务 {task_id}: {safe_response(body.get('errorMessage'), self.config.api_key)}")
                self.progress(50)
            for _ in range(max(1, int(interval))):
                self.check_interrupt()
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        raise ImageAPIError(f"RunningHub 任务 {task_id} 查询超时；任务可能仍在运行，请查询原任务，避免重复生成。")

    def generate(self, *, workflow_id, prompt, aspect_ratio="16:9", resolution="1k", seed=0,
                 images=None, poll_interval=5, max_poll_time=3600, channel="economy"):
        if self.config.protocol != "runninghub":
            raise ValueError("请将 QZW API Config 的 protocol 设置为 runninghub，并填写 RunningHub base_url 和 API Key。")
        if not str(workflow_id).strip().isdigit():
            raise ValueError("workflow_id 应填写 RunningHub 工作流链接末尾的数字 ID。")
        if not str(prompt).strip():
            raise ValueError("prompt 不能为空。")
        files = encode_images(images)
        if not files:
            raise ValueError("RunningHub GPT Image 2.5 图生图至少需要一张参考图。")
        self.upload_stats = {"cache_hits": 0, "uploaded": 0}
        graph = self.fetch_workflow(workflow_id)
        # Validate mapping and count before any uploads.
        build_workflow(graph, [f"reference_{i}.png" for i in range(len(files))], prompt, aspect_ratio, resolution, seed, channel)
        file_names = [self.upload(file) for file in files]
        workflow, overrides, mapping = build_workflow(graph, file_names, prompt, aspect_ratio, resolution, seed, channel)
        payload = {"apiKey": self.config.api_key, "workflowId": str(workflow_id),
                   "workflow": json.dumps(workflow, ensure_ascii=False), "nodeInfoList": overrides}
        body = self.post("/task/openapi/create", json=payload)
        data = body.get("data") or {}
        task_id = str(data.get("taskId") or "")
        if not task_id:
            raise ImageAPIError("RunningHub 提交响应没有 taskId；未重新提交，请先检查服务端任务记录。")
        LOG.info("[QZW RunningHub] task_id=%s workflow_id=%s", task_id, workflow_id)
        self.progress(20)
        if str(data.get("taskStatus", "")).upper() in ("FAILED", "FAILURE", "ERROR"):
            raise ImageAPIError(f"RunningHub 任务 {task_id} 创建失败: {safe_response(data.get('promptTips'), self.config.api_key)}")
        result = self.poll(task_id, poll_interval, max_poll_time)
        outputs = [{"url": item["url"]} for item in result["results"] if isinstance(item, dict) and item.get("url")
                   and str(item.get("outputType", "")).lower() in ("", "image", "png", "jpg", "jpeg", "webp")]
        if not outputs:
            raise ImageAPIError(f"RunningHub 任务 {task_id} 没有可下载的图片输出。")
        self.progress(90)
        try:
            tensor, urls, dimensions = self.decode({"data": outputs})
        except ImageAPIError as exc:
            raise ImageAPIError(f"RunningHub 任务 {task_id}: {exc}") from None
        report = {"protocol": "runninghub", "channel": channel, "image_class": workflow[mapping["image_node"]]["class_type"],
                  "workflow_id": str(workflow_id), "task_id": task_id,
                  "endpoint": self.config.base_url + "/task/openapi/create", "mapping": mapping,
                  "request": {"prompt": f"[omitted; {len(prompt)} characters]", "aspectRatio": aspect_ratio,
                              "resolution": resolution, "seed": int(seed), "reference_count": len(files)},
                  "reference_uploads": dict(self.upload_stats), "reference_sizes": [list(f[2]) for f in files],
                  "decoded_sizes": dimensions, "output_shape": list(tensor.shape),
                  "api_response": safe_response(result, self.config.api_key)}
        self.progress(100)
        return tensor, "\n".join(urls), json.dumps(report, ensure_ascii=False, indent=2)
