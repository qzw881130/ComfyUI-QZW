"""Image transport and conversion. No requests are made at import time."""

import base64
import copy
import io
import json
import logging
import math
import time
from urllib.parse import quote, urlsplit

import numpy as np
import requests
import torch
from PIL import Image


MODELS = (
    "gpt-image-2.5-flare", "gpt-image-2.5-flare-2k", "gpt-image-2.5-flare-4k",
    "gpt-image-2.5-sunburst", "gpt-image-2.5-sunburst-2k", "gpt-image-2.5-sunburst-4k",
)
RATIOS = ("1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5",
          "2:1", "1:2", "21:9", "9:21", "3:1", "1:3")
SIZES = ("16:9", "auto", "1024x1024", "1536x1024", "1024x1536", "2048x2048",
         "2048x1152", "1152x2048", "3840x2160", "2160x3840") + tuple(r for r in RATIOS if r != "16:9")
LOWPRICE = "aitudou-image-g-v2.5-lowprice"
LOG = logging.getLogger("ComfyUI-QZW")


class ImageAPIError(RuntimeError):
    pass


def resolve_model(model):
    selected = model["model"] if isinstance(model, dict) else model
    name = model.get("custom_model", "") if selected == "custom" and isinstance(model, dict) else selected
    name = str(name or "").strip()
    if not name or name == "custom":
        raise ValueError("选择 custom 后，请填写 custom_model。")
    return name


def resolve_size(size):
    selected = size["size"] if isinstance(size, dict) else size
    if selected == "custom":
        width, height = int(size["custom_width"]), int(size["custom_height"])
        if width <= 0 or height <= 0:
            raise ValueError("自定义宽高必须大于零。")
        return f"{width}x{height}"
    return str(selected)


def build_payload(protocol, model, prompt, size, resolution, quality, n, background, moderation, nsfw_check):
    if not str(prompt).strip():
        raise ValueError("prompt 不能为空。")
    effective_model, effective_resolution = model, resolution
    notes = []
    if protocol == "aitudou" and model in MODELS:
        suffix = next((s for s in ("-2k", "-4k") if model.endswith(s)), "")
        name = model[:-len(suffix)] if suffix else model
        effective_model = name.replace("gpt-image-2.5-", "aitudou-image-g-v2.5-", 1)
        if suffix:
            effective_resolution = suffix[1:]
        notes.append(f"Aitudou 预设映射：{model} -> {effective_model}")
    payload = {"model": effective_model, "prompt": str(prompt).strip(), "n": int(n), "size": size}
    if protocol == "aitudou":
        payload["resolution"] = effective_resolution
    if protocol == "aitudou" and effective_model == LOWPRICE:
        if int(n) != 1:
            raise ValueError("Aitudou lowprice 每次仅支持 n=1。")
        if "x" in size:
            width, height = map(int, size.split("x"))
            divisor = math.gcd(width, height)
            ratio = f"{width // divisor}:{height // divisor}"
            if ratio not in RATIOS:
                raise ValueError("lowprice 不支持该像素尺寸对应的比例，请选择其支持的 size 比例。")
            payload["size"] = ratio
            notes.append(f"lowprice 将 {size} 用作比例 {ratio}；输出档位为 {effective_resolution}，不保证精确像素。")
        payload["nsfw_check"] = bool(nsfw_check)
        notes.append("lowprice 不提供 quality/background/moderation；这些字段未发送。")
    else:
        payload.update(quality=quality, moderation=moderation)
        if background != "auto":
            payload["background"] = background
    return payload, notes


def encode_images(images):
    files = []
    images = images or {}
    for name in sorted(images, key=lambda key: int(key.rsplit("image", 1)[-1])):
        value = images[name]
        if value is None:
            continue
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4 or value.shape[-1] not in (1, 3, 4):
            raise ValueError("参考图必须为 [B,H,W,C] IMAGE tensor。")
        for frame in value:
            array = np.clip(frame.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if array.shape[-1] == 1:
                array = array[:, :, 0]
            image = Image.fromarray(array)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            files.append((f"reference_{len(files) + 1}.png", buffer.getvalue(), image.size))
    return files


def encode_mask(mask, files):
    if mask is None:
        return None
    if len(files) != 1:
        raise ValueError("mask 需要恰好一张参考图。")
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or (mask.shape[1], mask.shape[0]) != files[0][2]:
        raise ValueError("mask 尺寸必须与参考图一致。")
    alpha = np.clip((1.0 - mask.detach().cpu().numpy()) * 255, 0, 255).astype(np.uint8)
    array = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    array[:, :, 3] = alpha
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def image_items(result):
    """Read output containers only; never traverse echoed request images."""
    if not isinstance(result, dict):
        return []
    data = result.get("data")
    if isinstance(data, list):
        items = [v for v in data if isinstance(v, dict) and (v.get("url") or v.get("b64_json"))]
        if items:
            return items
    content = result.get("content")
    if isinstance(content, dict):
        urls = content.get("image_urls") or ([content["image_url"]] if content.get("image_url") else [])
        if urls:
            return [{"url": u} for u in dict.fromkeys(urls)]
    for key in ("result", "data"):
        found = image_items(result.get(key))
        if found:
            return found
    if result.get("result_url"):
        return [{"url": result["result_url"]}]
    return []


def task_info(result):
    inner = result.get("data")
    record = inner if isinstance(inner, dict) else result
    task_id = record.get("task_id") or record.get("id") or result.get("task_id") or result.get("id")
    if not task_id and isinstance(inner, str):
        task_id = inner
    status = str(record.get("status", "")).upper()
    error = record.get("fail_reason") or record.get("error") or record.get("message") or result.get("error")
    return str(task_id or ""), status, error


def safe_response(value, key):
    if isinstance(value, dict):
        return {k: "[base64 image omitted]" if k == "b64_json" else safe_response(v, key)
                for k, v in value.items() if k.lower() not in ("api_key", "authorization")}
    if isinstance(value, list):
        return [safe_response(v, key) for v in value]
    if isinstance(value, str) and key:
        return value.replace(key, "[redacted]")
    return value


class ImageClient:
    def __init__(self, config, *, session=None, media_session=None, check_interrupt=None, progress=None):
        self.config = config
        self.session = session or requests.Session()
        self.media_session = media_session or requests.Session()
        self.check_interrupt = check_interrupt or (lambda: None)
        self.progress = progress or (lambda value: None)

    def close(self):
        self.session.close()
        self.media_session.close()

    def request(self, method, path, **kwargs):
        self.check_interrupt()
        try:
            response = self.session.request(method, self.config.base_url + path,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=(30, 900 if method == "POST" else 90), allow_redirects=False, **kwargs)
        except requests.RequestException as exc:
            # A timed-out POST may have created a paid task. Never auto-resubmit.
            raise ImageAPIError(f"{method} {path}: {type(exc).__name__}；未自动重发，请先检查服务端任务记录。") from None
        try:
            body = response.json()
        except ValueError:
            raise ImageAPIError(f"{method} {path}: HTTP {response.status_code}，服务端未返回 JSON。") from None
        if not 200 <= response.status_code < 300:
            message = json.dumps(safe_response(body, self.config.api_key), ensure_ascii=False)[:1200]
            raise ImageAPIError(f"{method} {path}: HTTP {response.status_code}: {message}")
        if not isinstance(body, dict):
            raise ImageAPIError("服务端响应必须为 JSON 对象。")
        return body

    def upload(self, file):
        name, raw, _ = file
        body = self.request("POST", "/v1/files/upload", files={"file": (name, raw, "image/png")})
        url = body.get("url")
        if not isinstance(url, str) or urlsplit(url).scheme not in ("http", "https"):
            raise ImageAPIError("上传接口没有返回有效的图片 URL。")
        return url

    def poll(self, task_id, interval, max_time):
        protocol = self.config.protocol
        path = ("/v1/image/generations/" if protocol == "aitudou" else "/v1/images/tasks/") + quote(task_id, safe="")
        deadline, failures = time.monotonic() + max_time, 0
        while time.monotonic() < deadline:
            self.check_interrupt()
            for _ in range(max(1, int(interval))):
                self.check_interrupt()
                time.sleep(min(1, max(0, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                break
            try:
                result = self.request("GET", path)
            except ImageAPIError:
                failures += 1
                if failures >= 5:
                    raise
                continue
            failures = 0
            _, status, error = task_info(result)
            if status in ("FAILURE", "FAILED", "ERROR", "CANCELLED", "CANCELED"):
                raise ImageAPIError(f"任务 {task_id} {status}: {safe_response(error, self.config.api_key)}")
            if image_items(result) and status in ("", "SUCCESS", "SUCCEEDED", "COMPLETED"):
                return result
            if status in ("SUCCESS", "SUCCEEDED", "COMPLETED"):
                raise ImageAPIError(f"任务 {task_id} 已完成，但响应没有图片地址或 Base64 数据。")
            self.progress(50)
        raise ImageAPIError(f"任务 {task_id} 查询超时；任务可能仍在运行，请查询原任务，不要直接重复生成。")

    def decode(self, result):
        tensors, urls, sizes = [], [], []
        for item in image_items(result):
            self.check_interrupt()
            if item.get("b64_json"):
                encoded = item["b64_json"]
                raw = base64.b64decode(encoded.split(",", 1)[-1] if encoded.startswith("data:") else encoded)
            else:
                url = item["url"]
                if urlsplit(url).scheme not in ("http", "https"):
                    raise ImageAPIError("返回的图片地址不是 HTTP(S) URL。")
                # A separate session prevents the API credential from reaching image/CDN hosts.
                for attempt in range(3):
                    try:
                        response = self.media_session.get(url, timeout=(30, 300))
                        response.raise_for_status()
                        raw = response.content
                        break
                    except requests.RequestException:
                        if attempt == 2:
                            raise ImageAPIError("结果图片下载失败；可使用原任务 ID 重新查询。") from None
                        self.check_interrupt()
                        time.sleep(attempt + 1)
                urls.append(url)
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                sizes.append(list(image.size))
                tensors.append(torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)[None])
        if not tensors:
            raise ImageAPIError("响应没有可解码的图片。")
        if len({tuple(t.shape[1:]) for t in tensors}) != 1:
            raise ImageAPIError("返回图片尺寸不一致，无法组成 IMAGE 批次；请使用 n=1。")
        return torch.cat(tensors), urls, sizes

    def generate(self, *, model, prompt, size, resolution="1k", quality="auto", n=1,
                 background="auto", moderation="auto", nsfw_check=False, images=None,
                 mask=None, async_mode=False, poll_interval=5, max_poll_time=3600):
        protocol = self.config.protocol
        payload, notes = build_payload(protocol, model, prompt, size, resolution, quality, n,
                                       background, moderation, nsfw_check)
        if protocol == "aitudou" and mask is not None:
            raise ValueError("Aitudou GPT Image 2.5 文档未提供 mask 接口；请断开 mask，或使用 OpenAI 兼容协议。")
        if protocol == "openai" and async_mode:
            raise ValueError("OpenAI 兼容协议请关闭 async_mode；Sub2API 异步接口请选择 sub2api 协议。")
        files = encode_images(images)
        mask_png = encode_mask(mask, files)
        if protocol == "aitudou":
            if files:
                payload["images"] = [self.upload(f) for f in files]
            path = "/v1/image/generations"
            encoding = "application/json"
            result = self.request("POST", path, json=payload)
        else:
            if protocol == "workshop":
                payload["response_format"] = "url"
            suffix = ("?async=true" if protocol == "workshop" else "/async") if async_mode else ""
            path = ("/v1/images/edits" if files else "/v1/images/generations") + suffix
            encoding = "multipart/form-data" if files else "application/json"
            if files:
                parts = [("image", (name, raw, "image/png")) for name, raw, _ in files]
                if mask_png is not None:
                    parts.append(("mask", ("mask.png", mask_png, "image/png")))
                result = self.request("POST", path, data={k: str(v) for k, v in payload.items()}, files=parts)
            else:
                result = self.request("POST", path, json=payload)
        self.progress(20)
        task_id, status, error = task_info(result)
        if status in ("FAILURE", "FAILED", "ERROR", "CANCELLED", "CANCELED"):
            raise ImageAPIError(f"任务失败: {safe_response(error, self.config.api_key)}")
        if not image_items(result):
            if not task_id:
                raise ImageAPIError("提交响应没有图片或任务 ID。")
            LOG.info("[QZW] task_id=%s protocol=%s", task_id, protocol)
            result = self.poll(task_id, poll_interval, max_poll_time)
        self.progress(90)
        tensor, urls, dimensions = self.decode(result)
        if "x" in size:
            requested = list(map(int, size.split("x")))
            if any(s != requested for s in dimensions):
                notes.append(f"请求 {size}，API 图片实际为 {dimensions}；节点未缩放。")
                LOG.warning("[QZW] requested=%s decoded=%s (no resize)", size, dimensions)
        report_payload = copy.deepcopy(payload)
        report_payload["prompt"] = f"[omitted; {len(payload['prompt'])} characters]"
        if "images" in report_payload:
            report_payload["images"] = [f"[uploaded reference {i + 1}]" for i in range(len(files))]
        report = {"protocol": protocol, "endpoint": self.config.base_url + path,
                  "task_id": task_id, "requested_model": model, "requested_size": size,
                  "request_encoding": encoding, "request": report_payload,
                  "reference_sizes": [list(f[2]) for f in files], "decoded_sizes": dimensions,
                  "output_shape": list(tensor.shape), "notes": notes,
                  "api_response": safe_response(result, self.config.api_key)}
        self.progress(100)
        return tensor, "\n".join(urls), json.dumps(report, ensure_ascii=False, indent=2)
