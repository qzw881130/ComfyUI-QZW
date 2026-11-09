import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class APIConfig:
    base_url: str
    api_key: str = field(repr=False)
    protocol: str = "auto"


def make_config(base_url, api_key="", api_key_file="", protocol="auto"):
    base_url = str(base_url).strip().rstrip("/")
    parsed = urlsplit(base_url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("API base URL 必须是有效的 http(s) 服务地址，不能包含密钥或查询参数。")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    if protocol not in ("auto", "aitudou", "openai", "workshop", "sub2api", "runninghub"):
        raise ValueError("不支持的 API 协议。")
    key = str(api_key).strip()
    if not key and str(api_key_file).strip():
        key = Path(api_key_file.strip()).expanduser().read_text(encoding="utf-8-sig").strip()
    if not key:
        key = os.environ.get("QZW_API_KEY", "").strip()
    if not key or "\n" in key or "\r" in key:
        raise ValueError("请设置 api_key、单行密钥文件 api_key_file 或 QZW_API_KEY 环境变量。")
    if protocol == "auto":
        host = urlsplit(base_url).hostname
        protocol = "aitudou" if host == "api.aitudou.net" else (
            "runninghub" if host in ("www.runninghub.cn", "www.runninghub.ai", "runninghub.cn", "runninghub.ai") else (
                "workshop" if host in ("ai.t8star.org", "ai.t8star.cn") else "openai"))
    return APIConfig(base_url, key, protocol)
