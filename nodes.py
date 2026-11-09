import hashlib
import os
from pathlib import Path

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .config import make_config
from .image_api import AITUDOU_MODELS, ImageClient, MODELS, SIZES, resolve_model, resolve_size
from .upload_cache import UploadCache


QZWConfig = io.Custom("QZW_API_CONFIG")


class QZWAPIConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QZW_API_Config", display_name="QZW API Config", category="QZW/API",
            inputs=[
                io.String.Input("base_url", default="https://api.aitudou.net"),
                io.Combo.Input("protocol", options=["auto", "aitudou", "openai", "workshop", "sub2api"], default="auto",
                               tooltip="auto 按域名识别 Aitudou/AI工坊，其余使用 OpenAI 同步协议。"),
                io.String.Input("api_key", default="", tooltip="直接输入会保存在工作流中。建议使用密钥文件或 QZW_API_KEY。"),
                io.String.Input("api_key_file", default="", tooltip="可选：本机单行密钥文件完整路径；密钥不会写入工作流。"),
            ], outputs=[QZWConfig.Output(display_name="api_config")])

    @classmethod
    def execute(cls, base_url, protocol="auto", api_key="", api_key_file=""):
        return io.NodeOutput(make_config(base_url, api_key, api_key_file, protocol))

    @classmethod
    def fingerprint_inputs(cls, base_url, protocol="auto", api_key="", api_key_file=""):
        if api_key_file and not api_key:
            stat = Path(api_key_file).expanduser().stat()
            return (stat.st_mtime_ns, stat.st_size)
        if not api_key:
            return hashlib.sha256(os.environ.get("QZW_API_KEY", "").encode()).hexdigest()
        return None


class QZWGPTImage25(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        model_options = [io.DynamicCombo.Option(name, []) for name in (*MODELS, *AITUDOU_MODELS)]
        model_options.append(io.DynamicCombo.Option("custom", [
            io.String.Input("custom_model", default="aitudou-image-g-v2.5-lowprice",
                            tooltip="填写中转站文档中的完整模型 ID；可转为输入连接文本节点。")]))
        size_options = [io.DynamicCombo.Option(name, []) for name in SIZES]
        size_options.append(io.DynamicCombo.Option("custom", [
            io.Int.Input("custom_width", default=3840, min=16, max=16384, step=16),
            io.Int.Input("custom_height", default=2160, min=16, max=16384, step=16)]))
        return io.Schema(
            node_id="QZW_GPT_Image_2_5", display_name="QZW GPT Image 2.5", category="QZW/Image",
            description="GPT Image 2.5 文生图/多参考图编辑。支持 Aitudou、OpenAI 兼容、AI工坊及 Sub2API。",
            inputs=[
                QZWConfig.Input("api_config"),
                io.Autogrow.Input("images", template=io.Autogrow.TemplateNames(
                    io.Image.Input("image"), names=[f"image{i}" for i in range(1, 17)], min=0)),
                io.Mask.Input("mask", optional=True, tooltip="仅 OpenAI 兼容编辑：白色区域重绘，需要一张参考图。"),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
                io.DynamicCombo.Input("model", options=model_options),
                io.Combo.Input("quality", options=["auto", "low", "medium", "high", "xhigh", "max"], default="auto",
                               tooltip="Aitudou lowprice 不使用此参数。"),
                io.DynamicCombo.Input("size", options=size_options),
                io.Combo.Input("resolution", options=["1k", "2k", "4k"], default="1k",
                               tooltip="Aitudou 输出档位；官方版精确像素尺寸优先。lowprice 不保证精确像素。"),
                io.Int.Input("n", default=1, min=1, max=10),
                io.Combo.Input("background", options=["auto", "opaque", "transparent"], default="auto", advanced=True),
                io.Combo.Input("moderation", options=["auto", "low"], default="auto", advanced=True),
                io.Boolean.Input("nsfw_check", default=False, advanced=True, tooltip="仅 Aitudou lowprice 使用。"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                             tooltip="仅控制 ComfyUI 缓存，不发送给 API。改变 seed 才会重新请求。"),
                io.Boolean.Input("async_mode", default=False, advanced=True,
                                 tooltip="AI工坊/Sub2API 的异步开关；Aitudou 始终为异步，OpenAI 兼容使用同步。"),
                io.Int.Input("poll_interval", default=5, min=2, max=60, advanced=True),
                io.Int.Input("max_poll_time", default=3600, min=60, max=7200, advanced=True),
                io.Boolean.Input("force_upload", default=False, optional=True, advanced=True,
                                 tooltip="仅 Aitudou：跳过参考图上传缓存，重新上传并刷新缓存。通常保持关闭。"),
            ], outputs=[io.Image.Output(display_name="image"), io.String.Output(display_name="image_urls"),
                        io.String.Output(display_name="response")], is_output_node=True)

    @classmethod
    def execute(cls, api_config, prompt, model, quality, size, resolution, n, background="auto",
                moderation="auto", nsfw_check=False, seed=0, async_mode=False, poll_interval=5,
                max_poll_time=3600, images=None, mask=None, force_upload=False):
        pbar = comfy.utils.ProgressBar(100)
        upload_cache = None
        if api_config.protocol == "aitudou":
            upload_cache = UploadCache(Path(folder_paths.get_system_user_directory("qzw")) / "upload_cache.sqlite3",
                                       api_config.base_url, api_config.api_key)
        client = ImageClient(api_config,
            upload_cache=upload_cache, force_upload=force_upload,
            check_interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
            progress=lambda value: pbar.update_absolute(value, 100))
        try:
            result = client.generate(model=resolve_model(model), prompt=prompt, size=resolve_size(size),
                resolution=resolution, quality=quality, n=n, background=background, moderation=moderation,
                nsfw_check=nsfw_check, images=images, mask=mask, async_mode=async_mode,
                poll_interval=poll_interval, max_poll_time=max_poll_time)
            return io.NodeOutput(*result)
        finally:
            client.close()
