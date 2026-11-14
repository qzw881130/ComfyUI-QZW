from pathlib import Path

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .image_api import RATIOS
from .nodes import QZWConfig
from .runninghub_api import DEFAULT_WORKFLOW_ID, RunningHubClient
from .upload_cache import UploadCache


class QZWRunningHubGPTImage25(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QZW_RunningHub_GPT_Image_2_5", display_name="QZW RunningHub GPT Image 2.5", category="QZW/Image",
            description="通过 RunningHub ComfyUI 工作流运行 GPT Image 2.5 Flare 图生图。自动替换参考图、提示词和尺寸参数。",
            inputs=[
                QZWConfig.Input("api_config"),
                io.Autogrow.Input("images", template=io.Autogrow.TemplateNames(
                    io.Image.Input("image"), names=[f"image{i}" for i in range(1, 16)], min=0),
                    tooltip="新建任务至少接入一张参考图，恢复原任务可不接图。最多 15 个入口，超过 10 张请选择 channel=stable。"),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
                io.Combo.Input("aspect_ratio", options=["auto", *RATIOS], default="16:9"),
                io.Combo.Input("resolution", options=["1k", "2k", "4k"], default="1k"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                             tooltip="通过 nodeInfoList 显式传给云端节点。"),
                io.String.Input("workflow_id", default=DEFAULT_WORKFLOW_ID, advanced=True,
                                tooltip="云端模板 ID；读取 API JSON 后，只提交实际参考图及输出依赖，不修改已保存的云端模板。"),
                io.Int.Input("poll_interval", default=5, min=2, max=60, advanced=True),
                io.Int.Input("max_poll_time", default=3600, min=60, max=7200, advanced=True),
                io.Boolean.Input("force_upload", default=False, advanced=True,
                                 tooltip="跳过上传缓存并刷新。RunningHub 文件与 Aitudou 缓存隔离，最多复用 24 小时，提前 10 分钟失效。"),
                io.Combo.Input("channel", options=["economy", "stable"], default="economy", optional=True,
                               tooltip="economy：经济版，最多 10 张；stable：稳定版，支持 15 个入口，按 token 计费，quality=high。"),
                io.String.Input("resume_task_id", default="", optional=True, advanced=True,
                                tooltip="填写原任务 ID 时，只查询和下载该任务，不上传参考图或创建新任务。生成新图前请清空。"),
            ], outputs=[io.Image.Output(display_name="image"), io.String.Output(display_name="image_urls"),
                        io.String.Output(display_name="response")], is_output_node=True)

    @classmethod
    def execute(cls, api_config, prompt, images, aspect_ratio="16:9", resolution="1k", seed=0,
                workflow_id=DEFAULT_WORKFLOW_ID, poll_interval=5, max_poll_time=3600, force_upload=False, channel="economy",
                resume_task_id=""):
        progress = comfy.utils.ProgressBar(100)
        cache = UploadCache(Path(folder_paths.get_system_user_directory("qzw")) / "runninghub_upload_cache.sqlite3",
                            api_config.base_url, api_config.api_key, file_names=True)
        client = RunningHubClient(api_config, upload_cache=cache, force_upload=force_upload,
            check_interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
            progress=lambda value: progress.update_absolute(value, 100))
        try:
            return io.NodeOutput(*client.generate(workflow_id=workflow_id, prompt=prompt, images=images,
                aspect_ratio=aspect_ratio, resolution=resolution, seed=seed,
                poll_interval=poll_interval, max_poll_time=max_poll_time, channel=channel, resume_task_id=resume_task_id))
        finally:
            client.close()
