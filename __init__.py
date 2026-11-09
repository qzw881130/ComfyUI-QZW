from comfy_api.latest import ComfyExtension

from .nodes import QZWAPIConfig, QZWGPTImage25
from .runninghub_nodes import QZWRunningHubGPTImage25


class QZWExtension(ComfyExtension):
    async def get_node_list(self):
        return [QZWAPIConfig, QZWGPTImage25, QZWRunningHubGPTImage25]


async def comfy_entrypoint():
    return QZWExtension()
