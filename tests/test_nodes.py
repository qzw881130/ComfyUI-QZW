"""Integration with the installed ComfyUI schema/validation; no generation."""
import asyncio
import importlib.util
import json
import os
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_ROOT", str(ROOT.parent.parent)))


@unittest.skipUnless((COMFY / "execution.py").exists(), "Set COMFYUI_ROOT to test the ComfyUI integration")
class NodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(COMFY))
        spec = importlib.util.spec_from_file_location("qzw_integration", ROOT / "__init__.py",
                                                     submodule_search_locations=[str(ROOT)])
        cls.package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.package
        spec.loader.exec_module(cls.package)

    def test_model_custom_and_autogrow_schema(self):
        node = self.package.QZWGPTImage25
        schema = node.GET_SCHEMA()
        schema.validate()
        required = node.INPUT_TYPES()["required"]
        options = required["model"][1]["options"]
        self.assertEqual(len(options), 10)
        self.assertEqual([option["key"] for option in options[-4:-1]], [
            "aitudou-image-g-v2.5-flare", "aitudou-image-g-v2.5-sunburst", "aitudou-image-g-v2.5-lowprice"])
        self.assertEqual(options[-1]["key"], "custom")
        self.assertIn("custom_model", options[-1]["inputs"]["required"])
        self.assertEqual(required["images"][1]["template"]["min"], 0)

    def test_environment_credential_fingerprint_stays_stable(self):
        cls = self.package.QZWAPIConfig
        with patch.dict(os.environ, {"QZW_API_KEY": "fake-secret"}):
            first = cls.fingerprint_inputs("https://api.aitudou.net")
            self.assertEqual(first, cls.fingerprint_inputs("https://api.aitudou.net"))
        with patch.dict(os.environ, {"QZW_API_KEY": "another-fake-secret"}):
            self.assertNotEqual(first, cls.fingerprint_inputs("https://api.aitudou.net"))

    def test_text_workflow_passes_real_comfy_validation(self):
        import execution
        import nodes
        from comfy_extras.nodes_preview_any import PreviewAny
        from comfy_extras.nodes_primitive import StringMultiline
        graph = json.loads((ROOT / "examples" / "QZW_GPT_Image25_text_to_image.api.json").read_text(encoding="utf-8"))
        mapping = {"QZW_API_Config": self.package.QZWAPIConfig,
                   "QZW_GPT_Image_2_5": self.package.QZWGPTImage25,
                   "PreviewAny": PreviewAny, "PrimitiveStringMultiline": StringMultiline}
        # Windows creates a local socket pair when constructing the event loop.
        loop = asyncio.new_event_loop()
        try:
            with patch.dict(nodes.NODE_CLASS_MAPPINGS, mapping), patch.object(socket.socket, "connect", side_effect=AssertionError("No network")):
                valid, error, _, errors = loop.run_until_complete(execution.validate_prompt("qzw-test", graph, None))
        finally:
            loop.close()
        self.assertTrue(valid, (error, errors))

    def test_example_ui_links_are_consistent(self):
        for file in (ROOT / "examples").glob("*.json"):
            if file.name.endswith(".api.json"):
                continue
            workflow = json.loads(file.read_text(encoding="utf-8"))
            nodes = {n["id"]: n for n in workflow["nodes"]}
            for id, source, source_slot, target, target_slot, type in workflow["links"]:
                self.assertIn(id, nodes[source]["outputs"][source_slot]["links"])
                self.assertEqual(nodes[target]["inputs"][target_slot]["link"], id)
            config = next(n for n in nodes.values() if n["type"] == "QZW_API_Config")
            self.assertEqual(config["widgets_values_named"]["api_key"], "")
            self.assertEqual(config["widgets_values_named"]["api_key_file"], "")


if __name__ == "__main__":
    unittest.main()
