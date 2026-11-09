import base64
import importlib.util
import io
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for name in ("config", "image_api"):
    spec = importlib.util.spec_from_file_location("qzw_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
config = sys.modules["qzw_config"]
api = sys.modules["qzw_image_api"]


def png(size=(32, 16), mode="RGB"):
    data = io.BytesIO()
    Image.new(mode, size).save(data, format="PNG")
    return data.getvalue()


def response(body, status=200):
    return Mock(status_code=status, json=Mock(return_value=body))


def result(size=(32, 16)):
    return {"data": [{"b64_json": base64.b64encode(png(size)).decode(), "size": f"{size[0]}x{size[1]}"}]}


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Tests must not use network"))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.session, self.media = Mock(), Mock()
        self.client = api.ImageClient(config.APIConfig("https://api.aitudou.net", "fake-secret", "aitudou"),
                                      session=self.session, media_session=self.media)

    def args(self, **kw):
        return dict(model=api.LOWPRICE, prompt="red ceramic teapot", size="16:9", **kw)

    def payload(self, model=api.LOWPRICE, size="16:9", **kw):
        args = dict(protocol="aitudou", model=model, prompt="test image", size=size, resolution="4k",
                    quality="high", n=1, background="opaque", moderation="auto", nsfw_check=False)
        args.update(kw)
        return api.build_payload(**args)

    def test_config_file_and_no_secret_repr(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / "key.txt"
            f.write_text("\ufefffake-secret\n", encoding="utf-8")
            c = config.make_config("https://api.aitudou.net/v1/", api_key_file=str(f))
        self.assertEqual((c.base_url, c.protocol, c.api_key), ("https://api.aitudou.net", "aitudou", "fake-secret"))
        self.assertNotIn("fake-secret", repr(c))

    def test_config_rejects_credentials_and_queries(self):
        for url in ("https://user:key@host", "https://host?key=secret", "file:///key"):
            with self.assertRaises(ValueError): config.make_config(url, "fake")

    def test_custom_model_is_unrestricted(self):
        self.assertEqual(api.resolve_model({"model": "custom", "custom_model": " custom/provider-model "}), "custom/provider-model")
        with self.assertRaises(ValueError): api.resolve_model({"model": "custom", "custom_model": ""})

    def test_six_presets_and_custom_model_mapping(self):
        self.assertEqual(len(api.MODELS), 6)
        for model in api.MODELS:
            p, _ = self.payload(model=model)
            self.assertTrue(p["model"].startswith("aitudou-image-g-v2.5-"))
            self.assertFalse(p["model"].endswith(("-2k", "-4k")))
        p, _ = self.payload(model="custom-model")
        self.assertEqual(p["model"], "custom-model")

    def test_lowprice_fields_and_pixel_to_ratio(self):
        p, notes = self.payload(size="3840x2160")
        self.assertEqual(p, {"model": api.LOWPRICE, "prompt": "test image", "size": "16:9", "resolution": "4k", "n": 1, "nsfw_check": False})
        self.assertTrue(notes)
        with self.assertRaises(ValueError): self.payload(n=2)

    def test_official_pixel_size_is_unchanged(self):
        p, _ = self.payload(model="aitudou-image-g-v2.5-flare", size="3840x2160")
        self.assertEqual(p["size"], "3840x2160")
        self.assertEqual(p["quality"], "high")
        self.assertEqual(api.resolve_size({"size": "custom", "custom_width": 3840, "custom_height": 2160}), "3840x2160")

    def test_upload_order_and_reference_dimensions(self):
        a, b = torch.zeros((1, 16, 32, 3)), torch.zeros((1, 24, 48, 3))
        files = api.encode_images({"image10": b, "image2": a})
        self.assertEqual([f[2] for f in files], [(32, 16), (48, 24)])
        self.assertEqual(Image.open(io.BytesIO(files[1][1])).size, (48, 24))

    def test_aitudou_edit_upload_submit_poll_and_download(self):
        final = {"data": {"status": "SUCCESS", "result_url": "https://cdn.test/one.png",
                 "data": {"content": {"image_urls": ["https://cdn.test/one.png"]}}}}
        self.session.request.side_effect = [response({"url": "https://cdn.test/ref.png"}),
            response({"task_id": "task-a", "status": "queued"}),
            response({"data": {"status": "IN_PROGRESS"}}), response(final)]
        self.media.get.return_value = Mock(content=png((1672, 941)))
        with patch.object(api.time, "sleep"):
            image, urls, report = self.client.generate(**self.args(images={"image1": torch.zeros((1, 16, 32, 3))}, resolution="4k"))
        calls = self.session.request.call_args_list
        self.assertEqual([c.args[:2] for c in calls], [
            ("POST", "https://api.aitudou.net/v1/files/upload"),
            ("POST", "https://api.aitudou.net/v1/image/generations"),
            ("GET", "https://api.aitudou.net/v1/image/generations/task-a"),
            ("GET", "https://api.aitudou.net/v1/image/generations/task-a")])
        self.assertEqual(calls[1].kwargs["json"]["images"], ["https://cdn.test/ref.png"])
        self.assertEqual(tuple(image.shape), (1, 941, 1672, 3))
        self.assertNotIn("headers", self.media.get.call_args.kwargs)
        self.assertEqual(json.loads(report)["decoded_sizes"], [[1672, 941]])
        self.assertNotIn("fake-secret", report)

    def test_base64_4k_preserved(self):
        tensor, _, sizes = self.client.decode(result((3840, 2160)))
        self.assertEqual(tuple(tensor.shape), (1, 2160, 3840, 3))
        self.assertEqual(sizes, [[3840, 2160]])

    def test_mismatched_batch_fails_without_resize(self):
        value = {"data": result((32, 16))["data"] + result((16, 32))["data"]}
        with self.assertRaisesRegex(api.ImageAPIError, "尺寸不一致"): self.client.decode(value)

    def test_output_lists_and_no_echoed_input_images(self):
        body = {"data": {"result_url": "https://cdn/one", "data": {"content": {"image_urls": ["https://cdn/one", "https://cdn/two"]}}}}
        self.assertEqual(len(api.image_items(body)), 2)
        self.assertEqual(api.image_items({"images": ["https://input/ref"]}), [])

    def test_openai_sync_generation_and_edit(self):
        self.client.config = config.APIConfig("https://custom.test", "fake-secret", "openai")
        self.session.request.return_value = response(result())
        self.client.generate(model="custom-model", prompt="test", size="3840x2160")
        self.assertEqual(self.session.request.call_args.args[1], "https://custom.test/v1/images/generations")
        self.assertEqual(self.session.request.call_args.kwargs["json"]["size"], "3840x2160")
        self.client.generate(model="custom-model", prompt="test", size="3840x2160", images={"image1": torch.zeros((1, 16, 32, 3))})
        call = self.session.request.call_args
        self.assertTrue(call.args[1].endswith("/v1/images/edits"))
        self.assertEqual(call.kwargs["data"]["size"], "3840x2160")

    def test_workshop_and_sub2api_async_paths(self):
        for protocol, suffix in [("workshop", "?async=true"), ("sub2api", "/async")]:
            self.client.config = config.APIConfig("https://custom.test", "fake-secret", protocol)
            self.session.request.return_value = response(result())
            self.client.generate(model=api.MODELS[0], prompt="test", size="1024x1024", async_mode=True)
            self.assertTrue(self.session.request.call_args.args[1].endswith("/v1/images/generations" + suffix))

    def test_no_paid_post_retry_on_timeout(self):
        self.session.request.side_effect = requests.Timeout("contains fake-secret")
        with self.assertRaises(api.ImageAPIError) as caught: self.client.generate(**self.args())
        self.assertEqual(self.session.request.call_count, 1)
        self.assertNotIn("fake-secret", str(caught.exception))
        self.assertIn("提交结果未知", str(caught.exception))

    def test_lowprice_prompt_limit_before_upload(self):
        for length in (4, 5001, 10443):
            args = self.args(images={"image1": torch.zeros((1, 16, 32, 3))})
            args["prompt"] = "字" * length
            with self.assertRaisesRegex(ValueError, f"当前为 {length} 字符"):
                self.client.generate(**args)
        self.session.request.assert_not_called()
        for length in (5, 5000):
            payload, _ = self.payload(prompt="字" * length)
            self.assertEqual(len(payload["prompt"]), length)
        # Other protocols and models retain their own prompt limits.
        self.payload(model="custom-model", prompt="字" * 10443)

    def test_upload_ssl_error_details_without_generation_or_secrets(self):
        self.session.request.side_effect = requests.exceptions.SSLError(
            "UNEXPECTED_EOF_WHILE_READING fake-secret https://user:proxy-password@proxy.test:7890")
        with self.assertRaises(api.ImageAPIError) as caught:
            self.client.generate(**self.args(images={"image1": torch.zeros((1, 16, 32, 3))}))
        message = str(caught.exception)
        self.assertIn("UNEXPECTED_EOF_WHILE_READING", message)
        self.assertIn("reference_1.png", message)
        self.assertIn("32×16", message)
        self.assertIn("尚未提交生图任务", message)
        self.assertNotIn("fake-secret", message)
        self.assertNotIn("proxy-password", message)
        self.assertNotIn("检查服务端任务记录", message)
        self.assertEqual(self.session.request.call_count, 1)
        self.assertTrue(self.session.request.call_args.args[1].endswith("/v1/files/upload"))

    def test_query_network_error_is_distinct_from_submission(self):
        self.session.request.side_effect = requests.Timeout("read timed out")
        with self.assertRaisesRegex(api.ImageAPIError, "查询请求失败"):
            self.client.request("GET", "/v1/image/generations/task-a")
        self.assertEqual(self.session.request.call_count, 1)

    def test_rejection_redacts_key(self):
        self.session.request.return_value = response({"error": {"message": "bad fake-secret"}}, 503)
        with self.assertRaises(api.ImageAPIError) as caught: self.client.generate(**self.args())
        self.assertNotIn("fake-secret", str(caught.exception))
        self.assertEqual(self.session.request.call_count, 1)

    def test_failed_task_does_not_download_or_resubmit(self):
        self.session.request.side_effect = [response({"task_id": "x"}), response({"data": {"status": "FAILURE", "fail_reason": "unavailable"}})]
        with patch.object(api.time, "sleep"), self.assertRaisesRegex(api.ImageAPIError, "FAILURE"):
            self.client.generate(**self.args())
        self.assertEqual(self.session.request.call_count, 2)
        self.media.get.assert_not_called()

    def test_aitudou_mask_fails_before_upload(self):
        with self.assertRaisesRegex(ValueError, "mask"):
            self.client.generate(**self.args(mask=torch.zeros((1, 16, 32))))
        self.session.request.assert_not_called()

    def test_success_without_image_stops_polling(self):
        self.session.request.side_effect = [response({"task_id": "x"}), response({"data": {"status": "SUCCESS"}})]
        with patch.object(api.time, "sleep"), self.assertRaisesRegex(api.ImageAPIError, "没有图片"):
            self.client.generate(**self.args())
        self.assertEqual(self.session.request.call_count, 2)

    def test_mask_alpha_convention(self):
        files = api.encode_images({"image1": torch.zeros((1, 16, 32, 3))})
        raw = api.encode_mask(torch.ones((1, 16, 32)), files)
        self.assertEqual(Image.open(io.BytesIO(raw)).getpixel((0, 0))[3], 0)


if __name__ == "__main__":
    unittest.main()
