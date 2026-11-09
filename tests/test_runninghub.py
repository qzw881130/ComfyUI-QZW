import importlib.util
import json
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
import torch

from test_image_api import api, cache, config, png, response


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("qzw_runninghub_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
sys.modules[package.__name__ + ".image_api"] = api
spec = importlib.util.spec_from_file_location(package.__name__ + ".runninghub_api", ROOT / "runninghub_api.py")
rh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh)
LOAD_IDS = ["470", "510", "503", "520", "521", "522", "523", "524", "525"]


def template():
    graph = {key: {"class_type": "LoadImage", "inputs": {"image": "old_example.png"}} for key in reversed(LOAD_IDS)}
    graph.update({
        "516": {"class_type": "CR Prompt Text", "inputs": {"prompt": "old prompt"}},
        "550": {"class_type": rh.IMAGE_CLASS, "inputs": {"prompt": ["516", 0], "seed": 5, "skip_error": True,
            "aspectRatio": "1:1", "resolution": "1k", **{f"image{i + 1}": [key, 0] for i, key in enumerate(LOAD_IDS)}}},
        "515": {"class_type": "SaveImage", "inputs": {"filename_prefix": "test", "images": ["550", 0]}},
        "999": {"class_type": "UnusedNode", "inputs": {}},
    })
    return graph


def final_result():
    return {"taskId": "task-one", "status": "SUCCESS", "errorCode": "", "errorMessage": "",
            "results": [{"url": "https://cdn.test/output.png", "outputType": "png"}]}


class RunningHubTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("No network in tests"))
        guard.start()
        self.addCleanup(guard.stop)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "cache.sqlite3"
        self.session, self.media = Mock(), Mock()
        self.config = config.make_config("https://www.runninghub.cn", "fake-rh-key")
        self.client = rh.RunningHubClient(self.config, session=self.session, media_session=self.media,
            upload_cache=cache.UploadCache(self.path, self.config.base_url, self.config.api_key, file_names=True))
        self.addCleanup(self.client.close)
        self.media.get.return_value = Mock(content=png((1672, 941)))

    def args(self):
        return dict(workflow_id=rh.DEFAULT_WORKFLOW_ID, prompt="a useful prompt", aspect_ratio="16:9", resolution="4k",
                    seed=123456, images={"image1": torch.zeros((1, 16, 32, 3))})

    def test_image_mapping_grows_past_template_without_mutating_it(self):
        original = template()
        for count in (1, 3, 9, 10):
            load_ids = (LOAD_IDS + ["1000"])[:count]
            files = [f"openapi/reference_{i}.png" for i in range(count)]
            graph, overrides, mapping = rh.build_workflow(original, files, "长提示词" * 3000, "16:9", "4k", 123)
            self.assertEqual(mapping["reference_nodes"], load_ids)
            self.assertEqual(set(graph), {*load_ids, "515", "516", "550"})
            self.assertEqual(graph["516"]["inputs"]["prompt"], "长提示词" * 3000)
            self.assertFalse(graph["550"]["inputs"]["skip_error"])
            self.assertEqual(overrides, [{"nodeId": "550", "fieldName": "seed", "fieldValue": 123}])
            for index, key in enumerate(load_ids):
                self.assertEqual(graph["550"]["inputs"][f"image{index + 1}"], [key, 0])
                self.assertEqual(graph[key]["inputs"]["image"], files[index])
        self.assertEqual(original, template())

    def test_all_images_reach_selected_channel_in_numeric_order(self):
        self.client.upload_cache = None
        for channel, count in (("economy", 10), ("stable", 15), ("stable", 16)):
            with self.subTest(channel=channel, count=count):
                self.session.reset_mock()
                self.session.request.side_effect = [response({"code": 0, "data": {"prompt": json.dumps(template())}}),
                    *[response({"code": 0, "data": {"fileName": f"openapi/ref_{i}.png"}}) for i in range(1, count + 1)],
                    response({"code": 0, "data": {"taskId": "task-one", "taskStatus": "QUEUED"}}),
                    response(final_result())]
                args = self.args()
                args.update(channel=channel, images={f"image{i}": torch.full((1, 16, 32, 3), i / count)
                                                     for i in reversed(range(1, count + 1))})
                _, _, report = self.client.generate(**args)
                submissions = [call for call in self.session.request.call_args_list if call.args[1].endswith("/task/openapi/create")]
                self.assertEqual(len(submissions), 1)
                graph = json.loads(submissions[0].kwargs["json"]["workflow"])
                self.assertEqual(graph["550"]["class_type"], rh.STABLE_IMAGE_CLASS if channel == "stable" else rh.IMAGE_CLASS)
                if channel == "stable":
                    self.assertEqual(graph["550"]["inputs"]["quality"], "high")
                    self.assertEqual(graph["550"]["inputs"]["outputFormat"], "png")
                for i in range(1, count + 1):
                    load_id = graph["550"]["inputs"][f"image{i}"][0]
                    self.assertEqual(graph[load_id]["inputs"]["image"], f"openapi/ref_{i}.png")
                self.assertEqual(json.loads(report)["request"]["reference_count"], count)
                self.assertEqual(json.loads(report)["channel"], channel)

    def test_stable_auto_ratio_and_template_preservation(self):
        original = template()
        graph, _, _ = rh.build_workflow(original, ["ref.png"] * 15, "test prompt", "auto", "4k", 123, "stable")
        self.assertNotIn("aspectRatio", graph["550"]["inputs"])
        self.assertEqual(graph["550"]["inputs"]["resolution"], "4k")
        self.assertEqual(original, template())

    def test_full_flow_uploads_filename_submits_once_and_downloads_without_key(self):
        self.session.request.side_effect = [response({"code": 0, "data": {"prompt": json.dumps(template())}}),
            response({"code": 0, "data": {"fileName": "openapi/ref.png", "download_url": "https://cdn.test/ref.png"}}),
            response({"code": 0, "data": {"taskId": "task-one", "taskStatus": "QUEUED"}}),
            response({"status": "RUNNING", "errorCode": "0"}), response(final_result())]
        with patch.object(rh.time, "sleep"):
            tensor, urls, report = self.client.generate(**self.args())
        calls = self.session.request.call_args_list
        self.assertEqual([c.args[1].replace(self.config.base_url, "") for c in calls], [
            "/api/openapi/getJsonApiFormat", "/openapi/v2/media/upload/binary", "/task/openapi/create",
            "/openapi/v2/query", "/openapi/v2/query"])
        submitted = calls[2].kwargs["json"]
        graph = json.loads(submitted["workflow"])
        self.assertEqual(graph["470"]["inputs"]["image"], "openapi/ref.png")
        self.assertEqual(set(graph), {"470", "515", "516", "550"})
        self.assertNotIn("retainSeconds", submitted)
        self.assertEqual(submitted["nodeInfoList"][0]["fieldValue"], 123456)
        self.assertEqual(calls[-1].kwargs["json"], {"taskId": "task-one"})
        self.assertEqual(tuple(tensor.shape), (1, 941, 1672, 3))
        self.assertEqual(urls, "https://cdn.test/output.png")
        self.assertNotIn("headers", self.media.get.call_args.kwargs)
        self.assertEqual(json.loads(report)["reference_uploads"], {"cache_hits": 0, "uploaded": 1})
        self.assertNotIn("fake-rh-key", report)

    def test_file_upload_cache_survives_new_client_and_obeys_signature_expiry(self):
        file = ("reference.png", png(), (32, 16))
        self.session.request.return_value = response({"code": 0, "data": {"fileName": "openapi/a.png",
            "download_url": "https://cdn.test/a.png?q-sign-time=10000%3B13600"}})
        with patch.object(rh.time, "time", return_value=10000):
            self.assertEqual(self.client.upload(file), "openapi/a.png")
        restarted = rh.RunningHubClient(self.config, session=self.session, media_session=Mock(),
            upload_cache=cache.UploadCache(self.path, self.config.base_url, self.config.api_key, file_names=True))
        self.addCleanup(restarted.close)
        with patch.object(rh.time, "time", return_value=12999):
            self.assertEqual(restarted.upload(file), "openapi/a.png")
        self.assertEqual(self.session.request.call_count, 1)
        self.session.request.return_value = response({"code": 0, "data": {"fileName": "openapi/b.png",
            "download_url": "https://cdn.test/b.png?q-sign-time=13000%3B16600"}})
        with patch.object(rh.time, "time", return_value=13000):
            self.assertEqual(restarted.upload(file), "openapi/b.png")
        self.assertEqual(self.session.request.call_count, 2)
        self.assertNotIn(b"fake-rh-key", self.path.read_bytes())

    def test_submission_timeout_is_never_retried(self):
        self.session.request.side_effect = requests.Timeout("fake-rh-key connection timeout")
        with self.assertRaisesRegex(api.ImageAPIError, "提交结果未知") as caught:
            self.client.post("/task/openapi/create", json={"apiKey": "fake-rh-key"})
        self.assertEqual(self.session.request.call_count, 1)
        self.assertNotIn("fake-rh-key", str(caught.exception))

    def test_failure_stops_polling_without_new_task(self):
        self.session.request.return_value = response({"status": "FAILED", "errorMessage": "failed fake-rh-key"})
        with self.assertRaisesRegex(api.ImageAPIError, "task-one FAILED") as caught:
            self.client.poll("task-one", 2, 10)
        self.assertEqual(self.session.request.call_count, 1)
        self.assertNotIn("fake-rh-key", str(caught.exception))

    def test_poll_connection_failure_recovers_using_same_task(self):
        self.session.request.side_effect = [requests.Timeout("temporary"), response(final_result())]
        with patch.object(rh.time, "sleep"):
            self.assertEqual(self.client.poll("task-one", 2, 10)["status"], "SUCCESS")
        for call in self.session.request.call_args_list:
            self.assertTrue(call.args[1].endswith("/openapi/v2/query"))
            self.assertEqual(call.kwargs["json"], {"taskId": "task-one"})

    def test_wrong_template_fails_before_upload(self):
        graph = template()
        graph["550"]["class_type"] = "OtherModel"
        self.session.request.return_value = response({"code": 0, "data": {"prompt": json.dumps(graph)}})
        with self.assertRaisesRegex(ValueError, "需要恰好一个"):
            self.client.generate(**self.args())
        self.assertEqual(self.session.request.call_count, 1)

    def test_too_many_images_fail_before_upload(self):
        self.session.request.return_value = response({"code": 0, "data": {"prompt": json.dumps(template())}})
        args = self.args()
        args["images"] = {"image1": torch.zeros((11, 16, 32, 3))}
        with self.assertRaisesRegex(ValueError, "当前输入 11 张"):
            self.client.generate(**args)
        self.assertEqual(self.session.request.call_count, 1)

    def test_stable_over_capacity_fails_before_upload(self):
        self.session.request.return_value = response({"code": 0, "data": {"prompt": json.dumps(template())}})
        args = self.args()
        args.update(channel="stable", images={"image1": torch.zeros((17, 16, 32, 3))})
        with self.assertRaisesRegex(ValueError, "当前输入 17 张"):
            self.client.generate(**args)
        self.assertEqual(self.session.request.call_count, 1)

    def test_upload_download_url_cannot_replace_missing_filename(self):
        self.session.request.return_value = response({"code": 0, "data": {"download_url": "https://cdn.test/ref.png"}})
        with self.assertRaisesRegex(api.ImageAPIError, "缺少 fileName"):
            self.client.upload(("reference.png", png(), (32, 16)))
        self.assertEqual(self.session.request.call_count, 1)

    def test_wrong_protocol_and_empty_images_fail_without_requests(self):
        self.client.config = config.APIConfig("https://api.aitudou.net", "fake-key", "aitudou")
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.client.generate(**self.args())
        self.client.config = self.config
        args = self.args()
        args["images"] = {}
        with self.assertRaisesRegex(ValueError, "至少需要"):
            self.client.generate(**args)
        self.session.request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
