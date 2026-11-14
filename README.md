# ComfyUI-QZW

GPT Image 2.5 文生图、多参考图编辑、RunningHub 云端工作流与独立 API 配置节点。使用 ComfyUI V3 节点接口，需要支持 DynamicCombo / Autogrow 的新版 ComfyUI 和前端。

## 安装

在 ComfyUI 的 `custom_nodes` 目录运行：

```sh
git clone https://github.com/qzw881130/ComfyUI-QZW.git
python -m pip install -r ComfyUI-QZW/requirements.txt
```

使用运行 ComfyUI 的 Python 安装依赖，然后重启 ComfyUI、刷新页面。搜索 `QZW API Config`、`QZW GPT Image 2.5` 或 `QZW RunningHub GPT Image 2.5`。PyTorch 使用 ComfyUI 已安装的版本。

## 使用

1. 在 `QZW API Config` 设置 `base_url` 和凭据，再把 `api_config` 接到图片节点。
2. `model` 提供 Flare / Sunburst 及其 2k / 4k 六个预设，以及 `aitudou-image-g-v2.5-flare`、`aitudou-image-g-v2.5-sunburst`、`aitudou-image-g-v2.5-lowprice`。选择 `custom` 后显示 `custom_model`，可以填写完整模型 ID，也可以把该字符串参数转为输入连接文本节点。
3. 填写提示词。文生图无需接图片；参考图按 `image1`、`image2` 顺序连接，输入口自动增加，最多显示 16 个。批次中的每一帧也会单独上传。服务端各模型的参考图数量限制仍适用。
4. `image` 连接保存图片节点，`response` 可连接文本预览。返回图片不会按请求尺寸强行缩放。

凭据读取顺序：`api_key` → `api_key_file` → `QZW_API_KEY` 环境变量。密钥文件为只含密钥的一行 UTF-8 文本，路径填写本机完整路径。推荐文件或环境变量方式；直接填写 `api_key` 会把密钥保存到工作流 JSON 中。示例工作流不含密钥。

`seed` 仅参与 ComfyUI 缓存，不发送给 API。相同输入可复用缓存；需要新图片时改变 seed。文件凭据修改后会重新读取。

## Aitudou

API Base URL：`https://api.aitudou.net`，`protocol=auto` 或 `aitudou`。

| 用法 | 设置 |
|---|---|
| 官方 Flare | 直接选择 `aitudou-image-g-v2.5-flare`；原有 `gpt-image-2.5-flare` 也会自动映射到此模型 |
| 官方 Sunburst | 直接选择 `aitudou-image-g-v2.5-sunburst`；原有 `gpt-image-2.5-sunburst` 也会自动映射到此模型 |
| 低价版 | 直接选择 `aitudou-image-g-v2.5-lowprice`；原有 `custom` 填写方式仍可使用 |
| 自定义模型 | 原样发送 `custom_model`；须使用所选协议并支持图片节点发送的字段 |

预设模型的 `-2k` / `-4k` 后缀转换成 `resolution`，官方版模型 ID 本身不带后缀。自定义模型名称不会做别名转换。

节点先复用有效的参考图缓存，未命中时调用 `/v1/files/upload` 上传，再向 `/v1/image/generations` 提交 JSON，并轮询 `/v1/image/generations/{id}`。Aitudou 始终使用异步任务，不受 `async_mode` 开关影响。图片上传与生成只在执行节点时发生。

参考图上传缓存默认启用，按实际上传 PNG 字节的 MD5、中转站地址和 API Key 指纹区分。修改提示词、seed、模型或参考图顺序不影响相同图片的缓存复用；重启后仍有效。有效期取上传响应 `expires_in`、COS 签名时间和文件最长保留 24 小时中的最早者，提前 10 分钟停止复用；命中缓存不会延长有效期，无法确定有效期的链接不缓存。过期记录会在后续写入时清理。

缓存保存在 ComfyUI 用户目录的 `__qzw/upload_cache.sqlite3`，遵循 `--user-directory` 设置，不在插件 Git 目录中。只保存内容摘要、账户作用域指纹、临时图片链接和过期时间，不保存图片或 API Key。日志显示 `upload cache hit`，`response.reference_uploads` 显示缓存命中与新上传的数量。高级选项 `force_upload=true` 可强制重新上传并刷新缓存，之后恢复为 false。缓存文件损坏或无法写入时会正常上传，不阻断生图。此缓存仅适用于 Aitudou 的独立上传接口，其他协议的 multipart 编辑流程不变。

官方版支持精确像素尺寸，`size=custom` 后可填宽高；服务端文档要求宽高为 16 的倍数、单边最多 3840、总像素 655360–8294400、最大比例 3:1。精确像素尺寸优先于 `resolution`。

低价版推荐 `size=16:9`、`resolution=1k/2k/4k`、`n=1`。如选择像素尺寸，节点仅将其约分为支持的比例，分辨率仍由 `resolution` 决定，并在响应中提示。低价版不发送 `quality`、`background`、`moderation`，使用 `nsfw_check`。**4k 是档位，不保证精确像素或原生 4K。** 低价版最多 15 张参考图，官方版最多 16 张，以服务端实际能力为准。

Aitudou 2.5 文档未提供 mask 参数，接入 mask 会在上传前明确报错。

低价版接口实际校验提示词为 5–5000 字符；节点在上传前检查长度，超限时提示当前字符数，不会自动截断内容。此检查仅针对 `aitudou-image-g-v2.5-lowprice`，不限制 Flare / Sunburst 的提示词长度；它们的提示词完整发送，实际支持长度由服务端决定。

文档：[官方版](https://api.aitudou.net/docs/#image-g-v25) · [低价版](https://api.aitudou.net/docs/#image-g-v25-ext) · [上传](https://api.aitudou.net/docs/#upload)

## 其他服务

| protocol | 提交 | 查询 |
|---|---|---|
| `openai` | `/v1/images/generations` 或 multipart `/v1/images/edits`，同步 | 不使用异步接口 |
| `workshop` | 同上，异步时加 `?async=true` | `/v1/images/tasks/{id}` |
| `sub2api` | 同上，异步时加 `/async` | `/v1/images/tasks/{id}` |

`auto` 识别 Aitudou、`ai.t8star.org/.cn` 和 RunningHub；其他域名默认 `openai`。使用 Sub2API 异步功能时请明确选择 `sub2api` 并启用 `async_mode`，服务端需要支持相应功能。RunningHub 工作流使用下述独立节点。输入是同一套节点界面，不表示任意模型支持所有尺寸、质量或参数。

OpenAI 编辑支持单图 mask：ComfyUI mask 白色区域转换为透明区域以进行重绘。

## RunningHub 云端工作流

使用 `QZW RunningHub GPT Image 2.5`，把 `QZW API Config` 的 `base_url` 设置为 `https://www.runninghub.cn`，`protocol` 选择 `runninghub`（该域名也可用 auto），配置自己的 RunningHub 密钥文件。请勿使用其他中转站的 API Key。

默认模板 ID 为 `2117652716037173249`，对应 `RH_RhartImageG25FlareImageToImage`。参考图入口随连接逐个增加，最多 15 个；模板中的 LoadImage 不足时，在本次提交的工作流中自动补齐，不修改云端保存的模板。设置 `prompt`、`aspect_ratio`、`resolution`、`seed`，把 image 输出接 SaveImage。此处不套用 Aitudou lowprice 的 5000 字符检查，提示词完整传给云端，实际限制由服务端决定。输出图片不缩放。

`channel=economy` 为默认经济版，最多接收 10 张；旧工作流保持此通道。11–15 张请选择 `channel=stable`：提交时替换为 `RH_RhartImageG25OfficialTokenFlareEdit` 稳定版，按 token 计费，使用 `quality=high`、`background=auto`、`outputFormat=png`。稳定版的 `aspect_ratio=auto` 不提交比例参数，交由服务端决定。图片批次会按顺序展开，稳定版底层最多接收 16 张。超出通道容量会在上传前提示，不会丢弃多出的图片或自动切换计费通道。

每次执行先通过 `/api/openapi/getJsonApiFormat` 读取当前云端模板，再按模型输入 image1、image2… 的顺序识别 LoadImage 节点。默认模板对应 `470、510、503、520、521、522、523、524、525`，提示词来源为 `516`、生成节点为 `550`、保存节点为 `515`。

上传接口 `/openapi/v2/media/upload/binary` 返回的 **fileName** 用于 LoadImage，不使用 download_url。节点根据实际参考图数量构造本次 API 格式工作流，去除未使用的图片连接与无关输出；单图时不会混入其余示例图。通过 `/task/openapi/create` 的 `workflow` 字段提交执行副本，并通过 `nodeInfoList` 显式指定 seed，不修改云端保存的模板。`workflow_id` 用于读取模板；提交完整 `workflow` 时按平台约定执行该副本。若换模板，需要仍包含一个 Flare 图生图节点、独立 LoadImage 输入和直接连接的 SaveImage。

结果通过 `/openapi/v2/query` 查询并下载，`response` 包含任务 ID、图片节点映射、缓存统计与实际图片尺寸。创建任务不自动重试，查询失败只重试原任务。不会设置会额外计费的 `retainSeconds`。RunningHub 的工作流运行和模型调用按平台规则计费。

读取模板遇到临时连接错误或 HTTP 408/429/500/502/503/504 时，最多尝试 3 次。轮询遇到同类错误会逐步延长间隔，在 `max_poll_time` 时限内继续查询原任务；鉴权、参数错误及云端明确失败不会重复轮询。查询超时表示云端结果尚未确认，不代表生图失败。

查询超时或结果下载失败后，在高级选项 `resume_task_id` 填入日志/错误中的原任务 ID，再执行即可查询和下载原结果。此模式不读取模板、不上传参考图、不新建任务，提示词可留空、参考图可不连接，`response.resumed=true`。需要生成新图时清空该字段；留空后重新执行仍会按普通流程创建新任务。恢复模式仍会执行已连接的上游节点，失效的参考图连接可先断开。

参考图同样使用 MD5 持久化缓存，文件为用户目录下 `__qzw/runninghub_upload_cache.sqlite3`，保存 fileName；与 Aitudou 缓存独立。有效期最多 24 小时，按响应/签名较早时间并预留 10 分钟，`force_upload=true` 可刷新缓存。单图与九图的 UI/API 示例位于 `examples/QZW_RunningHub_*.json`，均不含密钥。

接口文档：[读取工作流](https://www.runninghub.cn/runninghub-api-doc-cn/api-425749014) · [上传](https://www.runninghub.cn/runninghub-api-doc-cn/api-425749007) · [提交工作流](https://www.runninghub.cn/runninghub-api-doc-cn/api-425749013) · [查询 V2](https://www.runninghub.cn/runninghub-api-doc-cn/api-425767306)

## 诊断与费用

`response` 包含：实际协议、接口、请求模型和尺寸、脱敏后的提交字段、参考图尺寸、任务 ID、原始响应（省略 Base64）、实际解码尺寸和输出 tensor 形状。提示词正文及参考图签名地址不记录到请求摘要中。输出 URL 和 API 原始响应可能含临时结果链接，请按需保管。

发生尺寸不符时给出提示并返回实际图片；不会静默缩放，也不会用空白图掩盖请求失败。创建生图任务的 POST 超时或失败不会自动重新提交，避免重复计费；查询超时也不会新建任务。可用日志或 `response` 中的 task_id 查询原任务。停止 ComfyUI 执行并不取消服务端已提交的任务。

参考图上传失败发生在生图提交之前。错误信息包含参考图序号、尺寸、文件大小及脱敏后的网络错误详情。`SSLError` 表示 TLS 连接出错，需根据具体原因检查代理、证书或连接中断；节点保持证书校验，不自动切换代理。

## 工作流与测试

`examples/` 提供无密钥的界面工作流及 API 格式示例。九图编辑工作流沿用参考图→提示词→生成→保存/响应预览的结构；导入后选择自己的参考图并配置凭据。示例不会自行执行。

```sh
python -m unittest discover -s tests -v
```

单元测试强制禁止网络连接，覆盖模型映射、真实协议结构的模拟提交/轮询/下载、密钥脱敏、超时不重复提交、参考图顺序和 4K 解码不缩放。Aitudou 的真实生图按中转站规则计费，离线测试不会产生生图费用。

节点交互设计参考 [Comfyui-zhenzhen](https://github.com/T8mars/Comfyui-zhenzhen) 的 GPT Image 2.5 节点；本项目独立实现，不依赖该插件。
