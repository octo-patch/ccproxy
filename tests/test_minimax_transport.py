"""MiniMax defaults and explicit selections at the outgoing HTTP boundary."""

import json
from importlib import resources
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml
from mitmproxy import http
from mitmproxy.proxy.mode_specs import ProxyMode
from mitmproxy.test import tflow

from ccproxy.config import CCProxyConfig
from ccproxy.inspector import InspectorMeta
from ccproxy.inspector.routes.transform import make_transform_router
from ccproxy.specs.model_catalog import build_models_response


@pytest.mark.parametrize("region", ["api.minimax.io", "api.minimaxi.com"])
@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("model_id", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-model"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("explicit_options", [False, True])
def test_minimax_selected_model_reaches_http_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    region: str,
    protocol: str,
    model_id: str,
    stream: bool,
    explicit_options: bool,
) -> None:
    templates = resources.files("ccproxy.templates")
    native = yaml.safe_load(templates.joinpath("ccproxy.yaml").read_text())
    models = yaml.safe_load(templates.joinpath("config.yaml").read_text())
    endpoint = "/v1/messages" if protocol == "anthropic" else "/chat/completions"
    base_url = f"https://{region}/{'anthropic' if protocol == 'anthropic' else 'v1'}"
    native["ccproxy"]["providers"]["minimax"].update(type=protocol, base_url=base_url, path=endpoint, auth=None)
    monkeypatch.setenv("CCPROXY_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    # User-owned legacy IDs and aliases must survive normal config compilation.
    if model_id in {"MiniMax-M2.7", "custom-model"}:
        models["model_list"].append(
            {"model_name": "saved-selection", "litellm_params": {"model": f"minimax/{model_id}"}}
        )
        selection = "saved-selection"
    else:
        selection = model_id
    native_path = tmp_path / "ccproxy.yaml"
    native_path.write_text(yaml.safe_dump(native))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(models))
    config = CCProxyConfig.from_yaml(native_path)
    choices = [entry["id"] for entry in build_models_response(config)["data"] if entry["owned_by"] == "minimax"]
    assert choices[:2] == ["MiniMax-M3.1-Flash", "MiniMax-M3"]

    image = (
        {"type": "image", "source": {"type": "url", "url": "https://example.invalid/image.png"}}
        if protocol == "anthropic"
        else {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}
    )
    body = {
        "model": selection,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}, image]}],
        "max_tokens": 32,
        "stream": stream,
    }
    if explicit_options and model_id == "MiniMax-M3.1-Flash":
        if protocol == "anthropic":
            body.update(thinking={"type": "adaptive"}, output_config={"effort": "max"})
        else:
            body.update(reasoning_effort="max")
    elif explicit_options and model_id == "MiniMax-M3":
        body.update(thinking={"type": "disabled"}, service_tier="priority")

    flow = tflow.tflow()
    flow.metadata = {InspectorMeta.DIRECTION: "inbound"}
    flow.client_conn.proxy_mode = ProxyMode.parse("reverse:http://localhost:1@4001")
    inbound_path = "/v1/messages" if protocol == "anthropic" else "/v1/chat/completions"
    flow.request = http.Request.make(
        "POST",
        f"http://localhost:4000{inbound_path}",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with patch("ccproxy.inspector.routes.transform.get_config", return_value=config):
        make_transform_router().request(flow)

    captured = []

    def capture(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True})

    # Replay exactly the route-produced URL, headers and bytes into a mock transport;
    # no live service or paid inference is used.
    with httpx.Client(transport=httpx.MockTransport(capture)) as client:
        client.post(flow.request.url, headers=dict(flow.request.headers), content=flow.request.raw_content)
    assert len(captured) == 1
    assert str(captured[0].url) == base_url + endpoint
    sent = json.loads(captured[0].content)
    assert sent == {**body, "model": model_id}
    if model_id == "MiniMax-M3.1-Flash":
        assert "service_tier" not in sent
        assert "reasoning_split" not in sent
        assert sent.get("thinking", {"type": "adaptive"}) == {"type": "adaptive"}
        assert sent.get("reasoning_effort", "max") == "max"
