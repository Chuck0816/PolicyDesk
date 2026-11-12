"""The MinerU 4.x adapter, run against stand-in MinerU modules."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import mineru_content_list


@pytest.fixture
def fake_mineru(monkeypatch):
    calls = {}
    assets = {"images/page_0_image_1.jpg": b"jpeg"}

    def parse(pdf, tier, page_range):
        calls["parse"] = (pdf.name, tier, page_range)
        return SimpleNamespace(middle_json="raw")

    def materialize_middle(middle_json):
        return "materialized", dict(assets)

    modules = {
        "mineru": ModuleType("mineru"),
        "mineru.parser": SimpleNamespace(parse=parse),
        "mineru.render": SimpleNamespace(
            render_content_list=lambda middle: [
                {"type": "image", "img_path": "images/page_0_image_1.jpg", "page_idx": 0}
            ]
        ),
        "docvortex": ModuleType("docvortex"),
        "docvortex.export": SimpleNamespace(
            materialize_middle=materialize_middle,
            validate_materialized_assets=lambda middle, found: None,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(calls=calls, assets=assets)


def test_adapter_writes_content_list_and_images(fake_mineru, tmp_path):
    path = mineru_content_list.convert(tmp_path / "报销制度.pdf", tmp_path / "out", "standard")
    assert path == (tmp_path / "out" / "报销制度_content_list.json").resolve()
    assert json.loads(path.read_text(encoding="utf-8"))[0]["img_path"] == (
        "images/page_0_image_1.jpg"
    )
    assert (tmp_path / "out" / "images" / "page_0_image_1.jpg").read_bytes() == b"jpeg"
    assert fake_mineru.calls["parse"] == ("报销制度.pdf", "standard", "")


def test_adapter_refuses_assets_outside_the_output_folder(fake_mineru, tmp_path):
    fake_mineru.assets["../escape.jpg"] = b"x"
    with pytest.raises(ValueError, match="leaves the output folder"):
        mineru_content_list.convert(tmp_path / "policy.pdf", tmp_path / "out", "standard")
    assert not (tmp_path / "escape.jpg").exists()
