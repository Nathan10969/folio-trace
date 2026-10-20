from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm_qa.config import Settings
from rlm_qa.platform_documents import PlatformDocumentError, PlatformDocumentLoader


def write_glm_document(
    root: Path,
    dir_name: str,
    *,
    status: str = "ok",
    include_elements: bool = True,
) -> Path:
    target = root / dir_name
    target.mkdir(parents=True)
    (target / "status.json").write_text(
        json.dumps({"status": status, "pages": 2}),
        encoding="utf-8",
    )
    (target / "metadata.json").write_text(
        json.dumps({"fileId": 501, "fileName": "swcnt.pdf"}),
        encoding="utf-8",
    )
    if include_elements:
        rows = [
            {
                "page": 1,
                "index": 0,
                "label": "text",
                "content": "Single-walled carbon nanotubes are cylindrical carbon materials.",
            },
            {
                "page": 1,
                "index": 1,
                "label": "image",
                "content": "https://example.invalid/image.png",
            },
            {
                "page": 2,
                "index": 0,
                "label": "table",
                "content": "Property | Dependence\nConductivity | chirality",
            },
            {
                "page": 2,
                "index": 1,
                "label": "text",
                "content": "Their electronic properties depend on diameter and chirality.",
            },
        ]
        (target / "elements.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    return target


def make_loader(tmp_path: Path) -> PlatformDocumentLoader:
    settings = Settings(
        platform_data_root=tmp_path / "glm",
        platform_cache_dir=tmp_path / "cache",
        chunk_chars=120,
        chunk_overlap_chars=20,
    )
    return PlatformDocumentLoader(
        settings.platform_data_root,
        settings.platform_cache_dir,
        settings,
    )


def test_loader_preserves_pages_metadata_and_evidence(tmp_path: Path) -> None:
    dir_name = "a" * 64
    write_glm_document(tmp_path / "glm", dir_name)
    loader = make_loader(tmp_path)

    document = loader.load(dir_name, "501")

    assert document.dir_name == dir_name
    assert document.doc_id == "501"
    assert document.profile.file_name == "swcnt.pdf"
    assert document.profile.page_count == 2
    assert document.profile.source_id.startswith("src_platform_")
    assert [page.page_number for page in document.pages] == [1, 2]
    assert "Single-walled carbon nanotubes" in document.pages[0].text
    assert "example.invalid" not in document.pages[0].text
    assert "Conductivity | chirality" in document.pages[1].text
    assert document.chunks
    assert {evidence.page_number for evidence in document.evidences} == {1, 2}
    assert document.cache_hit is False


def test_loader_uses_metadata_file_id_when_payload_doc_id_is_absent(tmp_path: Path) -> None:
    dir_name = "b" * 64
    write_glm_document(tmp_path / "glm", dir_name)

    document = make_loader(tmp_path).load(dir_name)

    assert document.doc_id == "501"


def test_catalog_reads_document_handles_without_loading_page_text(tmp_path: Path) -> None:
    dir_name = "d" * 64
    write_glm_document(tmp_path / "glm", dir_name, include_elements=False)
    loader = make_loader(tmp_path)

    catalog = loader.catalog_many([dir_name], ["9001"])

    assert len(catalog) == 1
    assert catalog[0].position == 1
    assert catalog[0].dir_name == dir_name
    assert catalog[0].doc_id == "9001"
    assert catalog[0].file_name == "swcnt.pdf"
    assert catalog[0].page_count == 2
    assert catalog[0].source_id.startswith("src_platform_")
    assert catalog[0].status == "available"
    assert not (tmp_path / "cache" / dir_name).exists()


def test_catalog_keeps_unavailable_handle_without_blocking_base_conversation(tmp_path: Path) -> None:
    loader = make_loader(tmp_path)

    catalog = loader.catalog_many(["missing-document"], ["999"])

    assert len(catalog) == 1
    assert catalog[0].position == 1
    assert catalog[0].doc_id == "999"
    assert catalog[0].file_name == "document-1.pdf"
    assert catalog[0].page_count == 0
    assert catalog[0].status == "unavailable"


def test_loader_reuses_cache_and_invalidates_when_elements_change(tmp_path: Path) -> None:
    dir_name = "c" * 64
    target = write_glm_document(tmp_path / "glm", dir_name)
    loader = make_loader(tmp_path)

    first = loader.load(dir_name, "501")
    second = loader.load(dir_name, "501")
    rows_path = target / "elements.jsonl"
    rows_path.write_text(
        rows_path.read_text(encoding="utf-8")
        + json.dumps({"page": 2, "index": 2, "label": "text", "content": "New result."})
        + "\n",
        encoding="utf-8",
    )
    third = loader.load(dir_name, "501")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.fingerprint == first.fingerprint
    assert third.cache_hit is False
    assert third.fingerprint != first.fingerprint
    assert "New result." in third.pages[1].text


@pytest.mark.parametrize(
    "dir_name",
    ["", ".", "..", "../escape", "/tmp/escape", "nested/name", "nested\\name"],
)
def test_loader_rejects_unsafe_directory_names(tmp_path: Path, dir_name: str) -> None:
    loader = make_loader(tmp_path)

    with pytest.raises(PlatformDocumentError) as captured:
        loader.load(dir_name, "501")

    assert captured.value.code == "DIR_NAME_INVALID"
    assert captured.value.status_code == 400


def test_loader_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "glm"
    root.mkdir()
    outside = write_glm_document(tmp_path / "outside", "real")
    try:
        (root / ("d" * 64)).symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows requires symlink creation privilege; covered by Linux CI")
        raise

    with pytest.raises(PlatformDocumentError) as captured:
        make_loader(tmp_path).load("d" * 64, "501")

    assert captured.value.code == "DIR_PATH_ESCAPE"


def test_loader_rejects_unready_and_empty_documents(tmp_path: Path) -> None:
    unready = "e" * 64
    empty = "f" * 64
    write_glm_document(tmp_path / "glm", unready, status="processing")
    write_glm_document(tmp_path / "glm", empty, include_elements=False)
    loader = make_loader(tmp_path)

    with pytest.raises(PlatformDocumentError) as unready_error:
        loader.load(unready, "501")
    with pytest.raises(PlatformDocumentError) as empty_error:
        loader.load(empty, "501")

    assert unready_error.value.code == "DIR_NOT_READY"
    assert unready_error.value.status_code == 422
    assert empty_error.value.code == "DIR_NO_TEXT"
    assert empty_error.value.status_code == 422


def test_load_many_requires_doc_ids_to_align_when_supplied(tmp_path: Path) -> None:
    first = "1" * 64
    second = "2" * 64
    write_glm_document(tmp_path / "glm", first)
    write_glm_document(tmp_path / "glm", second)
    loader = make_loader(tmp_path)

    with pytest.raises(PlatformDocumentError) as captured:
        loader.load_many([first, second], ["501"])

    assert captured.value.code == "DOC_ID_COUNT_MISMATCH"
    assert captured.value.status_code == 400


def test_cache_keeps_metadata_file_id_separate_from_request_doc_id(tmp_path: Path) -> None:
    dir_name = "9" * 64
    write_glm_document(tmp_path / "glm", dir_name)
    loader = make_loader(tmp_path)

    first = loader.load(dir_name, "request-doc-id")
    second = loader.load(dir_name)

    assert first.doc_id == "request-doc-id"
    assert second.cache_hit is True
    assert second.doc_id == "501"


def test_loader_rejects_artifact_symlink_escape(tmp_path: Path) -> None:
    dir_name = "8" * 64
    target = write_glm_document(tmp_path / "glm", dir_name)
    external = tmp_path / "external-elements.jsonl"
    external.write_text(
        json.dumps({"page": 1, "index": 0, "label": "text", "content": "outside"}) + "\n",
        encoding="utf-8",
    )
    (target / "elements.jsonl").unlink()
    try:
        (target / "elements.jsonl").symlink_to(external)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows requires symlink creation privilege; covered by Linux CI")
        raise

    with pytest.raises(PlatformDocumentError) as captured:
        make_loader(tmp_path).load(dir_name, "501")

    assert captured.value.code == "DIR_PATH_ESCAPE"


def test_cache_write_succeeds_when_final_path_fits_filesystem_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dir_name = "a" * 64
    write_glm_document(tmp_path / "glm", dir_name)
    loader = make_loader(tmp_path)
    cache_directory = loader.cache_dir / dir_name
    maximum_path_length = len(str(cache_directory / ("f" * 64 + ".json")))
    real_write_text = Path.write_text

    def bounded_write_text(path: Path, *args, **kwargs):
        if path.is_relative_to(loader.cache_dir) and len(str(path)) > maximum_path_length:
            raise OSError("filesystem path limit exceeded")
        return real_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", bounded_write_text)
    first = loader.load(dir_name, "501")
    second = loader.load(dir_name, "501")

    assert second.cache_hit is True
    assert second.pages == first.pages
    assert len(list(cache_directory.glob("*.json"))) == 1
    assert not list(cache_directory.glob("*.tmp"))
