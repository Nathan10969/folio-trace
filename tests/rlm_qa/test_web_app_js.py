from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_web_app_initializes_when_optional_render_targets_are_missing() -> None:
    script = textwrap.dedent(
        r"""
        const fs = require("fs");
        const vm = require("vm");

        class Element {
          constructor(tagName = "div") {
            this.tagName = tagName;
            this.children = [];
            this.listeners = {};
            this.className = "";
            this.id = "";
            this.textContent = "";
            this.innerHTML = "";
            this.disabled = false;
            this.files = [];
            this.value = "";
          }
          addEventListener(name, handler) {
            this.listeners[name] = handler;
          }
          appendChild(child) {
            this.children.push(child);
            return child;
          }
          insertBefore(child) {
            this.children.push(child);
            return child;
          }
          querySelector(selector) {
            if (selector === "input") return new Element("input");
            return null;
          }
        }

        const elements = new Map([
          ["uploadForm", new Element("form")],
          ["fileInput", new Element("input")],
          ["chatForm", new Element("form")],
          ["questionInput", new Element("textarea")],
          ["askButton", new Element("button")],
          ["selectionSummary", new Element("p")],
          ["answerText", new Element("div")],
          ["modeTag", new Element("span")],
          ["answerTypeTag", new Element("span")],
          ["debugText", new Element("pre")],
          ["serviceStatus", new Element("span")],
        ]);

        const document = {
          body: new Element("body"),
          getElementById(id) {
            return elements.get(id) || null;
          },
          querySelector(selector) {
            if (selector === ".side-panel") return this.body;
            if (selector === "table") return new Element("table");
            return null;
          },
          createElement(tagName) {
            return new Element(tagName);
          },
        };

        vm.runInNewContext(fs.readFileSync("web/static/app.js", "utf8"), {
          document,
          FormData: class FormData {},
          fetch: async () => ({ ok: true, json: async () => ({}) }),
          console,
        });
        """
    )

    subprocess.run(["node", "-e", script], cwd=ROOT, check=True)


def test_web_app_state_distinguishes_processing_and_askable_documents() -> None:
    script = textwrap.dedent(
        r"""
        const assert = require("assert");
        const state = require("./web/static/app_state.js");

        assert.equal(state.isDocumentProcessing({ status: "ocr_queued" }), true);
        assert.equal(state.isDocumentProcessing({ status: "ocr_running" }), true);
        assert.equal(state.isDocumentAskable({ status: "ocr_queued" }), false);
        assert.equal(state.isDocumentAskable({ status: "ocr_failed" }), false);
        assert.equal(state.isDocumentAskable({ status: "direct_ready" }), true);
        assert.equal(state.isDocumentAskable({ status: "rlm_ready" }), true);
        assert.deepEqual(
          state.selectedSourceIds([
            { source_id: "queued", selected: true, status: "ocr_queued" },
            { source_id: "ready", selected: true, status: "rlm_ready" },
          ]),
          ["ready"],
        );
        """
    )

    subprocess.run(["node", "-e", script], cwd=ROOT, check=True)


def test_web_text_upload_returns_service_status_to_ready() -> None:
    script = textwrap.dedent(
        r"""
        const assert = require("assert");
        const fs = require("fs");
        const vm = require("vm");

        class Element {
          constructor(tagName = "div") {
            this.tagName = tagName;
            this.children = [];
            this.listeners = {};
            this.textContent = "";
            this.innerHTML = "";
            this.disabled = false;
            this.files = [];
            this.value = "";
            this.submitButton = new ElementButton();
          }
          addEventListener(name, handler) { this.listeners[name] = handler; }
          appendChild(child) { this.children.push(child); return child; }
          insertBefore(child) { this.children.push(child); return child; }
          querySelector(selector) {
            if (selector === "input") return new Element("input");
            if (selector === "button[type='submit']") return this.submitButton;
            return null;
          }
        }
        class ElementButton { constructor() { this.disabled = false; } }

        const elements = new Map([
          ["uploadForm", new Element("form")],
          ["fileInput", new Element("input")],
          ["chatForm", new Element("form")],
          ["questionInput", new Element("textarea")],
          ["askButton", new ElementButton()],
          ["selectionSummary", new Element("p")],
          ["answerText", new Element("div")],
          ["modeTag", new Element("span")],
          ["answerTypeTag", new Element("span")],
          ["debugText", new Element("pre")],
          ["serviceStatus", new Element("span")],
        ]);
        elements.get("fileInput").files = [{ name: "note.txt" }];

        const document = {
          body: new Element("body"),
          getElementById(id) { return elements.get(id) || null; },
          querySelector(selector) {
            if (selector === ".side-panel") return this.body;
            if (selector === "table") return new Element("table");
            return null;
          },
          createElement(tagName) { return new Element(tagName); },
        };
        const result = {
          source_id: "src_text",
          status: "direct_ready",
          doc_profile: { file_name: "note.txt", page_count: 1, chunk_count: 1 },
          manifest: { file_name: "note.txt" },
        };
        const context = {
          document,
          FormData: class FormData { append() {} },
          fetch: async () => ({ ok: true, json: async () => result }),
          console,
          setInterval,
          clearInterval,
          encodeURIComponent,
        };
        vm.runInNewContext(fs.readFileSync("web/static/app.js", "utf8"), context);
        (async () => {
          await elements.get("uploadForm").listeners.submit({ preventDefault() {} });
          assert.equal(elements.get("serviceStatus").textContent, "ready");
        })().catch((error) => { console.error(error); process.exit(1); });
        """
    )

    subprocess.run(["node", "-e", script], cwd=ROOT, check=True)
