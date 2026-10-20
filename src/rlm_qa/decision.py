from __future__ import annotations

from .config import Settings
from .contracts import DocProfile


COMPLEX_TERMS = ("对比", "比较", "统计", "表格", "全文", "所有", "分别", "列出", "总结")


def decide_mode(profile: DocProfile, question: str, settings: Settings) -> str:
    if profile.status in {"no_text", "failed"}:
        return "base_fallback"
    if not profile.rlm_ready and not profile.direct_ready:
        return "base_fallback"
    if profile.page_count > settings.direct_max_pages or profile.token_estimate > settings.direct_max_tokens:
        return "rlm"
    if any(term in question for term in COMPLEX_TERMS) and profile.page_count > 3:
        return "rlm"
    if profile.direct_ready:
        return "direct"
    return "rlm"
