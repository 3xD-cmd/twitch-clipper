"""
studio_timeline.py
CapCut style jump cut editor for the TwitchClipper Studio (v3, new module).

A single Gradio 6 custom gr.HTML component (html_template / css_template / js_on_load) that
shows, for one clip:
  * the low resolution preview video on top (the preview window cut by Stage 3)
  * a left to right time ruler with the VOD clock
  * the waveform from precomputed loudness peaks (no audio decoding in the browser)
  * a laugh track overlay, word chips, and protected visual beats as shaded bands
  * draggable piece blocks (the kept pieces of the jump cut), snapping to word edges
    (hold Alt to drag freely)
  * virtual jump cut playback: plays only the kept pieces and skips the gaps, so the editor
    hears the exact cut before any 1080p render

Data flow
  Python -> browser: return timeline_value(...) as the component value. The JS watches
    the value and re-renders whenever the clip or its revision changes. The HTML template
    itself is constant, so Gradio never wipes the editor DOM.
  Browser -> Python: every edit sets props.value (pieces included, committed=False) so the
    current state is always readable as an input. The "✅ اعتماد التعديل (جاهز للرندر)"
    button (or Ctrl+Enter) sets committed=True and fires the custom "commit" event:
        tl.commit(on_commit, inputs=[tl, ...], outputs=[...])
    If a hidden textbox built with build_bridge(elem_id) exists, the same JSON is also
    written there (a second, plain channel for older Gradio builds).
  parse_timeline_edit(payload, window, vod_duration) validates and clamps the result.

Keyboard (when the editor has focus)
  Space play/pause the cut | J back 1 s | K pause | L play (again: 1.5x, 2x)
  S split at the playhead | Delete or Backspace remove the active piece
  Left / Right nudge the selected edge (or the playhead) 0.1 s, with Shift 0.5 s
  Ctrl+Z undo | Ctrl+Shift+Z or Ctrl+Y redo | Home go to the first piece | Ctrl+Enter approve
  Mouse: click a piece to select it, click it again to switch it off or on; drag its edges;
  drag on an empty area to add a piece; click the ruler or the waveform to seek;
  Ctrl + wheel to zoom.

Public contract
  timeline_supported() -> bool
  build_timeline(elem_id="tc_timeline", **html_kwargs) -> gr.HTML
  build_bridge(elem_id="tc_timeline") -> gr.Textbox (hidden JSON channel, optional)
  timeline_value(preview_url, window, pieces, words, peaks, laugh, beats, fps, rev) -> dict
  empty_timeline_value(message="") -> dict
  parse_timeline_edit(payload, window, vod_duration, min_piece_s=0.2) -> Optional[dict]
  file_url(path) -> str
  COMMIT_EVENT = "commit"

Everything (HTML, CSS, JS) is inline: no CDN, so it works over Colab share tunnels.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import urllib.parse
from typing import Any, Optional, Sequence

from common import clamp, coerce_number, text_hash

COMMIT_EVENT = "commit"
TIMELINE_SCHEMA = 1
MAX_WORDS = 1500
MAX_PEAKS = 4000
MERGE_GAP_S = 0.05


# ============================================================================
# Capability check and file URLs
# ============================================================================

def timeline_supported() -> bool:
    """True when the installed Gradio has custom gr.HTML components (Gradio 6:
    html_template, css_template and js_on_load). The Studio falls back to the v2 controls
    when this returns False."""
    try:
        import gradio as gr
        major = int(str(getattr(gr, "__version__", "0")).split(".")[0])
        if major < 6:
            return False
        params = inspect.signature(gr.HTML.__init__).parameters
        return all(name in params for name in ("html_template", "css_template", "js_on_load"))
    except Exception:
        return False


def file_url(path: str) -> str:
    """URL Gradio serves a local file from (the folder must be in launch(allowed_paths=...)).
    The path is percent encoded, so Arabic and spaces in file names are safe."""
    if not path:
        return ""
    absolute = os.path.abspath(str(path))
    return "/gradio_api/file=" + urllib.parse.quote(absolute, safe="/")


# ============================================================================
# Value builder (Python -> browser)
# ============================================================================

def _num(v: Any) -> Optional[float]:
    val = coerce_number(v, None)
    if val is None:
        return None
    val = float(val)
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _window(window: Any) -> Optional[tuple]:
    if not isinstance(window, (list, tuple)) or len(window) < 2:
        return None
    w0, w1 = _num(window[0]), _num(window[1])
    if w0 is None or w1 is None or w1 - w0 < 0.5:
        return None
    return max(0.0, w0), w1


def _clean_pieces(pieces: Any, w0: float, w1: float, min_len: float = 0.05) -> list:
    out = []
    for p in pieces or []:
        s = e = None
        on = True
        if isinstance(p, dict):
            s, e = _num(p.get("s", p.get("start"))), _num(p.get("e", p.get("end")))
            on = bool(p.get("on", True))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            s, e = _num(p[0]), _num(p[1])
            if len(p) >= 3:
                on = bool(p[2])
        if s is None or e is None:
            continue
        s, e = max(w0, min(s, e)), min(w1, max(s, e))
        if e - s >= min_len:
            out.append([round(s, 3), round(e, 3), on])
    out.sort(key=lambda x: x[0])
    return out


def _clean_words(words: Any, w0: float, w1: float) -> list:
    out = []
    for w in words or []:
        if isinstance(w, dict):
            text, s, e = w.get("w", w.get("word", "")), _num(w.get("s", w.get("start"))), _num(w.get("e", w.get("end")))
        elif isinstance(w, (list, tuple)) and len(w) >= 3:
            text, s, e = w[0], _num(w[1]), _num(w[2])
        else:
            continue
        if s is None or e is None or e < w0 or s > w1:
            continue
        text = " ".join(str(text or "").split())
        if not text:
            continue
        out.append([text[:40], round(max(w0, s), 3), round(min(w1, max(s, e)), 3)])
    out.sort(key=lambda x: x[1])
    return out[:MAX_WORDS]


def _clean_series(values: Any, limit: int = MAX_PEAKS) -> list:
    out = []
    for v in values or []:
        val = _num(v)
        out.append(round(float(clamp(val if val is not None else 0.0, 0.0, 1.0)), 3))
    if len(out) > limit:
        step = len(out) / float(limit)
        out = [max(out[int(i * step):max(int(i * step) + 1, int((i + 1) * step))]) for i in range(limit)]
    return out


def _clean_spans(values: Any, w0: float, w1: float) -> list:
    out = []
    for v in values or []:
        if isinstance(v, dict):
            s, e = _num(v.get("s", v.get("start"))), _num(v.get("e", v.get("end")))
            label = v.get("what", v.get("label", ""))
            strength = _num(v.get("score", v.get("p", 1.0)))
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            s, e = _num(v[0]), _num(v[1])
            label = v[2] if len(v) > 2 and isinstance(v[2], str) else ""
            strength = _num(v[2]) if len(v) > 2 and not isinstance(v[2], str) else 1.0
        else:
            continue
        if s is None or e is None or e <= s or e < w0 or s > w1:
            continue
        out.append([round(max(w0, s), 3), round(min(w1, e), 3), str(label or "")[:80],
                    round(float(clamp(strength if strength is not None else 1.0, 0.0, 1.0)), 3)])
    out.sort(key=lambda x: x[0])
    return out


def timeline_value(preview_url: str, window: Sequence[float], pieces: Sequence[Sequence[float]],
                   words: Sequence[Any], peaks: Sequence[float], laugh: Any, beats: Sequence[Any],
                   fps: Optional[float], rev: Any) -> dict:
    """Component value for one clip.
      preview_url  URL of the preview MP4 (file_url(path)); its time 0 is window[0]
      window       [t0, t1] in VOD seconds covered by the preview
      pieces       kept pieces [[s, e], ...] in VOD seconds
      words        [{"w", "s", "e"}, ...] (only the ones inside the window are kept)
      peaks        loudness values 0..1 spread evenly over the window (media_engine.waveform_peaks)
      laugh        either values 0..1 spread evenly over the window, or spans [[s, e, score], ...]
      beats        protected visual beats [[s, e, what], ...]
      fps          source fps (edges are shown on the frame grid), None when unknown
      rev          the clip's edit revision, echoed back by the browser"""
    win = _window(window)
    if win is None:
        return empty_timeline_value("ما فيه نافذة معاينة صالحة لهذي اللقطة")
    w0, w1 = win
    laugh_vals: list = []
    laugh_spans: list = []
    if laugh:
        first = next(iter(laugh), None)
        if isinstance(first, (list, tuple, dict)):
            laugh_spans = _clean_spans(laugh, w0, w1)
        else:
            laugh_vals = _clean_series(laugh)
    clean_pieces = _clean_pieces(pieces, w0, w1)
    fps_val = _num(fps)
    value = {
        "schema": TIMELINE_SCHEMA,
        "src": str(preview_url or ""),
        "window": [round(w0, 3), round(w1, 3)],
        "pieces": [[s, e] for s, e, on in clean_pieces if on],
        "all_pieces": clean_pieces,
        "orig_pieces": [[s, e] for s, e, on in clean_pieces if on],
        "words": _clean_words(words, w0, w1),
        "peaks": _clean_series(peaks),
        "laugh": laugh_vals,
        "laugh_spans": laugh_spans,
        "beats": _clean_spans(beats, w0, w1),
        "fps": round(fps_val, 4) if fps_val and fps_val > 0 else None,
        "rev": rev if isinstance(rev, (int, str)) else str(rev),
        "committed": False,
        "dirty": False,
        "message": "",
    }
    value["key"] = text_hash(json.dumps([value["src"], value["window"], value["orig_pieces"], value["rev"]],
                                        ensure_ascii=False), 12)
    return value


def empty_timeline_value(message: str = "") -> dict:
    """Value that shows a message instead of the editor (no clip selected, no preview)."""
    return {"schema": TIMELINE_SCHEMA, "src": "", "window": [0.0, 0.0], "pieces": [], "all_pieces": [],
            "orig_pieces": [], "words": [], "peaks": [], "laugh": [], "laugh_spans": [], "beats": [],
            "fps": None, "rev": "", "committed": False, "dirty": False,
            "message": str(message or "اختر لقطة عشان يظهر محرر القص"), "key": "empty"}


# ============================================================================
# Edit parser (browser -> Python)
# ============================================================================

def parse_timeline_edit(payload: Any, window: Sequence[float], vod_duration: float,
                        min_piece_s: float = 0.2) -> Optional[dict]:
    """Validate an edit sent by the editor. Returns
      {"rev", "pieces", "committed", "net", "count"}
    with pieces sorted, merged, clamped to the window and the VOD, and at least min_piece_s
    long. Returns None when the payload is not an edit, belongs to another clip window
    (stale editor) or keeps nothing."""
    if payload is None:
        return None
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return None
    if not isinstance(payload, dict):
        return None
    win = _window(window)
    if win is None:
        return None
    w0, w1 = win
    pay_win = _window(payload.get("window"))
    if pay_win is not None and (abs(pay_win[0] - w0) > 0.05 or abs(pay_win[1] - w1) > 0.05):
        return None
    vod_end = _num(vod_duration)
    hi = min(w1, vod_end) if vod_end and vod_end > 0 else w1
    lo = max(0.0, w0)
    if hi - lo < min_piece_s:
        return None
    raw = payload.get("pieces")
    if not isinstance(raw, (list, tuple)):
        raw = [p for p in payload.get("all_pieces") or [] if isinstance(p, (list, tuple)) and (len(p) < 3 or p[2])]
    spans = []
    for s, e, on in _clean_pieces(raw, lo, hi, min_len=0.0):
        if on:
            spans.append([s, e])
    merged: list = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1] + MERGE_GAP_S:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    min_len = max(0.0, float(min_piece_s or 0.0))
    pieces = [[round(s, 3), round(e, 3)] for s, e in merged if e - s >= min_len]
    if not pieces:
        return None
    rev = payload.get("rev")
    return {
        "rev": rev if isinstance(rev, (int, str)) else (str(rev) if rev is not None else None),
        "pieces": pieces,
        "committed": bool(payload.get("committed")),
        "net": round(sum(e - s for s, e in pieces), 3),
        "count": len(pieces),
    }


# ============================================================================
# Front end (inline HTML, CSS, JS)
# ============================================================================

HTML_TEMPLATE = (
    '<div class="tcl-root" tabindex="0" dir="ltr">'
    '<div class="tcl-boot">⏳ جاري تحميل محرر القص...</div>'
    '</div>'
)

# Note: css_template is evaluated by Gradio as a JS template string and scoped to the
# component, so it must not contain backticks, the dollar brace sequence or double braces.
CSS_TEMPLATE = r"""
.tcl-root { position: relative; outline: none; font-family: system-ui, "Segoe UI", Tahoma, sans-serif; color: #e8e8ee; background: #15161c; border: 1px solid #2c2e3a; border-radius: 12px; padding: 10px; user-select: none; -webkit-user-select: none; }
.tcl-root:focus { border-color: #5b8cff; box-shadow: 0 0 0 2px rgba(91, 140, 255, 0.25); }
.tcl-boot, .tcl-msg { padding: 28px 12px; text-align: center; color: #b9bccb; font-size: 15px; direction: rtl; }
.tcl-top { display: flex; gap: 10px; align-items: stretch; flex-wrap: wrap; }
.tcl-videobox { position: relative; flex: 0 0 auto; width: 100%; max-width: 640px; margin: 0 auto 8px auto; background: #000; border-radius: 8px; overflow: hidden; aspect-ratio: 16 / 9; }
.tcl-video { width: 100%; height: 100%; display: block; background: #000; object-fit: contain; }
.tcl-novideo { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #9aa0b4; font-size: 14px; direction: rtl; text-align: center; padding: 10px; }
.tcl-clock { position: absolute; left: 8px; bottom: 8px; background: rgba(0, 0, 0, 0.65); color: #fff; font: 600 12px ui-monospace, Menlo, Consolas, monospace; padding: 3px 7px; border-radius: 6px; direction: ltr; }
.tcl-modebadge { position: absolute; right: 8px; top: 8px; background: rgba(0, 0, 0, 0.6); color: #ffd166; font-size: 12px; padding: 3px 8px; border-radius: 6px; direction: rtl; }
.tcl-bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0 8px 0; direction: rtl; }
.tcl-btn { appearance: none; border: 1px solid #3a3d4d; background: #23252f; color: #e8e8ee; border-radius: 8px; padding: 6px 10px; font-size: 13px; cursor: pointer; line-height: 1.2; white-space: nowrap; }
.tcl-btn:hover { background: #2d3040; border-color: #4a4e63; }
.tcl-btn:active { transform: translateY(1px); }
.tcl-btn.tcl-on { background: #2b3b63; border-color: #5b8cff; }
.tcl-btn.tcl-approve { background: #1f6f43; border-color: #2fa765; font-weight: 700; }
.tcl-btn.tcl-approve:hover { background: #248451; }
.tcl-btn.tcl-approve.tcl-dirty { box-shadow: 0 0 0 2px rgba(255, 209, 102, 0.55); }
.tcl-btn[disabled] { opacity: 0.45; cursor: default; }
.tcl-sep { width: 1px; height: 22px; background: #3a3d4d; margin: 0 2px; }
.tcl-status { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; font-size: 13px; color: #c9ccda; margin: 2px 0 6px 0; direction: rtl; }
.tcl-status b { color: #fff; }
.tcl-state { padding: 2px 8px; border-radius: 6px; font-size: 12px; }
.tcl-state.tcl-clean { background: #1e3a2b; color: #8ee3b0; }
.tcl-state.tcl-dirtyst { background: #4a3b12; color: #ffd166; }
.tcl-state.tcl-sent { background: #1f2d4d; color: #9fc0ff; }
.tcl-scroll { position: relative; overflow-x: auto; overflow-y: hidden; background: #0f1015; border: 1px solid #2a2c37; border-radius: 8px; direction: ltr; scrollbar-color: #444 #15161c; }
.tcl-track { position: relative; height: 170px; cursor: crosshair; }
.tcl-canvas { position: absolute; top: 0; left: 0; height: 170px; pointer-events: none; }
.tcl-piece { position: absolute; top: 24px; height: 92px; background: rgba(64, 196, 128, 0.20); border: 2px solid rgba(64, 196, 128, 0.95); border-radius: 6px; box-sizing: border-box; cursor: pointer; }
.tcl-piece.tcl-off { background: repeating-linear-gradient(45deg, rgba(120, 120, 130, 0.18) 0 6px, rgba(60, 60, 70, 0.18) 6px 12px); border-color: rgba(150, 150, 160, 0.8); border-style: dashed; }
.tcl-piece.tcl-active { border-color: #ffd166; box-shadow: 0 0 0 2px rgba(255, 209, 102, 0.35); z-index: 3; }
.tcl-piece .tcl-h { position: absolute; top: -2px; bottom: -2px; width: 10px; cursor: ew-resize; background: rgba(64, 196, 128, 0.85); }
.tcl-piece .tcl-h.tcl-l { left: -2px; border-radius: 6px 0 0 6px; }
.tcl-piece .tcl-h.tcl-r { right: -2px; border-radius: 0 6px 6px 0; }
.tcl-piece.tcl-active .tcl-h { background: rgba(255, 209, 102, 0.9); }
.tcl-piece .tcl-h.tcl-edge { background: #ff8c42; }
.tcl-piece .tcl-num { position: absolute; left: 12px; top: 2px; font: 600 11px ui-monospace, Menlo, Consolas, monospace; color: #dff7ea; pointer-events: none; white-space: nowrap; }
.tcl-ghost { position: absolute; top: 24px; height: 92px; background: rgba(91, 140, 255, 0.25); border: 2px dashed #5b8cff; border-radius: 6px; box-sizing: border-box; pointer-events: none; }
.tcl-playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: #ff4d6d; pointer-events: none; z-index: 5; }
.tcl-playhead::before { content: ""; position: absolute; top: 0; left: -5px; width: 12px; height: 10px; background: #ff4d6d; clip-path: polygon(0 0, 100% 0, 50% 100%); }
.tcl-help { margin-top: 6px; font-size: 12px; color: #9aa0b4; line-height: 1.7; direction: rtl; }
.tcl-help kbd { background: #23252f; border: 1px solid #3a3d4d; border-bottom-width: 2px; border-radius: 4px; padding: 0 5px; font: 11px ui-monospace, Menlo, Consolas, monospace; color: #e8e8ee; }
.tcl-toast { position: absolute; left: 50%; top: 12px; transform: translateX(-50%); background: rgba(20, 22, 30, 0.95); border: 1px solid #5b8cff; color: #fff; padding: 6px 12px; border-radius: 8px; font-size: 13px; z-index: 20; direction: rtl; pointer-events: none; opacity: 0; transition: opacity 0.2s; }
.tcl-toast.tcl-show { opacity: 1; }
"""

JS_ON_LOAD = r"""
(function () {
  var root = element.querySelector('.tcl-root') || element;
  if (root.__tcl_ready) { return; }
  root.__tcl_ready = true;

  // ------------------------------------------------------------ helpers
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) { n.className = cls; }
    if (text !== undefined && text !== null) { n.textContent = text; }
    return n;
  }
  function r3(x) { return Math.round(x * 1000) / 1000; }
  function clampv(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function pad2(n) { return (n < 10 ? '0' : '') + n; }
  function clock(t) {
    t = Math.max(0, t);
    var h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
    var frac = Math.floor((t - Math.floor(t)) * 10);
    return (h > 0 ? h + ':' + pad2(m) : pad2(m)) + ':' + pad2(s) + '.' + frac;
  }
  function shortClock(t) {
    t = Math.max(0, t);
    var h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
    return (h > 0 ? h + ':' + pad2(m) : pad2(m)) + ':' + pad2(s);
  }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }

  // ------------------------------------------------------------ state
  var S = {
    v: null, key: '', rev: '', w0: 0, w1: 0, src: '',
    pieces: [], hist: [], fut: [], active: -1, edge: null,
    pps: 60, zoom: 1, snap: true, mode: 'cut', rate: 1,
    dirty: false, sent: false, playT: 0, raf: 0, drag: null, toastTimer: 0
  };

  // ------------------------------------------------------------ layout
  root.innerHTML = '';
  var msg = el('div', 'tcl-msg', '');
  var main = el('div', 'tcl-main');
  root.appendChild(msg);
  root.appendChild(main);

  var videoBox = el('div', 'tcl-videobox');
  var video = el('video', 'tcl-video');
  video.setAttribute('playsinline', '');
  video.setAttribute('preload', 'auto');
  video.controls = false;
  var noVideo = el('div', 'tcl-novideo', 'ما فيه معاينة فيديو لهذي اللقطة، تقدر تعدل القص من الموجة والكلمات');
  var clockEl = el('div', 'tcl-clock', '00:00.0');
  var modeBadge = el('div', 'tcl-modebadge', '');
  var toast = el('div', 'tcl-toast', '');
  videoBox.appendChild(video);
  videoBox.appendChild(noVideo);
  videoBox.appendChild(clockEl);
  videoBox.appendChild(modeBadge);
  main.appendChild(videoBox);
  main.appendChild(toast);

  var bar = el('div', 'tcl-bar');
  function btn(label, title, fn, cls) {
    var b = el('button', 'tcl-btn' + (cls ? ' ' + cls : ''), label);
    b.type = 'button';
    b.title = title;
    b.addEventListener('click', function (ev) { ev.preventDefault(); fn(ev); root.focus({ preventScroll: true }); });
    bar.appendChild(b);
    return b;
  }
  function sep() { bar.appendChild(el('span', 'tcl-sep')); }
  var bPlay = btn('▶️ تشغيل القص', 'تشغيل القطع المحفوظة فقط مع تخطي المحذوف (Space)', function () { togglePlay('cut'); });
  var bRaw = btn('🎞️ تشغيل كامل', 'تشغيل المعاينة كاملة بدون تخطي', function () { togglePlay('raw'); });
  var bHome = btn('⏮️ البداية', 'الرجوع لبداية أول قطعة (Home)', function () { goHome(); });
  sep();
  var bSplit = btn('✂️ قص عند المؤشر', 'تقسيم القطعة عند المؤشر (S)', function () { splitAtPlayhead(); });
  var bDel = btn('🗑️ حذف القطعة', 'حذف القطعة المحددة (Delete)', function () { deleteActive(); });
  var bToggle = btn('🔁 تشغيل/إيقاف القطعة', 'استبعاد القطعة المحددة أو رجوعها', function () { toggleActive(); });
  sep();
  var bUndo = btn('↩️ تراجع', 'تراجع (Ctrl+Z)', function () { undo(); });
  var bRedo = btn('↪️ إعادة', 'إعادة (Ctrl+Shift+Z)', function () { redo(); });
  var bReset = btn('🔄 رجوع للأصل', 'إرجاع قص الذكاء الاصطناعي الأصلي', function () { resetOriginal(); });
  sep();
  var bSnap = btn('🧲 لصق على الكلمات', 'حواف القطع تلتصق بحواف الكلمات (اضغط Alt أثناء السحب للتعطيل)', function () {
    S.snap = !S.snap; bSnap.classList.toggle('tcl-on', S.snap); showToast(S.snap ? 'اللصق على الكلمات شغال' : 'اللصق على الكلمات متوقف');
  }, 'tcl-on');
  var bZoomIn = btn('🔍+', 'تكبير (Ctrl + عجلة الماوس)', function () { setZoom(S.zoom * 1.5); });
  var bZoomOut = btn('🔍−', 'تصغير', function () { setZoom(S.zoom / 1.5); });
  sep();
  var bApprove = btn('✅ اعتماد التعديل (جاهز للرندر)', 'إرسال القص للاستوديو وحفظه (Ctrl+Enter)', function () { commit(); }, 'tcl-approve');
  main.appendChild(bar);

  var status = el('div', 'tcl-status');
  var stNet = el('span', '', '');
  var stCount = el('span', '', '');
  var stSel = el('span', '', '');
  var stState = el('span', 'tcl-state tcl-clean', '');
  status.appendChild(stNet); status.appendChild(stCount); status.appendChild(stSel); status.appendChild(stState);
  main.appendChild(status);

  var scroll = el('div', 'tcl-scroll');
  var track = el('div', 'tcl-track');
  var canvas = el('canvas', 'tcl-canvas');
  var layer = el('div', 'tcl-layer');
  var playhead = el('div', 'tcl-playhead');
  var ghost = el('div', 'tcl-ghost');
  ghost.style.display = 'none';
  track.appendChild(canvas);
  track.appendChild(layer);
  track.appendChild(ghost);
  track.appendChild(playhead);
  scroll.appendChild(track);
  main.appendChild(scroll);

  var help = el('div', 'tcl-help');
  help.innerHTML = '<kbd>Space</kbd> تشغيل/إيقاف القص · <kbd>J</kbd> رجوع ثانية · <kbd>K</kbd> إيقاف · <kbd>L</kbd> تشغيل وتسريع · ' +
    '<kbd>S</kbd> قص عند المؤشر · <kbd>Delete</kbd> حذف القطعة · <kbd>←</kbd> <kbd>→</kbd> تحريك الحافة المحددة أو المؤشر 0.1ث (مع <kbd>Shift</kbd> 0.5ث) · ' +
    '<kbd>Ctrl+Z</kbd> تراجع · <kbd>Ctrl+Enter</kbd> اعتماد. ' +
    'اضغط على القطعة لتحديدها ومرة ثانية لإيقافها، اسحب حوافها للتعديل، واسحب في مكان فاضي لإضافة قطعة جديدة. ' +
    'الشريط البنفسجي 👀 لحظات شاشة محمية، والبرتقالي ضحك.';
  main.appendChild(help);

  var ctx2d = canvas.getContext('2d');

  // ------------------------------------------------------------ geometry
  var RULER_H = 22, WAVE_TOP = 24, WAVE_H = 80, LAUGH_TOP = 106, LAUGH_H = 10, WORD_TOP = 124, WORD_H = 26, TRACK_H = 170;
  function dur() { return Math.max(0.5, S.w1 - S.w0); }
  function fitPps() {
    var w = scroll.clientWidth || root.clientWidth || 800;
    return Math.max(12, (w - 4) / dur());
  }
  function pps() { return S.pps * S.zoom; }
  function tToX(t) { return (t - S.w0) * pps(); }
  function xToT(x) { return S.w0 + x / pps(); }
  function frameRound(t) {
    var fps = S.v && S.v.fps;
    if (!fps) { return r3(t); }
    return r3(Math.round(t * fps) / fps);
  }

  // ------------------------------------------------------------ messages
  function showToast(text) {
    toast.textContent = text;
    toast.classList.add('tcl-show');
    clearTimeout(S.toastTimer);
    S.toastTimer = setTimeout(function () { toast.classList.remove('tcl-show'); }, 1600);
  }

  // ------------------------------------------------------------ load value
  function readPieces(v) {
    var out = [];
    var src = (v.all_pieces && v.all_pieces.length) ? v.all_pieces : (v.pieces || []);
    for (var i = 0; i < src.length; i++) {
      var p = src[i];
      if (!p || p.length < 2) { continue; }
      var s = Number(p[0]), e = Number(p[1]);
      if (!isFinite(s) || !isFinite(e) || e <= s) { continue; }
      out.push({ s: s, e: e, on: p.length < 3 ? true : !!p[2] });
    }
    out.sort(function (a, b) { return a.s - b.s; });
    return out;
  }

  function load(v, keepEdits) {
    if (!v || typeof v !== 'object' || !v.window || !(Number(v.window[1]) > Number(v.window[0]))) {
      stopPlay();
      S.v = null; S.key = v && v.key ? v.key : '';
      main.style.display = 'none';
      msg.style.display = 'block';
      msg.textContent = (v && v.message) ? v.message : 'اختر لقطة عشان يظهر محرر القص';
      return;
    }
    var sameClip = S.v && S.key === v.key;
    S.v = v;
    S.key = v.key || '';
    S.rev = v.rev;
    S.w0 = Number(v.window[0]);
    S.w1 = Number(v.window[1]);
    if (!(sameClip && keepEdits)) {
      S.pieces = readPieces(v);
      S.hist = []; S.fut = [];
      S.active = -1; S.edge = null;
      S.dirty = !!v.dirty;
      S.sent = !!v.committed;
    }
    msg.style.display = 'none';
    main.style.display = 'block';
    var src = v.src || '';
    if (src !== S.src) {
      stopPlay();
      S.src = src;
      if (src) {
        video.src = src;
        video.style.display = 'block';
        noVideo.style.display = 'none';
        video.load();
      } else {
        video.removeAttribute('src');
        video.style.display = 'none';
        noVideo.style.display = 'flex';
      }
    }
    if (!sameClip) {
      S.zoom = 1;
      S.pps = fitPps();
      var first = S.pieces.length ? S.pieces[0].s : S.w0;
      setPlay(first, true);
      scroll.scrollLeft = 0;
    }
    layout();
  }

  // ------------------------------------------------------------ history and output
  function snapshot() { return S.pieces.map(function (p) { return { s: p.s, e: p.e, on: p.on }; }); }
  function pushHist() {
    S.hist.push(snapshot());
    if (S.hist.length > 200) { S.hist.shift(); }
    S.fut = [];
  }
  function changed(note) {
    S.dirty = true;
    S.sent = false;
    normalize();
    layout();
    publish(false);
    if (note) { showToast(note); }
  }
  function normalize() {
    S.pieces.sort(function (a, b) { return a.s - b.s; });
    var out = [];
    for (var i = 0; i < S.pieces.length; i++) {
      var p = S.pieces[i];
      p.s = clampv(p.s, S.w0, S.w1); p.e = clampv(p.e, S.w0, S.w1);
      if (p.e - p.s < 0.05) { continue; }
      var last = out[out.length - 1];
      if (last && p.s < last.e - 0.001) {
        if (last.on === p.on) { last.e = Math.max(last.e, p.e); continue; }
        p.s = last.e;
        if (p.e - p.s < 0.05) { continue; }
      }
      out.push(p);
    }
    S.pieces = out;
    if (S.active >= S.pieces.length) { S.active = S.pieces.length - 1; }
  }
  function enabledPieces() {
    return S.pieces.filter(function (p) { return p.on; }).map(function (p) { return [r3(p.s), r3(p.e)]; });
  }
  function payload(committed) {
    var base = S.v ? clone(S.v) : {};
    base.pieces = enabledPieces();
    base.all_pieces = S.pieces.map(function (p) { return [r3(p.s), r3(p.e), p.on ? 1 : 0]; });
    base.committed = !!committed;
    base.dirty = S.dirty;
    base.rev = S.rev;
    base.key = S.key;
    base.edited_at = Date.now();
    return base;
  }
  function writeBridge(obj) {
    try {
      var id = (props && props.tl_elem) ? String(props.tl_elem) : '';
      if (!id) { return; }
      var host = document.getElementById(id + '_bridge');
      if (!host) { return; }
      var box = host.querySelector('textarea, input');
      if (!box) { return; }
      box.value = JSON.stringify(obj);
      box.dispatchEvent(new Event('input', { bubbles: true }));
    } catch (e) { }
  }
  function publish(committed) {
    if (!S.v) { return; }
    var obj = payload(committed);
    S.v = obj;
    try { props.value = obj; } catch (e) { }
    if (committed) { writeBridge(obj); }
  }
  function commit() {
    if (!S.v) { return; }
    var kept = enabledPieces();
    if (!kept.length) { showToast('⚠️ لازم تخلي قطعة وحدة على الأقل'); return; }
    S.dirty = false;
    S.sent = true;
    publish(true);
    try { trigger('commit', { rev: S.rev, count: kept.length }); } catch (e) { }
    layout();
    showToast('✅ اعتمدنا القص، الرندر يستخدم هذا القص');
  }

  // ------------------------------------------------------------ edits
  function undo() {
    if (!S.hist.length) { showToast('ما فيه شي نتراجع عنه'); return; }
    S.fut.push(snapshot());
    S.pieces = S.hist.pop();
    S.active = -1; S.edge = null;
    changed('↩️ تراجعنا');
  }
  function redo() {
    if (!S.fut.length) { return; }
    S.hist.push(snapshot());
    S.pieces = S.fut.pop();
    S.active = -1; S.edge = null;
    changed('↪️ رجعنا التعديل');
  }
  function resetOriginal() {
    if (!S.v) { return; }
    pushHist();
    S.pieces = (S.v.orig_pieces || []).map(function (p) { return { s: Number(p[0]), e: Number(p[1]), on: true }; });
    S.active = -1; S.edge = null;
    changed('🔄 رجعنا للقص الأصلي');
  }
  function pieceAt(t) {
    for (var i = 0; i < S.pieces.length; i++) {
      if (S.pieces[i].s <= t && t < S.pieces[i].e) { return i; }
    }
    return -1;
  }
  function splitAtPlayhead() {
    var t = S.playT;
    var i = (S.active >= 0 && S.pieces[S.active] && S.pieces[S.active].s < t && t < S.pieces[S.active].e) ? S.active : pieceAt(t);
    if (i < 0) { showToast('حط المؤشر داخل قطعة عشان تقصها'); return; }
    var p = S.pieces[i];
    var cut = frameRound(t);
    if (cut - p.s < 0.1 || p.e - cut < 0.1) { showToast('المؤشر قريب مرة من حافة القطعة'); return; }
    pushHist();
    S.pieces.splice(i + 1, 0, { s: cut, e: p.e, on: p.on });
    p.e = cut;
    S.active = i + 1; S.edge = null;
    changed('✂️ قسمنا القطعة');
  }
  function deleteActive() {
    if (S.active < 0 || !S.pieces[S.active]) { showToast('اختر قطعة أول'); return; }
    pushHist();
    S.pieces.splice(S.active, 1);
    S.active = -1; S.edge = null;
    changed('🗑️ حذفنا القطعة');
  }
  function toggleActive() {
    if (S.active < 0 || !S.pieces[S.active]) { showToast('اختر قطعة أول'); return; }
    pushHist();
    S.pieces[S.active].on = !S.pieces[S.active].on;
    changed(S.pieces[S.active].on ? 'رجعنا القطعة' : 'استبعدنا القطعة من القص');
  }
  function nudge(delta) {
    if (S.active >= 0 && S.edge && S.pieces[S.active]) {
      var p = S.pieces[S.active];
      var prev = S.pieces[S.active - 1], next = S.pieces[S.active + 1];
      pushHist();
      if (S.edge === 's') {
        p.s = frameRound(clampv(p.s + delta, prev ? prev.e : S.w0, p.e - 0.1));
      } else {
        p.e = frameRound(clampv(p.e + delta, p.s + 0.1, next ? next.s : S.w1));
      }
      changed();
      setPlay(S.edge === 's' ? p.s : Math.max(p.s, p.e - 0.4), true);
      return;
    }
    setPlay(S.playT + delta, true);
  }

  // ------------------------------------------------------------ zoom
  function setZoom(z) {
    var center = S.playT;
    S.zoom = clampv(z, 1, 24);
    layout();
    scroll.scrollLeft = Math.max(0, tToX(center) - scroll.clientWidth / 2);
    draw();
  }

  // ------------------------------------------------------------ playback (virtual jump cut)
  function hasVideo() { return !!S.src && video.readyState >= 1; }
  function setPlay(t, seek) {
    S.playT = clampv(t, S.w0, S.w1);
    if (seek && S.src) {
      try { video.currentTime = Math.max(0, S.playT - S.w0); } catch (e) { }
    }
    placePlayhead();
  }
  function placePlayhead() {
    playhead.style.left = Math.round(tToX(S.playT)) + 'px';
    clockEl.textContent = clock(S.playT);
    if (!video.paused) {
      var x = tToX(S.playT), left = scroll.scrollLeft, w = scroll.clientWidth;
      if (x < left + 30 || x > left + w - 60) { scroll.scrollLeft = Math.max(0, x - w * 0.3); }
    }
  }
  function nextOn(t) {
    for (var i = 0; i < S.pieces.length; i++) {
      var p = S.pieces[i];
      if (p.on && p.e > t + 0.02) { return i; }
    }
    return -1;
  }
  function firstOn() { return nextOn(S.w0 - 1); }
  function stopPlay() {
    try { video.pause(); } catch (e) { }
    if (S.raf) { cancelAnimationFrame(S.raf); S.raf = 0; }
    updateButtons();
  }
  function tick() {
    S.raf = 0;
    if (video.paused) { updateButtons(); return; }
    var t = S.w0 + video.currentTime;
    if (S.mode === 'cut') {
      var i = nextOn(t);
      if (i < 0) {
        video.pause();
        var f = firstOn();
        setPlay(f >= 0 ? S.pieces[f].s : S.w0, true);
        showToast('⏹️ خلص القص');
        updateButtons();
        return;
      }
      var p = S.pieces[i];
      if (t < p.s - 0.02) {
        video.currentTime = Math.max(0, p.s - S.w0);
        t = p.s;
      } else if (t >= p.e - 0.03) {
        var j = nextOn(p.e);
        if (j < 0 || j === i) {
          video.pause();
          var f2 = firstOn();
          setPlay(f2 >= 0 ? S.pieces[f2].s : S.w0, true);
          showToast('⏹️ خلص القص');
          updateButtons();
          return;
        }
        video.currentTime = Math.max(0, S.pieces[j].s - S.w0);
        t = S.pieces[j].s;
      }
    }
    if (S.w0 + video.currentTime >= S.w1 - 0.02) { video.pause(); }
    S.playT = clampv(t, S.w0, S.w1);
    placePlayhead();
    S.raf = requestAnimationFrame(tick);
  }
  function play(mode) {
    if (!S.src) { showToast('ما فيه فيديو معاينة لهذي اللقطة'); return; }
    S.mode = mode;
    if (mode === 'cut') {
      var i = nextOn(S.playT - 0.001);
      if (i < 0) { i = firstOn(); }
      if (i < 0) { showToast('ما فيه قطع شغالة'); return; }
      if (S.playT < S.pieces[i].s || S.playT >= S.pieces[i].e) { setPlay(S.pieces[i].s, true); }
    }
    video.playbackRate = S.rate;
    var pr = video.play();
    if (pr && pr.catch) { pr.catch(function () { showToast('اضغط على الفيديو مرة عشان المتصفح يسمح بالتشغيل'); }); }
    if (!S.raf) { S.raf = requestAnimationFrame(tick); }
    updateButtons();
  }
  function togglePlay(mode) {
    if (!video.paused && S.mode === mode) { stopPlay(); return; }
    play(mode);
  }
  function goHome() {
    var f = firstOn();
    setPlay(f >= 0 ? S.pieces[f].s : S.w0, true);
    scroll.scrollLeft = Math.max(0, tToX(S.playT) - 40);
  }
  video.addEventListener('play', function () { if (!S.raf) { S.raf = requestAnimationFrame(tick); } updateButtons(); });
  video.addEventListener('pause', updateButtons);
  video.addEventListener('loadedmetadata', function () { setPlay(S.playT, true); });
  video.addEventListener('click', function () { togglePlay('cut'); });
  video.addEventListener('error', function () {
    if (S.src) { noVideo.style.display = 'flex'; noVideo.textContent = 'تعذر تحميل فيديو المعاينة. تأكد إن مجلد Drive مسموح في الاستوديو'; }
  });

  function updateButtons() {
    var playing = !video.paused;
    bPlay.textContent = (playing && S.mode === 'cut') ? '⏸️ إيقاف' : '▶️ تشغيل القص';
    bRaw.textContent = (playing && S.mode === 'raw') ? '⏸️ إيقاف' : '🎞️ تشغيل كامل';
    bPlay.classList.toggle('tcl-on', playing && S.mode === 'cut');
    bRaw.classList.toggle('tcl-on', playing && S.mode === 'raw');
    modeBadge.textContent = playing ? (S.mode === 'cut' ? '✂️ تشغيل القص' + (S.rate !== 1 ? ' ×' + S.rate : '') : '🎞️ تشغيل كامل') : '';
    modeBadge.style.display = playing ? 'block' : 'none';
    var hasSel = S.active >= 0 && !!S.pieces[S.active];
    bDel.disabled = !hasSel;
    bToggle.disabled = !hasSel;
    bUndo.disabled = !S.hist.length;
    bRedo.disabled = !S.fut.length;
    bPlay.disabled = !S.src;
    bRaw.disabled = !S.src;
  }

  // ------------------------------------------------------------ drawing
  function draw() {
    if (!S.v) { return; }
    var dpr = window.devicePixelRatio || 1;
    var vw = Math.max(50, scroll.clientWidth);
    var left = scroll.scrollLeft;
    canvas.style.left = left + 'px';
    canvas.style.width = vw + 'px';
    canvas.width = Math.floor(vw * dpr);
    canvas.height = Math.floor(TRACK_H * dpr);
    var c = ctx2d;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, vw, TRACK_H);
    var t0 = xToT(left), t1 = xToT(left + vw);
    function X(t) { return tToX(t) - left; }

    // removed areas dim (outside enabled pieces)
    c.fillStyle = 'rgba(255, 77, 109, 0.06)';
    c.fillRect(0, WAVE_TOP, vw, LAUGH_TOP + LAUGH_H - WAVE_TOP);

    // protected beats
    var beats = S.v.beats || [];
    for (var b = 0; b < beats.length; b++) {
      var bs = beats[b][0], be = beats[b][1];
      if (be < t0 || bs > t1) { continue; }
      var bx = X(bs), bw = Math.max(2, X(be) - bx);
      c.fillStyle = 'rgba(155, 93, 229, 0.22)';
      c.fillRect(bx, RULER_H, bw, TRACK_H - RULER_H);
      c.fillStyle = 'rgba(190, 140, 255, 0.95)';
      c.fillRect(bx, RULER_H, bw, 3);
      c.font = '11px system-ui, sans-serif';
      c.fillStyle = '#d9c2ff';
      c.textBaseline = 'top';
      c.textBaseline = 'bottom';
      c.fillText('👀 ' + (beats[b][2] || ''), bx + 3, LAUGH_TOP - 2);
    }

    // waveform
    var peaks = S.v.peaks || [];
    if (peaks.length) {
      var n = peaks.length, span = dur();
      var mid = WAVE_TOP + WAVE_H / 2;
      c.fillStyle = 'rgba(120, 170, 255, 0.85)';
      var step = Math.max(1, Math.floor(n * (1 / pps()) / span * 1));
      var i0 = Math.max(0, Math.floor((t0 - S.w0) / span * n) - 1), i1 = Math.min(n, Math.ceil((t1 - S.w0) / span * n) + 1);
      var barW = Math.max(1, (span / n) * pps() - 0.5);
      for (var i = i0; i < i1; i += step) {
        var t = S.w0 + span * i / n;
        var h = Math.max(1, peaks[i] * (WAVE_H / 2 - 2));
        c.fillRect(X(t), mid - h, barW * step, h * 2);
      }
      c.fillStyle = 'rgba(120, 170, 255, 0.25)';
      c.fillRect(0, mid, vw, 1);
    } else {
      c.fillStyle = '#5c6075';
      c.font = '12px system-ui, sans-serif';
      c.textBaseline = 'middle';
      c.fillText('(ما فيه موجة صوت)', 8, WAVE_TOP + WAVE_H / 2);
    }

    // laugh track
    var laugh = S.v.laugh || [], spans = S.v.laugh_spans || [];
    if (laugh.length) {
      var ln = laugh.length, lspan = dur();
      for (var k = 0; k < ln; k++) {
        var v = laugh[k];
        if (v < 0.15) { continue; }
        var lt0 = S.w0 + lspan * k / ln, lt1 = S.w0 + lspan * (k + 1) / ln;
        if (lt1 < t0 || lt0 > t1) { continue; }
        c.fillStyle = 'rgba(255, 159, 28, ' + (0.25 + 0.7 * v).toFixed(2) + ')';
        c.fillRect(X(lt0), LAUGH_TOP, Math.max(1, X(lt1) - X(lt0)), LAUGH_H);
      }
    }
    for (var q = 0; q < spans.length; q++) {
      var sp = spans[q];
      if (sp[1] < t0 || sp[0] > t1) { continue; }
      c.fillStyle = 'rgba(255, 159, 28, ' + (0.3 + 0.65 * (sp[3] === undefined ? 1 : sp[3])).toFixed(2) + ')';
      c.fillRect(X(sp[0]), LAUGH_TOP, Math.max(2, X(sp[1]) - X(sp[0])), LAUGH_H);
    }

    // word chips
    var words = S.v.words || [];
    c.font = '12px system-ui, "Segoe UI", Tahoma, sans-serif';
    c.textBaseline = 'middle';
    for (var w = 0; w < words.length; w++) {
      var ws = words[w][1], we = words[w][2];
      if (we < t0 || ws > t1) { continue; }
      var wx = X(ws), ww = Math.max(2, X(we) - wx);
      var inside = false;
      for (var pi = 0; pi < S.pieces.length; pi++) {
        var pp = S.pieces[pi];
        if (pp.on && pp.s <= (ws + we) / 2 && (ws + we) / 2 < pp.e) { inside = true; break; }
      }
      c.fillStyle = inside ? 'rgba(64, 196, 128, 0.30)' : 'rgba(110, 112, 128, 0.22)';
      c.fillRect(wx, WORD_TOP, ww, WORD_H);
      c.fillStyle = inside ? 'rgba(64, 196, 128, 0.8)' : 'rgba(120, 122, 140, 0.6)';
      c.fillRect(wx, WORD_TOP, 1, WORD_H);
      if (ww >= 16) {
        c.save();
        c.beginPath();
        c.rect(wx + 2, WORD_TOP, ww - 4, WORD_H);
        c.clip();
        c.fillStyle = inside ? '#e9fff3' : '#8a8ea3';
        c.direction = 'rtl';
        c.textAlign = 'right';
        c.fillText(words[w][0], wx + ww - 3, WORD_TOP + WORD_H / 2);
        c.restore();
        c.direction = 'ltr';
        c.textAlign = 'left';
      }
    }

    // ruler
    c.fillStyle = '#1c1d25';
    c.fillRect(0, 0, vw, RULER_H);
    var pxs = pps();
    var steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60];
    var major = 60;
    for (var si = 0; si < steps.length; si++) { if (steps[si] * pxs >= 70) { major = steps[si]; break; } }
    var minor = major / 5;
    var kStart = Math.floor(t0 / minor), kEnd = Math.ceil(t1 / minor) + 1;
    c.strokeStyle = '#4a4d60';
    c.fillStyle = '#c9ccda';
    c.font = '11px ui-monospace, Menlo, Consolas, monospace';
    c.textBaseline = 'top';
    c.textAlign = 'left';
    for (var kk = kStart; kk <= kEnd; kk++) {
      var tt = kk * minor;
      var xx = Math.round(X(tt)) + 0.5;
      var isMajor = (kk % 5) === 0;
      c.beginPath();
      c.moveTo(xx, isMajor ? 8 : 15);
      c.lineTo(xx, RULER_H);
      c.stroke();
      if (isMajor) {
        var label = major < 1 ? clock(tt) : shortClock(tt);
        c.fillText(label, xx + 3, 2);
      }
    }
    c.fillStyle = '#2c2e3a';
    c.fillRect(0, RULER_H - 1, vw, 1);
  }

  // ------------------------------------------------------------ pieces DOM
  function layout() {
    if (!S.v) { return; }
    var total = Math.max(scroll.clientWidth, Math.ceil(dur() * pps()) + 2);
    track.style.width = total + 'px';
    layer.innerHTML = '';
    for (var i = 0; i < S.pieces.length; i++) {
      var p = S.pieces[i];
      var d = el('div', 'tcl-piece' + (p.on ? '' : ' tcl-off') + (i === S.active ? ' tcl-active' : ''));
      d.style.left = tToX(p.s) + 'px';
      d.style.width = Math.max(4, tToX(p.e) - tToX(p.s)) + 'px';
      d.dataset.i = String(i);
      var hl = el('div', 'tcl-h tcl-l' + (i === S.active && S.edge === 's' ? ' tcl-edge' : ''));
      var hr = el('div', 'tcl-h tcl-r' + (i === S.active && S.edge === 'e' ? ' tcl-edge' : ''));
      hl.dataset.edge = 's';
      hr.dataset.edge = 'e';
      d.appendChild(hl);
      d.appendChild(hr);
      if (tToX(p.e) - tToX(p.s) > 46) {
        d.appendChild(el('div', 'tcl-num', '#' + (i + 1) + ' ' + (p.e - p.s).toFixed(1) + 'ث'));
      }
      layer.appendChild(d);
    }
    placePlayhead();
    draw();
    var kept = S.pieces.filter(function (x) { return x.on; });
    var net = kept.reduce(function (a, x) { return a + (x.e - x.s); }, 0);
    stNet.innerHTML = 'المدة بعد القص <b>' + net.toFixed(1) + 'ث</b>';
    stCount.innerHTML = 'القطع <b>' + kept.length + '</b>' + (kept.length !== S.pieces.length ? ' (+' + (S.pieces.length - kept.length) + ' موقفة)' : '');
    if (S.active >= 0 && S.pieces[S.active]) {
      var ap = S.pieces[S.active];
      stSel.innerHTML = 'المحددة: <b>#' + (S.active + 1) + '</b> ' + shortClock(ap.s) + ' إلى ' + shortClock(ap.e) + (S.edge ? (S.edge === 's' ? ' (الحافة الأولى)' : ' (الحافة الأخيرة)') : '');
    } else {
      stSel.textContent = '';
    }
    stState.className = 'tcl-state ' + (S.dirty ? 'tcl-dirtyst' : (S.sent ? 'tcl-sent' : 'tcl-clean'));
    stState.textContent = S.dirty ? '✏️ فيه تعديلات ما اعتمدتها' : (S.sent ? '📨 القص معتمد' : '✔️ بدون تعديلات');
    bApprove.classList.toggle('tcl-dirty', S.dirty);
    updateButtons();
  }

  // ------------------------------------------------------------ snapping
  function snapT(t, alt, ignore) {
    var out = t;
    if (S.snap && !alt && S.v) {
      var best = null, bestD = 9 / pps();
      var words = S.v.words || [];
      for (var i = 0; i < words.length; i++) {
        var a = words[i][1], b = words[i][2];
        var da = Math.abs(a - t), db = Math.abs(b - t);
        if (da < bestD) { bestD = da; best = a; }
        if (db < bestD) { bestD = db; best = b; }
      }
      for (var j = 0; j < S.pieces.length; j++) {
        if (j === ignore) { continue; }
        var ps = S.pieces[j].s, pe = S.pieces[j].e;
        if (Math.abs(ps - t) < bestD) { bestD = Math.abs(ps - t); best = ps; }
        if (Math.abs(pe - t) < bestD) { bestD = Math.abs(pe - t); best = pe; }
      }
      if (best !== null) { out = best; }
    }
    return frameRound(out);
  }

  // ------------------------------------------------------------ pointer interactions
  function localX(ev) {
    var rect = track.getBoundingClientRect();
    return ev.clientX - rect.left;
  }
  track.addEventListener('pointerdown', function (ev) {
    if (!S.v || ev.button !== 0) { return; }
    root.focus({ preventScroll: true });
    var target = ev.target;
    var pieceEl = target.closest ? target.closest('.tcl-piece') : null;
    var x = localX(ev);
    var t = xToT(x);
    if (pieceEl) {
      var i = Number(pieceEl.dataset.i);
      var edge = target.dataset ? target.dataset.edge : null;
      S.drag = { kind: edge ? 'edge' : 'piece', i: i, edge: edge || null, x0: x, moved: false, before: snapshot() };
    } else {
      S.drag = { kind: 'empty', x0: x, t0: t, moved: false, before: snapshot() };
    }
    try { track.setPointerCapture(ev.pointerId); } catch (e) { }
    ev.preventDefault();
  });
  track.addEventListener('pointermove', function (ev) {
    var dg = S.drag;
    if (!dg) { return; }
    var x = localX(ev);
    if (!dg.moved && Math.abs(x - dg.x0) < 4) { return; }
    dg.moved = true;
    var t = clampv(xToT(x), S.w0, S.w1);
    if (dg.kind === 'edge') {
      var p = S.pieces[dg.i];
      if (!p) { return; }
      var prev = S.pieces[dg.i - 1], next = S.pieces[dg.i + 1];
      var st = snapT(t, ev.altKey, dg.i);
      if (dg.edge === 's') { p.s = clampv(st, prev ? prev.e : S.w0, p.e - 0.1); }
      else { p.e = clampv(st, p.s + 0.1, next ? next.s : S.w1); }
      S.active = dg.i; S.edge = dg.edge;
      layout();
      setPlay(dg.edge === 's' ? p.s : p.e, true);
    } else if (dg.kind === 'empty') {
      var a = snapT(Math.min(dg.t0, t), ev.altKey, -1), b = snapT(Math.max(dg.t0, t), ev.altKey, -1);
      var lo = S.w0, hi = S.w1;
      for (var k = 0; k < S.pieces.length; k++) {
        var q = S.pieces[k];
        if (q.e <= dg.t0) { lo = Math.max(lo, q.e); }
        if (q.s >= dg.t0) { hi = Math.min(hi, q.s); }
      }
      a = clampv(a, lo, hi); b = clampv(b, lo, hi);
      dg.a = a; dg.b = b;
      ghost.style.display = 'block';
      ghost.style.left = tToX(a) + 'px';
      ghost.style.width = Math.max(2, tToX(b) - tToX(a)) + 'px';
    }
  });
  function endDrag(ev) {
    var dg = S.drag;
    S.drag = null;
    ghost.style.display = 'none';
    try { track.releasePointerCapture(ev.pointerId); } catch (e) { }
    if (!dg) { return; }
    if (!dg.moved) {
      var t = clampv(xToT(localX(ev)), S.w0, S.w1);
      if (dg.kind === 'piece' || dg.kind === 'edge') {
        if (dg.kind === 'piece' && S.active === dg.i && !S.edge) {
          S.hist.push(dg.before); S.fut = [];
          S.pieces[dg.i].on = !S.pieces[dg.i].on;
          changed(S.pieces[dg.i].on ? 'رجعنا القطعة' : 'استبعدنا القطعة من القص');
        } else {
          S.active = dg.i;
          S.edge = dg.kind === 'edge' ? dg.edge : null;
          layout();
        }
        setPlay(dg.kind === 'edge' ? (dg.edge === 's' ? S.pieces[dg.i].s : S.pieces[dg.i].e) : t, true);
      } else {
        S.active = -1; S.edge = null;
        setPlay(t, true);
        layout();
      }
      return;
    }
    if (dg.kind === 'edge') {
      S.hist.push(dg.before); S.fut = [];
      changed();
    } else if (dg.kind === 'empty' && dg.b - dg.a >= 0.15) {
      S.hist.push(dg.before); S.fut = [];
      S.pieces.push({ s: dg.a, e: dg.b, on: true });
      normalize();
      for (var i = 0; i < S.pieces.length; i++) { if (Math.abs(S.pieces[i].s - dg.a) < 0.002) { S.active = i; } }
      S.edge = null;
      changed('➕ أضفنا قطعة');
    }
  }
  track.addEventListener('pointerup', endDrag);
  track.addEventListener('pointercancel', endDrag);

  scroll.addEventListener('scroll', function () { draw(); });
  scroll.addEventListener('wheel', function (ev) {
    if (!(ev.ctrlKey || ev.metaKey)) { return; }
    ev.preventDefault();
    setZoom(S.zoom * (ev.deltaY < 0 ? 1.25 : 0.8));
  }, { passive: false });

  // ------------------------------------------------------------ keyboard
  root.addEventListener('keydown', function (ev) {
    var tag = (ev.target && ev.target.tagName) ? ev.target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select') { return; }
    if (!S.v) { return; }
    var k = ev.key, code = ev.code || '', low = (k || '').toLowerCase(), ctrl = ev.ctrlKey || ev.metaKey;
    var handled = true;
    if (ctrl && (k === 'Enter' || code === 'Enter' || code === 'NumpadEnter')) { commit(); }
    else if (ctrl && (code === 'KeyZ' || low === 'z') && ev.shiftKey) { redo(); }
    else if (ctrl && (code === 'KeyZ' || low === 'z')) { undo(); }
    else if (ctrl && (code === 'KeyY' || low === 'y')) { redo(); }
    else if (ctrl) { handled = false; }
    else if (k === ' ' || code === 'Space') { togglePlay('cut'); }
    else if (code === 'KeyS' || low === 's' || low === 'س') { splitAtPlayhead(); }
    else if (k === 'Delete' || k === 'Backspace') { deleteActive(); }
    else if (k === 'ArrowLeft') { nudge(ev.shiftKey ? -0.5 : -0.1); }
    else if (k === 'ArrowRight') { nudge(ev.shiftKey ? 0.5 : 0.1); }
    else if (k === 'Home') { goHome(); }
    else if (k === 'Escape') { S.active = -1; S.edge = null; layout(); }
    else if (ev.code === 'KeyK' || low === 'k' || low === 'ن') { stopPlay(); S.rate = 1; video.playbackRate = 1; updateButtons(); }
    else if (ev.code === 'KeyL' || low === 'l' || low === 'م') {
      if (!video.paused && S.mode === 'cut') { S.rate = S.rate >= 2 ? 1 : (S.rate >= 1.5 ? 2 : 1.5); video.playbackRate = S.rate; showToast('سرعة ×' + S.rate); updateButtons(); }
      else { S.rate = 1; play('cut'); }
    }
    else if (ev.code === 'KeyJ' || low === 'j' || low === 'ت') {
      var back = S.playT - 1.0;
      if (!video.paused && S.mode === 'cut') {
        var i = pieceAt(S.playT);
        if (i >= 0 && back < S.pieces[i].s) {
          var prevOn = -1;
          for (var q = i - 1; q >= 0; q--) { if (S.pieces[q].on) { prevOn = q; break; } }
          back = prevOn >= 0 ? Math.max(S.pieces[prevOn].s, S.pieces[prevOn].e - (S.pieces[i].s - back)) : S.pieces[i].s;
        }
      }
      setPlay(back, true);
    }
    else { handled = false; }
    if (handled) { ev.preventDefault(); ev.stopPropagation(); }
  });

  // ------------------------------------------------------------ sync from Python
  function incoming() {
    var v = props.value;
    if (!v || typeof v !== 'object') {
      if (typeof v === 'string' && v) { try { v = JSON.parse(v); } catch (e) { v = null; } }
    }
    if (!v || typeof v !== 'object') { load({ message: '' }, false); return; }
    if (S.v && v.key === S.key && v.edited_at && S.v.edited_at === v.edited_at) { return; }
    var sameClip = S.v && v.key === S.key && String(v.rev) === String(S.rev);
    if (sameClip && v.edited_at) { return; }
    load(v, false);
  }
  incoming();
  if (typeof watch === 'function') {
    watch('value', incoming);
  } else {
    var lastSeen = '';
    setInterval(function () {
      var v = props.value;
      var sig = v && typeof v === 'object' ? (String(v.key) + '|' + String(v.rev) + '|' + String(v.edited_at || '')) : String(v);
      if (sig !== lastSeen) { lastSeen = sig; incoming(); }
    }, 400);
  }
  if (window.ResizeObserver) {
    var ro = new ResizeObserver(function () {
      if (!S.v) { return; }
      var oldPps = S.pps;
      S.pps = fitPps();
      if (Math.abs(oldPps - S.pps) > 0.5) { layout(); } else { draw(); }
    });
    ro.observe(scroll);
  }
})();
"""


# ============================================================================
# Gradio builders
# ============================================================================

def build_timeline(elem_id: str = "tc_timeline", **html_kwargs: Any) -> Any:
    """The editor component. Needs Gradio 6 (check timeline_supported() first).
    Extra keyword arguments go to gr.HTML (for example visible=False)."""
    import gradio as gr

    params = inspect.signature(gr.HTML.__init__).parameters
    kwargs: dict = {
        "value": html_kwargs.pop("value", empty_timeline_value()),
        "html_template": HTML_TEMPLATE,
        "css_template": CSS_TEMPLATE,
        "js_on_load": JS_ON_LOAD,
        "elem_id": elem_id,
        "tl_elem": elem_id,
    }
    if "apply_default_css" in params:
        kwargs["apply_default_css"] = False
    if "padding" in params:
        kwargs["padding"] = False
    if "container" in params:
        kwargs["container"] = False
    kwargs.update(html_kwargs)
    return gr.HTML(**kwargs)


def build_bridge(elem_id: str = "tc_timeline") -> Any:
    """Optional hidden textbox that receives the same JSON as the commit event. Read it with
    parse_timeline_edit(). It must be rendered (hidden, not removed) for the JS to find it."""
    import gradio as gr

    params = inspect.signature(gr.Textbox.__init__).parameters
    visible: Any = "hidden"
    if "visible" in params:
        ann = str(params["visible"].annotation)
        if "hidden" not in ann and "Literal" not in ann:
            visible = True
    kwargs: dict = {"value": "", "elem_id": f"{elem_id}_bridge", "show_label": False, "container": False,
                    "lines": 1, "max_lines": 1, "interactive": True, "visible": visible}
    if visible is True:
        kwargs["elem_classes"] = ["tcl-bridge-hidden"]
    return gr.Textbox(**kwargs)


# === END OF studio_timeline.py ===
