import datetime
import json
import os
import zipfile
import gradio as gr
from stage3_clipper import call_gemini_text, sec_to_hms

QUICK_TAGS = [
    "ضحك / صراخ رهيب",
    "سالفة حلوة وكاملة السياق",
    "التوتر والخوف مضحك",
    "طقطقة رهيبة مع الشات",
    "القصّ بدأ متأخر أو انقطع بدري",
    "تشتتت وضاعت السالفة",
    "طويلة ومملة / فيها سكوت",
    "كلام عادي ما يستاهل نشر",
]


def _load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _rating_badge(rating: str) -> str:
    if rating == "liked":
        return "👍 أعجبتني"
    if rating == "disliked":
        return "👎 مرفوضة"
    return "⏳ لم تُقيّم"


def _pace_label(wps: float) -> str:
    if wps >= 3.0:
        return f"سريع/متوتر ({wps} كلمة/ث) → تقطيع سريع"
    if wps >= 2.1:
        return f"متوسط ({wps} كلمة/ث) → متوازن"
    if wps > 0:
        return f"هادي/سالفة ({wps} كلمة/ث) → جمل كاملة"
    return "غير محسوب"


def launch_studio(vod_dir: str, drive_root: str, api_key: str):
    manifest_path = os.path.join(vod_dir, "clips_manifest.json")
    mem_dir = os.path.join(drive_root, "memory")
    os.makedirs(mem_dir, exist_ok=True)
    history_path = os.path.join(mem_dir, "feedback_history.json")
    rules_path = os.path.join(mem_dir, "my_editing_rules.md")
    vod_id = os.path.basename(vod_dir.rstrip("/"))

    def get_clips():
        return _load_json(manifest_path, [])

    def get_choices():
        clips = get_clips()
        return [
            f"#{c['id']:02d} | {c['title']} [{_rating_badge(c.get('rating', 'pending'))}]"
            for c in clips
        ]

    def find_clip_by_label(label: str):
        clips = get_clips()
        if not clips:
            return None, None, clips
        if label and label.startswith("#"):
            try:
                cid = int(label.split("|")[0].replace("#", "").strip())
                for idx, c in enumerate(clips):
                    if c["id"] == cid:
                        return idx, c, clips
            except ValueError:
                pass
        return 0, clips[0], clips

    def load_clip_ui(label: str):
        _, clip, _ = find_clip_by_label(label)
        if not clip:
            return None, "⚠️ ما فيه لقطات في هذا البث بعد.", "", "⏳ لم تُقيّم", [], ""

        video_path = clip.get("video_path")
        if not video_path or not os.path.exists(video_path):
            video_path = None

        srt_text = ""
        srt_path = clip.get("srt_path")
        if srt_path and os.path.exists(srt_path):
            with open(srt_path, "r", encoding="utf-8") as f:
                srt_text = f.read()

        info_md = (
            f"### 🎬 {clip['title']}\n"
            f"- **التوقيت في البث:** `{sec_to_hms(clip['start'])}` ← `{sec_to_hms(clip['end'])}` (**المدة:** {clip['duration']} ثانية)\n"
            f"- **التصنيف:** `{clip.get('category', 'منوع')}` | **تقييم الذكاء الاصطناعي:** `{clip.get('virality_score', '-')}/10`\n"
            f"- **إيقاع كلام الستريمر:** {_pace_label(float(clip.get('avg_wps', 0)))}\n"
            f"- **ليش اختارها الذكاء الاصطناعي:** {clip.get('reason', '-')}"
        )

        rating_val = _rating_badge(clip.get("rating", "pending"))
        saved_tags = clip.get("tags", [])
        saved_note = clip.get("note", "")
        return video_path, info_md, srt_text, rating_val, saved_tags, saved_note

    def save_srt_edit(label: str, new_srt: str):
        _, clip, _ = find_clip_by_label(label)
        if not clip:
            return "❌ ما تم اختيار لقطة"
        srt_path = clip.get("srt_path")
        if not srt_path:
            return "❌ مسار ملف السبتايتل غير موجود"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(new_srt)
        return f"✅ تم حفظ تعديلات السبتايتل مباشرة في Drive ({os.path.basename(srt_path)})"

    def save_feedback(label: str, rating_choice: str, selected_tags: list, note_text: str):
        idx, clip, clips = find_clip_by_label(label)
        if clip is None:
            return gr.update(), "❌ ما فيه لقطة مختارة"

        if "أعجبتني" in rating_choice:
            rating_code = "liked"
        elif "مرفوضة" in rating_choice:
            rating_code = "disliked"
        else:
            rating_code = "pending"

        tags_list = list(selected_tags or [])
        combined_note = " | ".join(tags_list + ([note_text.strip()] if note_text.strip() else []))

        clip["rating"] = rating_code
        clip["tags"] = tags_list
        clip["note"] = note_text.strip()
        clips[idx] = clip
        _save_json(manifest_path, clips)

        # تحديث الذاكرة الدائمة على Google Drive
        history = _load_json(history_path, [])
        entry_key = f"{vod_id}_{clip['id']}"
        history = [h for h in history if h.get("key") != entry_key]

        if rating_code in ("liked", "disliked"):
            history.append({
                "key": entry_key,
                "vod_id": vod_id,
                "clip_id": clip["id"],
                "title": clip["title"],
                "category": clip.get("category", ""),
                "duration": clip.get("duration", 0),
                "avg_wps": clip.get("avg_wps", 0),
                "metrics": clip.get("metrics", {}),
                "excerpt": clip.get("excerpt", ""),
                "ai_reason": clip.get("reason", ""),
                "rating": rating_code,
                "note": combined_note or ("أعجبتني" if rating_code == "liked" else "ما تنفع"),
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            })
        _save_json(history_path, history)

        new_choices = get_choices()
        new_label = new_choices[idx]
        liked_cnt = sum(1 for h in history if h["rating"] == "liked")
        disliked_cnt = sum(1 for h in history if h["rating"] == "disliked")
        status = f"✅ تم حفظ تقييمك! (إجمالي ذاكرة التدريب: 👍 {liked_cnt} | 👎 {disliked_cnt})"
        return gr.update(choices=new_choices, value=new_label), status

    def load_rules_text():
        if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def save_rules_manual(text: str):
        with open(rules_path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        return "✅ تم حفظ قواعد الذوق يدوياً في Google Drive! بيطبقها الذكاء الاصطناعي من اللقطة الجاية."

    def train_ai_rules():
        history = _load_json(history_path, [])
        if not history:
            return load_rules_text(), "⚠️ قيّم لقطتين على الأقل أول (👍 أو 👎) عشان يتعلم منها!"

        old_rules = load_rules_text()
        liked = [h for h in history if h["rating"] == "liked"]
        disliked = [h for h in history if h["rating"] == "disliked"]

        liked_str = "\n".join(
            f"- العنوان: {x['title']} | المدة: {x['duration']}ث | التصنيف: {x['category']} | "
            f"ملاحظة الممنتج: {x['note']} | جزء من الكلام: {x.get('excerpt', '')[:200]}"
            for x in liked[-20:]
        )
        disliked_str = "\n".join(
            f"- العنوان: {x['title']} | المدة: {x['duration']}ث | التصنيف: {x['category']} | "
            f"سبب الرفض: {x['note']} | جزء من الكلام: {x.get('excerpt', '')[:200]}"
            for x in disliked[-20:]
        )

        prompt = f"""أنت خبير تحليل أسلوب ومونتاج محتوى تويتش وتيك توك.
مهمتك تحديث ملف "قواعد ذوق الممنتج" بناءً على اللقطات التي أعجبته (👍) واللقطات التي رفضها (👎) وأسبابه.

### القواعد السابقة (إن وجدت):
{old_rules or "لا توجد قواعد سابقة"}

### اللقطات التي أعجبت الممنتج ({len(liked)} لقطة):
{liked_str or "لا يوجد بعد"}

### اللقطات التي رفضها الممنتج ({len(disliked)} لقطة):
{disliked_str or "لا يوجد بعد"}

اكتب دليلاً عملياً مختصراً ومرتباً بالعربية (نقاط واضحة ومباشرة بحد أقصى 12 قاعدة) مقسّماً إلى:
1. **وش يحب الممنتج (نقاط القوة):**
2. **وش يرفض الممنتج (الأغلاط الممنوعة):**
3. **أسلوب الستريمر والتعامل مع تشتتها وطول اللقطة:**

اجعل القواعد دقيقة جداً وقابلة للتطبيق المباشر عند اختيار وقصّ لقطات البث القادم.
"""
        try:
            new_rules = call_gemini_text(api_key, prompt, json_mode=False, timeout_sec=90).strip()
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(new_rules + "\n")
            return new_rules, f"🧠 تم التحديث بنجاح! الذكاء الاصطناعي حلّل {len(history)} تقييم وحفظ قواعد ذوقك الجديدة."
        except Exception as e:
            return old_rules, f"❌ تعذر تحديث القواعد: {e}"

    def export_zip(only_liked: bool):
        clips = get_clips()
        if only_liked:
            target = [c for c in clips if c.get("rating") == "liked"]
        else:
            target = clips

        if not target:
            return None, "⚠️ ما فيه لقطات مطابقة للتصدير (تأكد إنك عطيت 👍 على اللقطات اللي تبيها)."

        suffix = "liked" if only_liked else "all"
        zip_name = f"clips_{vod_id}_{suffix}.zip"
        zip_path = os.path.join(vod_dir, zip_name)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for c in target:
                for p in (c.get("video_path"), c.get("srt_path")):
                    if p and os.path.exists(p):
                        zf.write(p, arcname=os.path.basename(p))

        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        return zip_path, f"📦 تم تجهيز {len(target)} لقطة + ملفات الـ .srt ({size_mb:.1f} MB) — حملها واسحبها مباشرة لـ CapCut!"

    choices = get_choices()
    first_choice = choices[0] if choices else None

    with gr.Blocks(title="TwitchClipper Studio") as demo:
        gr.Markdown("# 🎬 TwitchClipper Studio — المراجعة، تعديل السبتايتل، وتدريب الذكاء الاصطناعي")

        with gr.Tab("✂️ مراجعة اللقطات والتقييم"):
            clip_dropdown = gr.Dropdown(
                choices=choices,
                value=first_choice,
                label="اختر اللقطة للمعاينة والتعديل",
                interactive=True,
            )

            with gr.Row():
                with gr.Column(scale=5):
                    video_player = gr.Video(label="معاينة اللقطة")
                    info_box = gr.Markdown()

                with gr.Column(scale=5):
                    srt_editor = gr.Textbox(
                        label="📝 محرر السبتايتل المنفصل (.srt) — عدّل أي كلمة غلط واحفظ",
                        lines=13,
                        rtl=True,
                    )
                    save_srt_btn = gr.Button("💾 حفظ تعديل السبتايتل في Drive", variant="secondary")
                    srt_status = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("### 🧠 تقييم اللقطة (عشان الذكاء الاصطناعي يتعلم ذوقك)")

            with gr.Row():
                rating_radio = gr.Radio(
                    choices=["👍 أعجبتني", "👎 مرفوضة", "⏳ لم تُقيّم"],
                    value="⏳ لم تُقيّم",
                    label="رأيك في اللقطة",
                )
                tags_check = gr.CheckboxGroup(
                    choices=QUICK_TAGS,
                    label="أسباب سريعة (اختياري)",
                )

            note_input = gr.Textbox(
                label="ليش عجبتك أو ليش رفضتها؟ (اكتب بكلامك العادي)",
                placeholder="مثال: أحب لما تتوتر وتدور اللي كتب في الشات / القصّة انقطعت قبل ما تكمل السالفة...",
                rtl=True,
            )
            save_fb_btn = gr.Button("✅ حفظ التقييم في ذاكرة الذكاء الاصطناعي", variant="primary")
            fb_status = gr.Markdown()

        with gr.Tab("🧠 دماغ الذكاء الاصطناعي (قواعد ذوقك)"):
            gr.Markdown(
                "بعد ما تقيّم اللقطات في التبويب الأول، اضغط الزر تحت عشان الذكاء الاصطناعي يراجع كل تقييماتك "
                "ويحدّث قواعد اختيار وقصّ اللقطات تلقائياً للبث الجاي (أو لما تطلب لقطات زيادة من نفس البث)."
            )
            train_btn = gr.Button("🚀 حدّث قواعد الذكاء الاصطناعي من تقييماتي الآن", variant="primary")
            train_status = gr.Markdown()
            rules_editor = gr.Textbox(
                value=load_rules_text(),
                label="📜 ملف قواعد ذوقك الدائم (my_editing_rules.md) — تقدر تعدل وتضيف فيه بيدك",
                lines=15,
                rtl=True,
            )
            save_rules_btn = gr.Button("💾 حفظ تعديلات القواعد يدوياً")

        with gr.Tab("📦 تصدير سريع لـ CapCut"):
            only_liked_cb = gr.Checkbox(value=True, label="تصدير اللقطات اللي عطيتها (👍 أعجبتني) فقط")
            zip_btn = gr.Button("📦 تجهيز ملف ZIP (فيديو + .srt) للتحميل", variant="primary")
            zip_status = gr.Markdown()
            zip_file = gr.File(label="اضغط لتحميل الملف لجهازك")

        # ربط الأحداث
        clip_dropdown.change(
            fn=load_clip_ui,
            inputs=[clip_dropdown],
            outputs=[video_player, info_box, srt_editor, rating_radio, tags_check, note_input],
        )
        demo.load(
            fn=load_clip_ui,
            inputs=[clip_dropdown],
            outputs=[video_player, info_box, srt_editor, rating_radio, tags_check, note_input],
        )
        save_srt_btn.click(fn=save_srt_edit, inputs=[clip_dropdown, srt_editor], outputs=[srt_status])
        save_fb_btn.click(
            fn=save_feedback,
            inputs=[clip_dropdown, rating_radio, tags_check, note_input],
            outputs=[clip_dropdown, fb_status],
        )
        train_btn.click(fn=train_ai_rules, inputs=[], outputs=[rules_editor, train_status])
        save_rules_btn.click(fn=save_rules_manual, inputs=[rules_editor], outputs=[train_status])
        zip_btn.click(fn=export_zip, inputs=[only_liked_cb], outputs=[zip_file, zip_status])

    demo.launch(share=True, allowed_paths=[drive_root, "/tmp"])
