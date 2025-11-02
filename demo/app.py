import gradio as gr
import subprocess, shutil, os, zipfile, datetime, sys, time, uuid, stat, re
from pathlib import Path
import base64


# =====================
# Version guard
# =====================
def _ensure_versions():
    import importlib, subprocess, sys

    def get_version(pkg):
        try:
            m = importlib.import_module(pkg)
            return getattr(m, "__version__", "0")
        except Exception:
            return "0"

    try:
        from packaging.version import Version
    except ImportError:
        # 安装packaging，确保下面版本比较能用
        subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
        from packaging.version import Version

    # 检查 huggingface_hub
    hub_ver = get_version("huggingface_hub")
    hv = Version(hub_ver)

    required_min = Version("0.24.0")
    required_max = Version("1.0.0")

    hub_ok = required_min <= hv < required_max

    if not hub_ok:
        print(f"[INFO] huggingface_hub=={hub_ver} not in range "
              f"[{required_min}, {required_max}), reinstalling...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "huggingface-hub==0.27.1",
            "transformers==4.48.0",
            "--force-reinstall", "--no-deps"
        ])
    else:
        print(f"[INFO] huggingface_hub version OK: {hub_ver}")

_ensure_versions()

# =====================
# Paths (read-only repo root; DO NOT write here)
# =====================
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"              # all per-run workspaces live here
RUNS_DIR.mkdir(parents=True, exist_ok=True)

TIMEOUT_SECONDS = 1800  # 30 minutes
RETENTION_HOURS = 1    # auto-clean runs older than N hours
DEFAULT_RIGHT_LOGO_PATH = ROOT / "template" / "logos" / "right_logo.png"

# ---------------------
# Utils
# ---------------------
def _now_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def _write_logs(log_path: Path, logs):
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
    except Exception:
        pass

def _on_rm_error(func, path, exc_info):
    # fix "PermissionError: [Errno 13] Permission denied" for readonly files
    os.chmod(path, stat.S_IWRITE)
    func(path)

def _copytree(src: Path, dst: Path, symlinks=True, ignore=None):
    if dst.exists():
        shutil.rmtree(dst, onerror=_on_rm_error)
    shutil.copytree(src, dst, symlinks=symlinks, ignore=ignore)

def _safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def _cleanup_old_runs(max_age_hours=12):
    try:
        now = datetime.datetime.now().timestamp()
        for run_dir in RUNS_DIR.iterdir():
            try:
                if not run_dir.is_dir():
                    continue
                mtime = run_dir.stat().st_mtime
                age_h = (now - mtime) / 3600.0
                if age_h > max_age_hours:
                    shutil.rmtree(run_dir, onerror=_on_rm_error)
            except Exception:
                continue
    except Exception:
        pass

def _prepare_workspace(logs):
    """Create isolated per-run workspace and copy needed code/assets into it."""
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    work_dir = RUNS_DIR / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Per-run log & zip path
    log_path = work_dir / "run.log"
    zip_path = work_dir / "output.zip"

    logs.append(f"🧩 New workspace: {work_dir.relative_to(ROOT)} (run_id={run_id})")

    # Copy code/assets that do file IO so they are run-local (avoid shared writes)
    # Keep copies as cheap as possible (symlinks=True when supported)
    needed_dirs = ["posterbuilder", "Paper2Poster"]
    for d in needed_dirs:
        src = ROOT / d
        if src.exists():
            _copytree(src, work_dir / d, symlinks=True)
            logs.append(f"   ↪ copied {d}/ → runs/{run_id}/{d}/ (symlink where possible)")

    # template/ optional
    tmpl = ROOT / "template"
    if tmpl.exists():
        _copytree(tmpl, work_dir / "template", symlinks=True)
        logs.append("   ↪ copied template/")

    # pipeline.py must live inside workspace so that ROOT_DIR=work_dir
    _safe_copy(ROOT / "pipeline.py", work_dir / "pipeline.py")

    # Create standard IO dirs in workspace
    (work_dir / "input" / "pdf").mkdir(parents=True, exist_ok=True)
    (work_dir / "input" / "logo").mkdir(parents=True, exist_ok=True)
    (work_dir / "posterbuilder" / "latex_proj").mkdir(parents=True, exist_ok=True)

    return run_id, work_dir, log_path, zip_path

# ---------------------
# Helpers for new features (post-processing)
# ---------------------
def _parse_rgb(val):
    """Return (R, G, B) as ints in [0,255] from '#RRGGBB', 'rgb(...)', 'rgba(...)', 'r,g,b', [r,g,b], or (r,g,b)."""
    if val is None:
        return None

    import re

    def clamp255(x):
        try:
            return max(0, min(255, int(round(float(x)))))
        except Exception:
            return None

    s = str(val).strip()

    # list/tuple
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        r, g, b = [clamp255(val[0]), clamp255(val[1]), clamp255(val[2])]
        if None not in (r, g, b):
            return (r, g, b)

    # hex: #RGB or #RRGGBB
    if s.startswith("#"):
        hx = s[1:].strip()
        if len(hx) == 3:
            hx = "".join(c*2 for c in hx)
        if len(hx) == 6 and re.fullmatch(r"[0-9A-Fa-f]{6}", hx):
            return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))

    # rgb/rgba(...)
    m = re.match(r"rgba?\(\s*([^)]+)\)", s, flags=re.IGNORECASE)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) >= 3:
            def to_int(p):
                if p.endswith("%"):
                    # percentage to 0-255
                    return clamp255(float(p[:-1]) * 255.0 / 100.0)
                return clamp255(p)
            r, g, b = to_int(parts[0]), to_int(parts[1]), to_int(parts[2])
            if None not in (r, g, b):
                return (r, g, b)

    # 'r,g,b'
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 3:
            def to_int(p):
                if p.endswith("%"):
                    return clamp255(float(p[:-1]) * 255.0 / 100.0)
                return clamp255(p)
            r, g, b = to_int(parts[0]), to_int(parts[1]), to_int(parts[2])
            if None not in (r, g, b):
                return (r, g, b)

    return None


def _apply_meeting_logo(OUTPUT_DIR: Path, meeting_logo_file, logs):
    """Replace output/poster_latex_proj/logos/right_logo.png if meeting_logo_file provided."""
    if not meeting_logo_file:
        return False

    logos_dir = OUTPUT_DIR / "poster_latex_proj" / "logos"
    target = logos_dir / "right_logo.png"
    try:
        logos_dir.mkdir(parents=True, exist_ok=True)
        # Try to convert to PNG for safety
        try:
            from PIL import Image
            img = Image.open(meeting_logo_file.name)
            # preserve alpha if available
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            img.save(target, format="PNG")
            logs.append(f"🖼️ Meeting logo converted to PNG and saved → {target.relative_to(OUTPUT_DIR)}")
        except Exception as e:
            # Fallback: raw copy with .png name
            shutil.copy(meeting_logo_file.name, target)
            logs.append(f"🖼️ Meeting logo copied (no conversion) → {target.relative_to(OUTPUT_DIR)} (note: ensure it's a valid PNG).")
        return True
    except Exception as e:
        logs.append(f"⚠️ Failed to apply meeting logo: {e}")
        return False

def _apply_theme_rgb(OUTPUT_DIR: Path, rgb_tuple, logs):
    if not rgb_tuple:
        return False

    tex_path = OUTPUT_DIR / "poster_latex_proj" / "poster_output.tex"
    if not tex_path.exists():
        logs.append(f"⚠️ Theme RGB skipped: {tex_path.relative_to(OUTPUT_DIR)} not found.")
        return False

    try:
        content = tex_path.read_text(encoding="utf-8")
        r, g, b = rgb_tuple
        name_pattern = r"(?:nipspurple|neuripspurple|themecolor)"

        rgb_pat = rf"(\\definecolor\{{{name_pattern}\}}\{{RGB\}}\{{)\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(\}})"

        def repl_rgb(m):
            return f"{m.group(1)}{r},{g},{b}{m.group(2)}"

        new_content, n = re.subn(rgb_pat, repl_rgb, content, flags=re.MULTILINE)

        if n == 0:
            hexval = f"{r:02X}{g:02X}{b:02X}"
            html_pat = rf"(\\definecolor\{{{name_pattern}\}}\{{HTML\}}\{{)[0-9A-Fa-f]{{6}}(\}})"

            def repl_html(m):
                return f"{m.group(1)}{hexval}{m.group(2)}"

            new_content, n = re.subn(html_pat, repl_html, content, flags=re.MULTILINE)

        if n > 0:
            tex_path.write_text(new_content, encoding="utf-8")
            logs.append(f"🎨 Theme color updated to RGB {{{r},{g},{b}}}")
            return True
        else:
            logs.append("ℹ️ No \\definecolor target found.")
            return False

    except Exception as e:
        logs.append(f"⚠️ Failed to update theme RGB: {e}")
        return False



def _apply_left_logo(OUTPUT_DIR: Path, logo_files, logs):
    """
    Use the first institutional logo uploaded by the user:
    - Copy it into output/poster_latex_proj/logos/ as left_logo.<ext>
    - Replace 'logos/left_logo.png' in poster_output.tex with the proper file extension
    Does NOT convert formats. Simply renames and rewrites the tex reference.
    """
    if not logo_files:
        logs.append("ℹ️ No institutional logo uploaded.")
        return False

    if isinstance(logo_files, (list, tuple)) and len(logo_files) > 1:
        logs.append("Multiple institutional logos uploaded.")
        return False

    # Single file case
    f = logo_files[0] if isinstance(logo_files, (list, tuple)) else logo_files
    if not f:
        logs.append("ℹ️ No institutional logo uploaded.")
        return False

    ext = Path(f.name).suffix or ".png"  # fallback to .png if no extension
    logos_dir = OUTPUT_DIR / "poster_latex_proj" / "logos"
    tex_path = OUTPUT_DIR / "poster_latex_proj" / "poster_output.tex"

    try:
        logos_dir.mkdir(parents=True, exist_ok=True)
        dst = logos_dir / f"left_logo{ext}"
        shutil.copy(f.name, dst)
        logs.append(f"🏷️ Institutional logo copied to: {dst.relative_to(OUTPUT_DIR)}")
    except Exception as e:
        logs.append(f"⚠️ Failed to copy institutional logo: {e}")
        return False

    if not tex_path.exists():
        logs.append("⚠️ poster_output.tex not found, cannot replace left_logo path.")
        return False

    try:
        text = tex_path.read_text(encoding="utf-8")
        old = "logos/left_logo.png"
        new = f"logos/left_logo{ext}"

        if old in text:
            tex_path.write_text(text.replace(old, new), encoding="utf-8")
            logs.append(f"🛠️ Replaced left_logo.png → left_logo{ext} in poster_output.tex")
            return True

        # Fallback (covers weird spacing or macro variations)
        import re
        pattern = r"(logos/left_logo)\.png"
        new_text, n = re.subn(pattern, r"\1" + ext, text)

        if n > 0:
            tex_path.write_text(new_text, encoding="utf-8")
            logs.append(f"🛠️ Replaced left_logo.png → left_logo{ext} (regex fallback)")
            return True

        logs.append("ℹ️ No left_logo.png reference found in poster_output.tex.")
        return False

    except Exception as e:
        logs.append(f"⚠️ Failed to modify poster_output.tex: {e}")
        return False

def render_overleaf_button(overleaf_b64):
    if not overleaf_b64:
        return ""
    
    html = f"""
    <form action="https://www.overleaf.com/docs" method="post" target="_blank">
      <input type="hidden" name="snip_uri" value="data:application/zip;base64,{overleaf_b64}">
      <input type="hidden" name="engine" value="xelatex">
      <button style="
        background:#4CAF50;color:white;padding:8px 14px;
        border:none;border-radius:6px;cursor:pointer; margin-top:8px;
      ">
        🚀 Open in Overleaf
      </button>
    </form>
    """
    return html

# =====================
# Gradio pipeline function (ISOLATED)
# =====================
def run_pipeline(arxiv_url, pdf_file, openai_key, logo_files, meeting_logo_file, theme_rgb):
    _cleanup_old_runs(RETENTION_HOURS)

    start_time = datetime.datetime.now()
    logs = [f"🚀 Starting pipeline at {_now_str()}"]

    # --- Prepare per-run workspace ---
    run_id, WORK_DIR, LOG_PATH, ZIP_PATH = _prepare_workspace(logs)
    INPUT_DIR = WORK_DIR / "input"
    OUTPUT_DIR = WORK_DIR / "output"
    LOGO_DIR = INPUT_DIR / "logo"
    POSTER_LATEX_DIR = WORK_DIR / "posterbuilder" / "latex_proj"

    _write_logs(LOG_PATH, logs)
    yield "\n".join(logs), None

    # ====== Validation: must upload LOGO ======
    if logo_files is None:
        logo_files = []
    if not isinstance(logo_files, (list, tuple)):
        logo_files = [logo_files]
    logo_files = [f for f in logo_files if f]

    # if len(logo_files) == 0:
    #     msg = "❌ You must upload at least one institutional logo (multiple allowed)."
    #     logs.append(msg)
    #     _write_logs(LOG_PATH, logs)
    #     yield "\n".join(logs), None
    #     return

    # Save logos into run-local dir
    for item in LOGO_DIR.iterdir():
        if item.is_file():
            item.unlink()
    saved_logo_paths = []
    for lf in logo_files:
        p = LOGO_DIR / Path(lf.name).name
        shutil.copy(lf.name, p)
        saved_logo_paths.append(p)
    logs.append(f"🏷️ Saved {len(saved_logo_paths)} logo file(s) → {LOGO_DIR.relative_to(WORK_DIR)}")
    _write_logs(LOG_PATH, logs)
    yield "\n".join(logs), None

    # ====== Handle uploaded PDF (optional) ======
    pdf_path = None
    if pdf_file:
        pdf_dir = INPUT_DIR / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / Path(pdf_file.name).name
        shutil.copy(pdf_file.name, pdf_path)
        logs.append(f"📄 Uploaded PDF → {pdf_path.relative_to(WORK_DIR)}")

        # For pipeline Step 1.5 compatibility: also copy to input/paper.pdf
        canonical_pdf = INPUT_DIR / "paper.pdf"
        shutil.copy(pdf_file.name, canonical_pdf)
        _write_logs(LOG_PATH, logs)
        yield "\n".join(logs), None

    # ====== Validate input source ======
    if not arxiv_url and not pdf_file:
        msg = "❌ Please provide either an arXiv link or upload a PDF file (choose one)."
        logs.append(msg)
        _write_logs(LOG_PATH, logs)
        yield "\n".join(logs), None
        return

    # ====== Build command (run INSIDE workspace) ======
    cmd = [
        sys.executable, "pipeline.py",
        "--model_name_t", "gpt-5",
        "--model_name_v", "gpt-5",
        "--result_dir", "output",
        "--paper_latex_root", "input/latex_proj",
        "--openai_key", openai_key,
        "--gemini_key", "##",
        "--logo_dir", str(LOGO_DIR)  # run-local logo dir
    ]
    if arxiv_url:
        cmd += ["--arxiv_url", arxiv_url]
    # (Keep pdf via input/paper.pdf; pipeline will read it if exists)

    logs.append("\n======= REAL-TIME LOG =======")
    logs.append(f"cwd = runs/{WORK_DIR.name}")
    _write_logs(LOG_PATH, logs)
    yield "\n".join(logs), None

    # ====== Run with REAL-TIME streaming, inside workspace ======
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(WORK_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        msg = f"❌ Pipeline failed to start: {e}"
        logs.append(msg)
        _write_logs(LOG_PATH, logs)
        yield "\n".join(logs), None
        return

    last_yield = time.time()
    try:
        while True:
            # Timeout guard
            if (datetime.datetime.now() - start_time).total_seconds() > TIMEOUT_SECONDS:
                logs.append("❌ Pipeline timed out (30 min limit). Killing process…")
                try:
                    process.kill()
                except Exception:
                    pass
                _write_logs(LOG_PATH, logs)
                yield "\n".join(logs), None
                return

            line = process.stdout.readline()
            if line:
                print(line, end="")  # echo to Space logs
                logs.append(line.rstrip("\n"))
                _write_logs(LOG_PATH, logs)
                now = time.time()
                if now - last_yield >= 0.3:
                    last_yield = now
                    yield "\n".join(logs), None
            elif process.poll() is not None:
                break
            else:
                time.sleep(0.05)

        return_code = process.wait()
        logs.append(f"\nProcess finished with code {return_code}")
        _write_logs(LOG_PATH, logs)
        yield "\n".join(logs), None

        if return_code != 0:
            logs.append("❌ Process exited with non-zero status. See logs above.")
            _write_logs(LOG_PATH, logs)
            yield "\n".join(logs), None
            return

    except Exception as e:
        logs.append(f"❌ Error during streaming: {e}")
        _write_logs(LOG_PATH, logs)
        yield "\n".join(logs), None
        return
    finally:
        try:
            if process.stdout:
                process.stdout.close()
        except Exception:
            pass

    # ====== Check output ======
    has_output = False
    try:
        if OUTPUT_DIR.exists():
            for _ in OUTPUT_DIR.iterdir():
                has_output = True
                break
    except FileNotFoundError:
        has_output = False

    if not has_output:
        msg = "❌ No output generated. Please check logs above."
        logs.append(msg)
        _write_logs(LOG_PATH, logs)
        yield "\n".join(logs), None
        return

    # ====== NEW: Post-processing (optional features) ======
    # 1) Optional meeting logo replacement
    applied_logo = _apply_meeting_logo(OUTPUT_DIR, meeting_logo_file, logs)

    # 2) Optional theme color update
    rgb_tuple = _parse_rgb(theme_rgb)
    if theme_rgb and not rgb_tuple:
        logs.append(f"⚠️ Ignored Theme RGB input '{theme_rgb}': expected like '94,46,145'.")
    applied_rgb = _apply_theme_rgb(OUTPUT_DIR, rgb_tuple, logs) if rgb_tuple else False

    # 3) Optional institutional logo -> left_logo.<ext>
    _apply_left_logo(OUTPUT_DIR, logo_files, logs)

    _write_logs(LOG_PATH, logs)
    yield "\n".join(logs), None


    _write_logs(LOG_PATH, logs)
    yield "\n".join(logs), None

    # ====== Zip output (run-local) ======
    try:
        target_dir = OUTPUT_DIR / "poster_latex_proj"

        if not target_dir.exists():
            logs.append("❌ poster_latex_proj folder not found")
        else:
            with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(target_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(target_dir)  # only relative to subfolder
                        zipf.write(file_path, arcname=arcname)

            logs.append(f"✅ Zipped poster_latex_proj → {ZIP_PATH.relative_to(WORK_DIR)}")

    except Exception as e:
        logs.append(f"❌ Failed to create zip: {e}")

    # ====== Prepare Overleaf base64 payload (optional) ======
    overleaf_zip_b64 = ""
    try:
        with open(ZIP_PATH, "rb") as f:
            overleaf_zip_b64 = base64.b64encode(f.read()).decode("utf-8")
        logs.append("🔗 Prepared Overleaf base64 payload")
    except Exception as e:
        logs.append(f"⚠️ Failed Overleaf payload: {e}")

    end_time = datetime.datetime.now()
    dur = (end_time - start_time).seconds
    logs.append(f"🏁 Completed at {_now_str()} (Duration: {dur}s)")
    logs.append(f"🆔 run_id = {WORK_DIR.name}")

    _write_logs(LOG_PATH, logs)
    yield "\n".join(logs), (
    str(ZIP_PATH) if ZIP_PATH.exists() else None
    ), render_overleaf_button(overleaf_zip_b64)


# =====================
# Gradio UI
# =====================
with gr.Blocks(title="🎓 Paper2Poster") as iface:
    gr.Markdown("# 🎓 Paper2Poster")
    gr.Markdown("""
[Paper](https://arxiv.org/abs/2505.21497) | [GitHub](https://github.com/Paper2Poster/Paper2Poster) | [Project Page](https://paper2poster.github.io/)  

**TL;DR:** Upload your paper and get an auto-generated poster.
Please be patient — each paper takes about 8–10 minutes to process.

This work, developed in collaboration with [TVG@Oxford](https://torrvision.com/index.html) and [UWaterloo](https://uwaterloo.ca/), has been accepted to [NeurIPS 2025 D&B](https://neurips.cc/).
The framework builds upon [CAMEL-ai](https://github.com/camel-ai/camel).
""")

    # -------- Input box --------
    with gr.Row():
        # ========== LEFT: INPUT ==========
        with gr.Column(scale=1):
            with gr.Accordion("Input", open=True):
                arxiv_in = gr.Textbox(label="📘 ArXiv URL (choose one)", placeholder="https://arxiv.org/abs/2505.xxxxx")
                pdf_in   = gr.File(label="📄 Upload PDF (choose one)")
                key_in   = gr.Textbox(label="🔑 OpenAI API Key", placeholder="sk-...", type="password")

                inst_logo_in = gr.File(
                    label="🏷️ Institutional Logo (optional, multiple allowed)",
                    file_count="multiple",
                    file_types=["image"],
                )

                with gr.Row():
                    with gr.Column():
                        conf_logo_in = gr.File(
                            label="🧩 Optional: Conference Logo (defaults to NeurIPS logo)",
                            file_count="single",
                            file_types=["image"],
                        )
                    with gr.Column():
                        conf_preview = gr.Image(
                            value=str(DEFAULT_RIGHT_LOGO_PATH) if DEFAULT_RIGHT_LOGO_PATH.exists() else None,
                            label="Default conference logo preview",
                            interactive=False,
                        )

                theme_in = gr.ColorPicker(label="🎨 Theme Color (optional)", value="#5E2E91")
                run_btn = gr.Button("🚀 Run", variant="primary")

        # ========== RIGHT: OUTPUT ==========
        with gr.Column(scale=1):
            with gr.Accordion("Output", open=True):
                logs_out     = gr.Textbox(label="🧾 Logs (8–10 minutes)", lines=30, max_lines=50)
                zip_out      = gr.File(
                    label="📦 Download Results (.zip)",
                    info="📌 After uploading the ZIP to Overleaf, please select **XeLaTeX** as the compile engine.",
                )                
    run_btn.click(
        fn=run_pipeline,
        inputs=[arxiv_in, pdf_in, key_in, inst_logo_in, conf_logo_in, theme_in],
        outputs=[logs_out, zip_out],
    )

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
