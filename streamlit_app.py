import streamlit as st
import json

@st.cache_data(show_spinner=False)
def load_rollouts(raw_bytes: bytes) -> list[dict]:
    """Parse raw JSONL bytes into a list of rollout dicts."""
    lines = raw_bytes.decode("utf-8").splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            # ignore bad lines
            pass
    out = list(sorted(out, key=lambda x: x['prompt']['messages'][1]['content']['parts'][0]))
    return out

def render_rollout(rollout: dict, idx: int, total: int):
    """Render a single rollout (prompt + analysis + final)."""
    st.markdown(f"---\n## Rollout {idx+1}/{total}")

    # Prompt
    prompt = rollout["prompt"]["messages"][1]
    with st.chat_message(prompt["author"]["role"]):
        st.write(prompt["content"]["parts"][0])

    # Analysis channel
    with st.expander("Analysis channel", expanded=True):
        for msg in rollout["conversation"]["messages"]:
            if msg["channel"] != "analysis":
                continue
            with st.chat_message(msg["author"]["role"]):
                content = msg.get("content", {})
                if parts := content.get("parts"):
                    st.write(parts[0])
                elif text := content.get("text"):
                    st.write(text)
                elif result := content.get("result"):
                    st.write(result)

    # Final channel with per-message LaTeX toggle
    for i, msg in enumerate(rollout["conversation"]["messages"]):
        if msg["channel"] != "final":
            continue
        part = msg.get("content", {}).get("parts", [""])[0]
        with st.chat_message(msg["author"]["role"]):
            st.write(part, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Rollouts", layout="wide")
    st.title("Rollouts")

    # Allow uploading multiple JSONL files at once
    uploaded_files = st.file_uploader(
        "Upload one JSONL file per model", 
        type="jsonl", 
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Please upload at least one JSONL file to begin.")
        return

    # Create one tab per uploaded file, named by filename
    tabs = st.tabs([f.name for f in uploaded_files])
    for tab, uploaded in zip(tabs, uploaded_files):
        with tab:
            model_name = uploaded.name
            raw = uploaded.getvalue()
            rollouts = load_rollouts(raw)
            total = len(rollouts)
            st.markdown(f"**{model_name}: Loaded {total} rollouts**")

            # Pick which rollout to show
            idx = st.slider(
                "Choose rollout", 
                min_value=1, 
                max_value=total, 
                value=1, 
                key=f"slider_{model_name}"
            ) - 1

            render_rollout(rollouts[idx], idx, total)

if __name__ == "__main__":
    main()
