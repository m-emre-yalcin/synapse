# Synapse AI Assistant

An AI-powered companion that provides **empathetic**, **context-aware**, and **Obsidian-ready** Markdown responses based *strictly* on your personal notes.

Built with:
- 🧠 **LangChain** for chaining prompts & retrieval
- 📚 **ChromaDB** for vector storage
- 🔮 **VertexAI (LLM)** as the language model
- 🗂️ Markdown input/output for seamless journaling workflows

---

## ⚙️ How It Works

1. Loads and splits your local `.md` notes
2. Embeds and stores them in **ChromaDB**
3. Uses **MMR-based retrieval** to fetch relevant chunks
4. Generates safe, friendly, context-based answers via **VertexAI**
5. Saves each interaction as Markdown (with hashtags) under a `history/` folder

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/yourusername/empathic-ai.git
cd empathic-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure .env
cp .env.example .env
# → set NOTES_PATH and other values

# 4. Run the assistant
python main.py
