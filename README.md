# DocuMind — Chat with your PDFs

A small, deployable PDF RAG app built with Streamlit. Upload one or more PDFs, ask questions in a chat interface, and get answers based on relevant passages from those documents.

## Features

- Upload multiple PDF files.
- Extract text page by page and split it into overlapping passages.
- Retrieve relevant passages using in-memory TF-IDF vectors and cosine similarity.
- Generate grounded answers with the Groq API.
- Keep a short conversation history for follow-up questions.
- Show the PDF filename and page number for retrieved sources.
- No SQL/NoSQL database, vector database, LangChain, or permanent document storage.

## Tech stack

- **UI:** Streamlit
- **PDF parsing:** pypdf
- **Retrieval:** scikit-learn TF-IDF + cosine similarity (local and in memory)
- **Answer generation:** Groq Chat Completions API

> This is a deliberately lightweight RAG example. TF-IDF is lexical vector retrieval, not a neural embedding model. It is easy to deploy and works well for many documents with clear wording, but semantic retrieval can be added later if needed.

## Run locally

1. Install Python 3.10–3.12.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment:

   **Windows (PowerShell)**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Paste a Groq API key into the sidebar, upload PDFs, click **Process documents**, and start chatting.

Create a Groq API key from the [Groq Console](https://console.groq.com/keys). The app defaults to `openai/gpt-oss-120b` (with `openai/gpt-oss-20b` as an alternative). The app does not need the key to index PDFs; it needs the key only to generate AI answers.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and push the contents of this folder to it.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. In **Advanced settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

5. Deploy. The app installs its dependencies from `requirements.txt`.

You can also skip adding a Streamlit Secret and paste the API key into the app's sidebar after it opens. Never commit a real API key or `.streamlit/secrets.toml` to GitHub.

## How the RAG flow works

1. **Load:** `pypdf` extracts text from each PDF page.
2. **Chunk:** page text is split into overlapping passages while preserving the source filename and page number.
3. **Index:** `TfidfVectorizer` turns passages into sparse vectors in memory.
4. **Retrieve:** cosine similarity finds the most relevant passages for the user's question.
5. **Generate:** the selected passages and question are sent to the chosen Groq model. The prompt tells the model not to guess when the supplied passages do not contain the answer.
6. **Show sources:** the UI displays the PDF filenames and page numbers used for the response.

## Notes and limitations

- The index and chat history live in Streamlit session memory. They are not saved to a database and may be lost when the session restarts or the app sleeps.
- Scanned/image-only PDFs are not OCR processed. Use text-selectable PDFs.
- TF-IDF retrieval is strongest when the question and document use similar terminology. It is intentionally chosen to keep deployment lightweight and avoid downloading a large embedding model.
- Groq model availability, rate limits, and pricing may change. The app includes a few model choices; select one enabled for your Groq account.
- Do not upload confidential documents to a public app unless you understand the hosting and API data-handling terms. Retrieved PDF excerpts are sent to Groq to generate answers.
