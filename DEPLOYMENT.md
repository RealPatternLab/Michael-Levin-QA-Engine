# Deployment Guide for Streamlit Cloud

## Local Development

1. **Using .env file** (recommended for local development):
   ```bash
   # Add your API keys to .env file
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

2. **Using .streamlit/secrets.toml** (alternative for local development):
   ```toml
   # Add your API keys to .streamlit/secrets.toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

## Streamlit Cloud Deployment

1. **Push your code to GitHub**

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set the app path to `app.py`

3. **Add Secrets in Streamlit Cloud**:
   - In your Streamlit Cloud dashboard, go to your app
   - Click on "Settings" → "Secrets"
   - Add your API keys in TOML format:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

4. **Deploy**:
   - Click "Deploy" in Streamlit Cloud
   - Your app will be available at the provided URL

## Required Files for Deployment

Make sure these files are included in your GitHub repository:
- `app.py` (main app)
- `requirements.txt` (dependencies)
- `outputs/faiss_index.bin` (vector index)
- `outputs/combined_chunks.json` (semantic chunks)
- `configs/settings.py` (configuration)

## Troubleshooting

- **API Key Not Found**: Make sure you've added the secrets in Streamlit Cloud dashboard
- **Missing Files**: Ensure `faiss_index.bin` and `combined_chunks.json` are committed to GitHub
- **Import Errors**: Check that all dependencies are in `requirements.txt` 