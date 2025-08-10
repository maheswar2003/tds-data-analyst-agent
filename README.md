# TDS Data Analyst Agent

An AI-powered data analysis API that can source, prepare, analyze, and visualize data based on natural language questions.

## Features

- Accepts multipart form-data POST requests with questions and optional attachments
- Supports both LLM-powered analysis (with OpenAI API) and offline fallback handlers
- Can scrape web data, analyze datasets, and generate visualizations
- Returns results in JSON format with base64-encoded plots
- Sub-3-minute response time guarantee

## API Endpoint

```
POST /api/
```

### Request Format

```bash
curl "http://your-domain/api/" \
  -F "questions.txt=@question.txt" \
  -F "data.csv=@data.csv" \
  -F "image.png=@image.png"
```

- `questions.txt` - **Required**: Contains the analysis questions
- Additional files - **Optional**: Supporting data files

### Response Format

The API returns either:
- A JSON array (for specific formatted questions)
- A JSON object with analysis results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tds-data-analyst-agent.git
cd tds-data-analyst-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables (optional, for LLM support):
```bash
export OPENAI_API_KEY=your-api-key-here
```

4. Start the server:
```bash
python start_server.py
```

The API will be available at `http://127.0.0.1:8000`

## Deployment

### Option 1: Deploy to Railway

1. Install Railway CLI: https://docs.railway.app/develop/cli
2. Run:
```bash
railway login
railway init
railway up
```

### Option 2: Deploy to Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Option 3: Use ngrok (for testing)

```bash
ngrok http 8000
```

## API Documentation

Once running, visit:
- API Docs: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

## Example Questions

### Wikipedia Data Scraping
```
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI under 100,000 bytes.
```

## Testing

Run the test suite:
```bash
python test_server.py
```

## License

MIT License - see LICENSE file for details

## Author

Created for the TDS Data Analyst Agent project
