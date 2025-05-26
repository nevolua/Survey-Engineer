import os
import json
from typing import Dict

from flask import Flask, render_template, request, abort
from openai import OpenAI, OpenAIError

app = Flask(__name__, static_folder='static', template_folder='templates')

FORMAT_PATH = os.getenv('FORMAT_JSON_PATH', 'format.json')
try:
    with open(FORMAT_PATH, 'r', encoding='utf-8') as f:
        SURVEY_FORMAT = json.load(f)
except (OSError, json.JSONDecodeError) as exc:
    raise RuntimeError(f"Failed to load survey format from {FORMAT_PATH}") from exc


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or abort(500, description="OpenAI API key not configured")
client = OpenAI(api_key=OPENAI_API_KEY)


def sanitize_json(text: str) -> str:
    return text.replace("'", '"')


async def generate_survey(prompt: str, system_prompt: str) -> Dict:
    try:
        res = await client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ],
        )
    except OpenAIError as exc:
        abort(502, description=f"AI service error: {exc}")

    content = res.choices[0].message.content
    sanitized = sanitize_json(content)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError as exc:
        abort(500, description="AI returned malformed JSON")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
async def submit():
    prompt = request.form.get('prompt') or abort(400, description="Missing prompt")
    system_prompt = (
        'You are an assistant that takes a user description of a survey and outputs valid JSON '
        f'using this format: {json.dumps(SURVEY_FORMAT)}. Never use single quotes in JSON.'
    )

    survey = await generate_survey(prompt, system_prompt)
    return render_template('json.html', survey_json=json.dumps(survey, indent=2))


@app.route('/update', methods=['POST'])
async def update():
    """Modify existing survey JSON according to user instructions."""
    current_json = request.form.get('json') or abort(400, description="Missing JSON payload")
    prompt = request.form.get('prompt') or abort(400, description="Missing prompt")

    system_prompt = (
        'Modify the following survey JSON only as instructed. Do not remove or change anything else. '
        f'Current JSON: {current_json} Format: {json.dumps(SURVEY_FORMAT)}'
    )

    updated = await generate_survey(prompt, system_prompt)
    return render_template('json.html', survey_json=json.dumps(updated, indent=2))


if __name__ == '__main__':
    port = int(os.getenv('PORT', 80))
    app.run(host='0.0.0.0', port=port, debug=False)
