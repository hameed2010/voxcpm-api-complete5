# VoxCPM2 Professional API

خدمة FastAPI احترافية لاستنساخ الصوت باستخدام VoxCPM2، التفريغ العربي باستخدام Whisper، التحقق من تشابه المتحدث باستخدام ECAPA-TDNN، ودمج الصوتيات وإخراجها بصيغة OGG/Opus.

## المبدأ

يتم تحميل النماذج الثلاثة عند Startup:

1. VoxCPM2
2. Whisper large-v3
3. ECAPA speaker verification

وبالتالي لا يحدث تحميل للنماذج عند أول طلب.

## الهيكل

```text
app/
├── api/routes/              # HTTP فقط
├── core/                    # Config / Security / Lifecycle / Models
├── schemas/                 # API contracts
├── services/
│   ├── voice/               # Voice cloning + Best-of-N
│   ├── transcription/       # Whisper
│   ├── audio/               # Merge + FFmpeg
│   └── storage/             # Files
└── utils/                   # Audio utilities
storage/
├── uploads/
├── outputs/
└── temp/
```

## Endpoints

### Health
`GET /api/v1/health`

### Voice cloning
`POST /api/v1/voice/clone`

Multipart:
- `reference_audio`
- `prompt_text`
- `target_text`
- `cfg_value`
- `inference_timesteps`
- `candidates`
- `seed`
- `retry_badcase`
- `retry_max_times`
- `retry_ratio_threshold`
- `auto_transcribe`

### Transcription
`POST /api/v1/audio/transcribe`

### Merge
`POST /api/v1/audio/merge`

### Audio
`GET /api/v1/voice/audio/{filename}`
`GET /api/v1/audio/output/{filename}`

## Authentication

أرسل:

`X-API-Key: YOUR_SECRET`

## تشغيل

```bash
cp .env.example .env
# عدّل API_KEY
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Swagger:
`http://SERVER:8000/docs`

## ملاحظة

تم فصل طبقة VoxCPM2 عن FastAPI. منطق `model.generate()` الموجود في Notebook المستخدم تم الحفاظ عليه، بما في ذلك:
- prompt_wav_path
- prompt_text
- reference_wav_path
- cfg_value
- inference_timesteps
- retry_badcase
- Best-of-N
- seed compatibility
- ECAPA speaker scoring
- إرسال النص كاملاً بدون Smart Chunking

كما تم دمج Whisper وFFmpeg في الخدمات المنفصلة.

## Security requirements
Set a strong `API_KEY` in `.env`; the application refuses to start with the default `change-me`. The API enforces `Candidate=1` and limits request rate/file size.
