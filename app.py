"""
Aplicación Flask — Verificador de documentos de identidad colombianos.
Usa Server-Sent Events (SSE) para enviar cada fila procesada en tiempo real.
"""
import json
import uuid
from pathlib import Path

from flask import Flask, Response, render_template, request, stream_with_context

from processor import process_excel_stream

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
TMP_DIR = BASE_DIR / "uploads" / "tmp_images"
UPLOAD_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".xlsx", ".xlsm"}


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _sse(data: dict) -> str:
    """Formatea un dict como evento SSE."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/procesar", methods=["POST"])
def procesar():
    if "archivo" not in request.files:
        return Response(
            _sse({"type": "error", "message": "No se recibió ningún archivo."}),
            mimetype="text/event-stream",
        ), 400

    file = request.files["archivo"]
    if not file.filename or not _allowed(file.filename):
        return Response(
            _sse({"type": "error", "message": "Solo se aceptan archivos .xlsx o .xlsm."}),
            mimetype="text/event-stream",
        ), 400

    safe_name = f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower()}"
    excel_path = UPLOAD_DIR / safe_name
    file.save(str(excel_path))

    def generate():
        try:
            for event in process_excel_stream(excel_path, TMP_DIR):
                yield _sse(event)
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            excel_path.unlink(missing_ok=True)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # evita buffering en nginx/proxies
        },
    )


if __name__ == "__main__":
    # threaded=True es necesario para que el streaming funcione correctamente
    app.run(debug=True, threaded=True)
