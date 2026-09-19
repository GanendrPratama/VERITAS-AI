"""PDF -> Markdown conversion for uploaded reports, via markitdown
(https://github.com/microsoft/markitdown) -- preserves headings/lists/tables
instead of pypdf's flat text dump, so the Analyst/Interviewer prompts see the
report's actual structure.

Falls back to pypdf's plain-text extraction if markitdown isn't installed or
fails on a given PDF -- same degrade-never-crash pattern as every other
optional dependency in this codebase (services/llm.py, services/sensors.py,
etc.; see design doc Section 4).
"""
import io
import tempfile


def pdf_to_markdown(pdf_bytes):
    try:
        from markitdown import MarkItDown

        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(pdf_bytes)
            f.flush()
            result = MarkItDown().convert(f.name)
            text = (result.text_content or "").strip()
            if text:
                return text
    except Exception:
        pass
    return _pypdf_fallback(pdf_bytes)


def _pypdf_fallback(pdf_bytes):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _selfcheck():
    # No bundled sample PDF -- build a minimal blank one with pypdf (already
    # a dependency) and check the conversion returns a string without
    # crashing, exercising markitdown when installed and the fallback path
    # when it isn't.
    from pypdf import PdfWriter

    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buf)
    text = pdf_to_markdown(buf.getvalue())
    assert isinstance(text, str)
    print("docconvert.py self-check passed")


if __name__ == "__main__":
    _selfcheck()
