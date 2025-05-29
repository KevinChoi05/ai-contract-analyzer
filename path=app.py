from flask import Flask, request, Response, url_for
import time, json
from pdfminer.high_level import extract_text
# … your existing imports & configuration …

@app.route("/process", methods=["POST"])
def process_stream():
    """
    An SSE endpoint that streams status updates during processing.
    """
    file = request.files.get("contract")
    if not file:
        # immediately error back
        return Response("data: ERROR: no file\n\n", mimetype="text/event-stream")

    # save to disk
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    def generate():
        # STEP 1
        yield f"data: Extracting text from PDF…\n\n"
        try:
            text = extract_text(filepath)
        except Exception as e:
            yield f"data: ❌ Failed to extract text: {e}\n\n"
            yield f"event: done\ndata: {{\"error\":true}}\n\n"
            return

        # STEP 2
        yield f"data: Chunking document…\n\n"
        chunks = chunk_text_with_overlap(text)
        yield f"data: → {len(chunks)} chunks generated\n\n"

        # STEP 3
        for i, chunk in enumerate(chunks, 1):
            yield f"data: Summarizing chunk {i}/{len(chunks)}…\n\n"
            try:
                summary = deepseek_summarize_chunk(chunk)
            except Exception as e:
                yield f"data: ❌ Chunk {i} summary failed: {e}\n\n"
                yield f"event: done\ndata: {{\"error\":true}}\n\n"
                return
            # optionally buffer summaries…

        # STEP 4
        yield f"data: Combining summaries…\n\n"
        compressed = deepseek_combine_summaries([...])  # pass your collected summaries

        # STEP 5
        yield f"data: Analyzing contract risk…\n\n"
        try:
            clauses = analyze_contract(compressed)
        except Exception as e:
            yield f"data: ❌ Risk analysis failed: {e}\n\n"
            yield f"event: done\ndata: {{\"error\":true}}\n\n"
            return

        # DONE
        # you can either redirect or return the data here
        payload = json.dumps({ "filename": file.filename,
                               "preview": text[:1000],
                               "summary": compressed,
                               "clauses": clauses })
        yield f"event: done\n"
        yield f"data: {payload}\n\n"

    return Response(generate(), mimetype="text/event-stream") 