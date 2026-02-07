
# Multi-Agent PDF Extractor

A modular, multi-agent system for extracting, cleaning, and structuring information from PDF documents.  
Built for scalability and downstream NLP / AI workflows.

---

## Overview

This project uses a **multi-agent architecture** where each agent is responsible for a specific stage in the PDF processing pipeline.

The objective is to convert **unstructured PDF files into structured, machine-readable data**.

---

## Architecture

```

PDF File
↓
PDF Loader Agent
↓
Text Extraction Agent
↓
Cleaning & Normalization Agent
↓
Structure Detection Agent
↓
Structured Output (JSON / TXT)

```

Each agent is independent and easy to extend or replace.

---

## Tech Stack

- Python 3.x
- PDF Processing (PyPDF / pdfplumber)
- Text Processing (regex, standard NLP utilities)
- Output Format: JSON
- Design Pattern: Multi-Agent, Modular Pipeline

---

## Project Structure

```

Multi-agent_pdfextractor/
│
├── agents/
│   ├── pdf_loader.py
│   ├── text_extractor.py
│   ├── cleaner.py
│   └── structure_agent.py
│
├── input_pdfs/
│   └── sample.pdf
│
├── outputs/
│   └── extracted_output.json
│
├── main.py
├── requirements.txt
└── README.md

````

---

## How It Works

1. **PDF Loader Agent**  
   Loads and validates PDF files.

2. **Text Extraction Agent**  
   Extracts raw text page-by-page.

3. **Cleaning Agent**  
   Removes noise such as headers, footers, and extra spaces.

4. **Structure Agent**  
   Identifies headings, paragraphs, and logical sections.

---

## Installation

```bash
pip install -r requirements.txt
````

---

## Usage

```bash
python main.py
```

* Place input PDFs inside `input_pdfs/`
* Extracted results will be saved in `outputs/`

---

## Output Example

```json
{
  "title": "Sample Document",
  "sections": [
    {
      "heading": "Introduction",
      "content": "This document explains..."
    }
  ]
}
```

---

## Use Cases

* Research paper parsing
* Resume and document analysis
* Legal or medical PDF processing
* NLP dataset creation
* AI document understanding pipelines

---

## Future Enhancements

* OCR support for scanned PDFs
* LLM-based semantic section detection
* Parallel agent execution
* REST API or GUI interface



```

If you want a **1-page minimal README**, or **LLM / research-oriented version**, say which one.
```
