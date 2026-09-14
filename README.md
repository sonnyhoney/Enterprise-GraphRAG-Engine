# 🕸️ Enterprise GraphRAG Engine
> **High-Performance Multi-Format Knowledge Graph & Zero-Hallucination Intelligence System powered by Neo4j Aura Cloud & Google Gemini.**

[![Neo4j](https://img.shields.io/badge/Neo4j-Aura%20GDS%20Cloud-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Flash-4285F4?logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![GraphRAG](https://img.shields.io/badge/Architecture-Universal%20GraphRAG-00d2ff)](https://github.com/sonnyhoney/Enterprise-GraphRAG-Engine)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Zero-Hallucination](https://img.shields.io/badge/Grounding-Deterministic%20Audit-10b981)]()

---

## 📌 Executive Overview

Standard Vector Search (RAG) breaks down when handling **complex, multi-hop relationships** across unstructured enterprise data (e.g., cross-border trade regulations, supply chain dependencies, and legal compliance contracts).

**Enterprise GraphRAG Engine** solves this by converting unstructured documents into an instant **Neo4j Property Graph** and using **2-Hop topological Cypher traversals** to feed verifiable graph context into Google Gemini. If factual proof does not exist in the graph, the system explicitly reports missing data rather than hallucinating.

---

## 🏗️ System Architecture

[ Unstructured Enterprise Ingestion ]
                          (PDF, Scanned OCR, DOCX, CSV, TXT, Markdown)
                                                │
                                                ▼
                              [ Chunking & Entity Extraction Pipeline ]
                           (Sliding Window Text Chunks + Gemini Flash)
                                                │
                                                ▼
                                 [ Dynamic Knowledge Graph ]
                                (Neo4j Aura Cloud GDS Instance)
                                 /                             \
                                /                               \
 [ 2-Hop Cypher GraphRAG Retrieval ]                   [ Interactive Graph Visualizer ]
   (Topological Context Assembly)                          (PyVis Physics Simulation)
                 │
                 ▼
   [ Grounded Reasoning Engine ]
     (Gemini Guardrail Prompt)
                 │
                 ▼

[ Auditable Multi-Format Export Engine ]
(JSON / MD / TXT)


## ✨ Core Engineering Highlights

* **Multi-Page Sliding Chunk Parser:** Automatically partitions large multi-page dossiers (PDF, DOCX, CSV, TXT, MD) into 3.5k-character chunks for comprehensive entity-relationship extraction.
* **Multimodal OCR Fallback:** Detects scanned image PDFs and triggers Gemini Vision OCR to extract graph triples without external OCR dependencies.
* **Deterministic 2-Hop GraphRAG:** Executes dynamic Cypher path traversal:
  ```cypher
  MATCH (a)-[r]->(b) OPTIONAL MATCH (b)-[r2]->(c)
  WHERE size(matches) > 0
  RETURN labels(a)[0], a.name, type(r), labels(b)[0], b.name, type(r2), labels(c)[0], c.name
  LIMIT 40

* **Interactive In-Browser Topology (PyVis):** Renders interactive subgraphs filtered strictly by the user's query or specific uploaded document.
* **Full Auditability & Verifiable Citations:** Every advisory report includes the exact retrieved Neo4j triples used as ground truth evidence.
* **Automated Data Privacy Engine:** Ephemeral upload caching with automatic disk cleanup routines post-execution.

## 📊 Benchmark Showcase: AfCFTA Cross-Border Trade

The platform includes pre-configured datasets based on the **African Continental Free Trade Area (AfCFTA)** regulatory framework, demonstrating multi-hop trade intelligence:

* Rules of Origin threshold verification
* Tariff schedule & duty rate mapping
* Cross-border transit documentation compliance

🚀 Quickstart & Local Setup
1. Clone the repository

git clone https://github.com/sonnyhoney/Enterprise-GraphRAG-Engine.git
cd Enterprise-GraphRAG-Engine

## 🛠️ Tech Stack & Ecosystem

| Layer | Technologies Used |
| :--- | :--- |
| **Knowledge Engine** | Neo4j Aura Cloud, Cypher, Property Graph Modeling |
| **LLM & Vision Engine** | Google Gemini Flash (`google-genai` SDK), Multimodal OCR |
| **Graph Visualization** | PyVis, Vis.js Physics Engine, Streamlit Components |
| **Data Parsing & ETL** | PyMuPDF (MuPDF), Python-Docx, Pandas, Regular Expressions |
| **Frontend UI** | Streamlit, Custom Dark CSS |

---

## 👨‍💻 Engineer & Contact

* **Specialization:** Neo4j Certified Graph Data Science & Enterprise AI Architectures
* **LinkedIn:** [linkedin.com/in/sonnyhoney](https://www.linkedin.com/in/sonnyhoney/)
* **GitHub:** [@sonnyhoney](https://github.com/sonnyhoney)

---
*MIT License © 2026 Sonny Honey. Built for Enterprise Knowledge Graph & GraphRAG Deployment.*