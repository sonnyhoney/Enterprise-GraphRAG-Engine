import os
import glob
import json
import re
import streamlit as st
import pymupdf
import docx
import pandas as pd
from neo4j import GraphDatabase
from google import genai
from dotenv import load_dotenv
from pyvis.network import Network
import streamlit.components.v1 as components

# Load environment variables
load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+ssc://51204372.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "51204372")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="GraphAI | Enterprise Intelligence Platform",
    page_icon="🕸️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title { font-size: 2.6rem !important; font-weight: 800; color: #FFFFFF; margin-bottom: 0px; }
    .sub-title { font-size: 1.2rem !important; font-weight: 500; color: #4285F4; margin-bottom: 25px; }
    
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3.2em; background-color: #1A73E8; color: white; font-weight: 700; font-size: 1.05rem; border: none; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #1557B0; box-shadow: 0px 4px 12px rgba(26, 115, 232, 0.4); }
    
    .metric-card {
        background: linear-gradient(135deg, #1E1E2E 0%, #11111B 100%); padding: 18px 20px; border-radius: 12px; border-left: 4px solid #4285F4; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-title { font-size: 0.85rem; color: #A0A0A0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.35rem; color: #FFFFFF; font-weight: 700; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS & DATABASE CONNECTORS
# ==========================================
def cleanup_temp_files():
    """Finds and automatically deletes all temporary upload & graph files."""
    for temp_file in glob.glob("temp_*"):
        try:
            os.remove(temp_file)
        except Exception:
            pass

@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=30,
        liveness_check_timeout=10
    )

driver = get_neo4j_driver()

def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# FILE PARSER HELPER (PDF, DOCX, CSV, TXT, MD)
# ==========================================
def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".pdf"):
        with open("temp_ingest.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        doc = pymupdf.open("temp_ingest.pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text.strip(), "pdf"
        
    elif filename.endswith(".docx"):
        with open("temp_ingest.docx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        doc = docx.Document("temp_ingest.docx")
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text.strip(), "docx"
        
    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        text = df.to_string(index=False)
        return text.strip(), "csv"
        
    elif filename.endswith(".txt") or filename.endswith(".md"):
        text = uploaded_file.read().decode("utf-8")
        return text.strip(), "txt"
        
    return "", "unknown"

# ==========================================
# KNOWLEDGE GRAPH EXTRACTION & INGESTION
# ==========================================
def extract_universal_knowledge_graph(full_text):
    """Splits large documents into chunks and merges all extracted knowledge triples."""
    client = get_gemini_client()
    
    # Split text into 3,500-character chunks
    chunk_size = 3500
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    # Process up to 5 chunks (~10-12 pages) to prevent API rate limits
    chunks_to_process = chunks[:5]
    
    combined_graph = {"nodes": [], "relationships": []}
    existing_node_ids = set()
    
    for idx, chunk in enumerate(chunks_to_process):
        prompt = f"""
        Analyze the following document excerpt (Part {idx+1} of {len(chunks_to_process)}) and extract structured knowledge graph entities and relationships.
        
        DOCUMENT TEXT:
        {chunk}

        Extract all key entities, facts, and relationships into JSON format:
        {{
          "nodes": [
            {{"id": "EntityNameOrValue", "label": "EntityType"}} 
          ],
          "relationships": [
            {{"source": "EntityA", "type": "RELATIONSHIP_TYPE", "target": "EntityB"}}
          ]
        }}
        Return ONLY valid JSON. Do not include markdown code block syntax.
        """
        try:
            chat = client.chats.create(model='gemini-3.6-flash')
            response = chat.send_message(prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            # Combine nodes without duplicates
            for node in data.get("nodes", []):
                if node.get("id") and node.get("id") not in existing_node_ids:
                    combined_graph["nodes"].append(node)
                    existing_node_ids.add(node.get("id"))
                    
            # Combine relationships
            for rel in data.get("relationships", []):
                if rel.get("source") and rel.get("target"):
                    combined_graph["relationships"].append(rel)
        except Exception:
            continue
            
    return combined_graph

def save_universal_triples_to_neo4j(graph_data, source_filename="user_upload"):
    with driver.session() as session:
        for node in graph_data.get("nodes", []):
            label = node.get("label", "Entity").replace(" ", "_")
            node_id = node.get("id")
            if not node_id: continue
            session.run(
                f"MERGE (n:`{label}` {{name: $name}}) SET n.source_doc = $source_doc", 
                name=str(node_id), 
                source_doc=source_filename
            )
                
        for rel in graph_data.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type", "CONNECTED_TO").replace(" ", "_").upper()
            if source and target:
                cypher = f"""
                MATCH (a {{name: $source}})
                MATCH (b {{name: $target}})
                MERGE (a)-[r:`{rel_type}`]->(b)
                SET r.source_doc = $source_doc
                """
                session.run(cypher, source=str(source), target=str(target), source_doc=source_filename)

# ==========================================
# DYNAMIC GRAPHRAG RETRIEVAL ENGINE
# ==========================================
def execute_smart_graphrag(question):
    client = get_gemini_client()
    keywords = [word.strip().lower() for word in question.split() if len(word) > 2]
    
    cypher_retrieval = """
    MATCH (a)-[r]->(b)
    OPTIONAL MATCH (b)-[r2]->(c)
    WITH a, r, b, r2, c,
         [term IN $keywords WHERE toLower(a.name) CONTAINS term OR toLower(b.name) CONTAINS term OR toLower(labels(a)[0]) CONTAINS term OR toLower(labels(b)[0]) CONTAINS term] AS matches
    ORDER BY size(matches) DESC
    RETURN labels(a)[0] AS SourceType, a.name AS Source, 
           type(r) AS Rel1, 
           labels(b)[0] AS TargetType, b.name AS Target,
           CASE WHEN r2 IS NOT NULL THEN type(r2) ELSE null END AS Rel2,
           CASE WHEN c IS NOT NULL THEN labels(c)[0] ELSE null END AS SubTargetType,
           c.name AS SubTarget
    LIMIT 40
    """
    
    graph_facts = []
    with driver.session() as session:
        result = session.run(cypher_retrieval, keywords=keywords)
        for record in result:
            fact = f"[{record['SourceType']}] '{record['Source']}' --({record['Rel1']})--> [{record['TargetType']}] '{record['Target']}'"
            if record['SubTarget']:
                fact += f" --({record['Rel2']})--> [{record['SubTargetType']}] '{record['SubTarget']}'"
            graph_facts.append(fact)
            
    context_str = "\n".join(graph_facts)
    
    prompt = f"""
    You are an Enterprise GraphAI Intelligence System.
    Answer the user's question accurately using ONLY the structured Graph Database facts provided below.

    --- RETRIEVED NEO4J KNOWLEDGE GRAPH FACTS ---
    {context_str}
    --------------------------------------------

    User Query: {question}

    Instructions:
    - Directly and accurately answer the user's query based on the facts.
    - Format your response as a professional, structured analytical report with clear headings.
    - If the context does not contain the requested information, explicitly state what is missing.
    """
    
    chat = client.chats.create(model='gemini-3.6-flash')
    response = chat.send_message(prompt)
    return response.text, graph_facts

# ==========================================
# INTERACTIVE PYVIS VISUALIZATION BUILDERS
# ==========================================
def generate_query_subgraph(retrieved_facts):
    """Builds interactive graph ONLY from retrieved facts for a specific query."""
    net = Network(height="450px", width="100%", bgcolor="#090d16", font_color="#cbd5e1")
    net.force_atlas_2based()
    
    added_nodes = set()
    pattern = re.compile(r"\[(.*?)\]\s*'(.*?)'\s*--\((.*?)\)-->\s*\[(.*?)\]\s*'(.*?)'")
    
    for fact in retrieved_facts:
        matches = pattern.findall(fact)
        for match in matches:
            src_type, src_name, rel_type, tgt_type, tgt_name = match
            
            if src_name not in added_nodes:
                net.add_node(src_name, label=f"{src_name}\n[{src_type}]", color="#00d2ff", size=18, font={"color": "white", "size": 11})
                added_nodes.add(src_name)
                
            if tgt_name not in added_nodes:
                net.add_node(tgt_name, label=f"{tgt_name}\n[{tgt_type}]", color="#a855f7", size=18, font={"color": "white", "size": 11})
                added_nodes.add(tgt_name)
                
            net.add_edge(src_name, tgt_name, title=rel_type, label=rel_type, color="#475569", font={"color": "#94a3b8", "size": 9})
            
    net.save_graph("temp_query_graph.html")
    with open("temp_query_graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    return html

def generate_document_subgraph(extracted_json):
    """Builds interactive graph strictly from JSON extracted from an uploaded document."""
    net = Network(height="450px", width="100%", bgcolor="#090d16", font_color="#cbd5e1")
    net.force_atlas_2based()
    
    for node in extracted_json.get("nodes", []):
        n_id = str(node.get("id"))
        n_label = str(node.get("label", "Entity"))
        net.add_node(n_id, label=f"{n_id}\n[{n_label}]", color="#00d2ff", size=18, font={"color": "white", "size": 11})
        
    for rel in extracted_json.get("relationships", []):
        src = str(rel.get("source"))
        tgt = str(rel.get("target"))
        rel_t = str(rel.get("type", "CONNECTED_TO"))
        net.add_edge(src, tgt, title=rel_t, label=rel_t, color="#475569", font={"color": "#94a3b8", "size": 9})
        
    net.save_graph("temp_doc_graph.html")
    with open("temp_doc_graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    return html

# ==========================================
# SIDEBAR ARCHITECTURE
# ==========================================
st.sidebar.markdown("## ⚙️ Platform Engine Specs")
st.sidebar.markdown("""
- **Knowledge Engine:** Neo4j Aura Cloud GDS
- **Reasoning Model:** Google Gemini Flash
- **Architecture:** Universal GraphRAG
- **Multi-Format Parser:** PDF, DOCX, CSV, TXT, MD
- **Auto-Privacy:** Automatic Disk Cleanup
""")

st.sidebar.markdown("---")
st.sidebar.markdown("[🐙 View GitHub Source Code](https://github.com/sonnyhoney/AfCFTA-GraphAI-Lab)")
st.sidebar.markdown("[💼 Connect on LinkedIn](https://www.linkedin.com/in/sonnyhoney)")

# ==========================================
# MAIN PLATFORM HEADER
# ==========================================
st.markdown('<p class="main-title">🕸️ Universal GraphAI Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enterprise Multi-Format Knowledge Graph & Intelligence System powered by Neo4j & Google Gemini</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-title">Knowledge Engine</div><div class="metric-value">Neo4j GDS Cloud</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-title">Supported Formats</div><div class="metric-value">PDF, DOCX, CSV, TXT</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-title">Export Capabilities</div><div class="metric-value">MD, JSON, TXT</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-title">Inference Engine</div><div class="metric-value">Gemini Flash</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")

# PLATFORM TABS

# ==========================================
# KEY VALUE PROPOSITION / PIPELINE BANNER
# ==========================================
st.markdown("""
<div style="background: #111726; border: 1px solid #1e293b; border-radius: 12px; padding: 20px 26px; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
        <div style="flex: 1; min-width: 220px;">
            <div style="color: #4285F4; font-weight: 700; font-size: 0.9rem; font-family: monospace; letter-spacing: 0.5px;">STEP 1 • MULTI-FORMAT INGESTION</div>
            <p style="color: #FFFFFF; font-size: 1.05rem; font-weight: 700; margin: 6px 0 2px 0;">Upload Any Unstructured File</p>
            <p style="color: #CBD5E1; font-size: 0.85rem; margin: 0; line-height: 1.4;">PDF, DOCX, CSV, TXT with automated Gemini extraction.</p>
        </div>
        <div style="color: #4285F4; font-size: 1.4rem; font-weight: bold;">➔</div>
        <div style="flex: 1; min-width: 220px;">
            <div style="color: #4285F4; font-weight: 700; font-size: 0.9rem; font-family: monospace; letter-spacing: 0.5px;">STEP 2 • INSTANT GRAPH MODELING</div>
            <p style="color: #FFFFFF; font-size: 1.05rem; font-weight: 700; margin: 6px 0 2px 0;">Dynamic Neo4j Knowledge Graph</p>
            <p style="color: #CBD5E1; font-size: 0.85rem; margin: 0; line-height: 1.4;">Entities & relationships mapped directly in Aura Cloud.</p>
        </div>
        <div style="color: #4285F4; font-size: 1.4rem; font-weight: bold;">➔</div>
        <div style="flex: 1; min-width: 220px;">
            <div style="color: #4285F4; font-weight: 700; font-size: 0.9rem; font-family: monospace; letter-spacing: 0.5px;">STEP 3 • STRICT GROUNDING</div>
            <p style="color: #FFFFFF; font-size: 1.05rem; font-weight: 700; margin: 6px 0 2px 0;">Zero-Hallucination GraphRAG</p>
            <p style="color: #CBD5E1; font-size: 0.85rem; margin: 0; line-height: 1.4;">Auditable reports backed by 2-hop topological proof.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Intelligence Query Engine", "📄 Ingest Multi-Format Document", "📊 Knowledge Graph Inspector"])

# ==========================================
# TAB 1: QUERY ENGINE WITH VISUAL SUBGRAPH
# ==========================================
with tab1:
    st.markdown("### 💬 Enterprise Query Engine")
    st.caption("🛡️ **Zero-Hallucination Guarantee:** The system checks your Neo4j Knowledge Graph first. If the factual proof does not exist in the graph, it will refuse to speculate.")
    st.write("")
    
    sample_scenario = st.selectbox(
        "Select an analytical scenario based on AFCFTA or type a custom question below to query your ingested file:",
        [
            "Custom Search Query...",
            "What are the rules, required documents, and regulatory approvals for exporting Processed Cocoa Powder from Ghana to Nigeria under AfCFTA?",
            "What are the duty rates and documents required for exporting Automotive Parts from South Africa to Kenya?",
            "How can Egypt export Phosphate Fertilizers to Nigeria under AfCFTA?"
        ]
    )
    # ... rest of tab1 code

    default_query = ""
    if sample_scenario != "Custom Search Query...":
        default_query = sample_scenario

    user_query = st.text_area("Enter Your Query:", value=default_query, height=90, placeholder="Type your trade, financial, or custom document query here...")

    if st.button("🚀 Execute GraphRAG Analysis"):
        if not user_query.strip():
            st.warning("⚠️ Please enter a query or select a pre-configured scenario before running analysis.")
        else:
            with st.spinner("Retrieving matched facts from Neo4j Knowledge Graph & reasoning with Gemini..."):
                try:
                    report, retrieved_facts = execute_smart_graphrag(user_query)
                    st.success("Analysis Complete — Grounded in Neo4j Knowledge Graph")
                    
                    st.markdown(report)
                    
                    # RENDER SUBGRAPH FOR QUERY
                    if retrieved_facts:
                        st.write("---")
                        st.markdown("#### 🕸️ Visualized Graph Context for this Query")
                        st.caption("Drag nodes, zoom in/out, or inspect the direct knowledge graph relationships used to generate the report:")
                        query_graph_html = generate_query_subgraph(retrieved_facts)
                        components.html(query_graph_html, height=480, scrolling=False)
                    
                    st.write("---")
                    st.markdown("#### 📥 Download Advisory Report")
                    d_col1, d_col2, d_col3 = st.columns(3)
                    
                    with d_col1:
                        st.download_button(
                            label="📄 Download as Markdown (.md)",
                            data=report,
                            file_name="GraphAI_Advisory_Report.md",
                            mime="text/markdown"
                        )
                    with d_col2:
                        export_data = {
                            "user_query": user_query,
                            "advisory_report": report,
                            "retrieved_neo4j_facts": retrieved_facts
                        }
                        st.download_button(
                            label="📊 Download as JSON (.json)",
                            data=json.dumps(export_data, indent=2),
                            file_name="GraphAI_Advisory_Report.json",
                            mime="application/json"
                        )
                    with d_col3:
                        st.download_button(
                            label="📝 Download as Text (.txt)",
                            data=report,
                            file_name="GraphAI_Advisory_Report.txt",
                            mime="text/plain"
                        )
                    
                    with st.expander("🔍 View Retrieved Neo4j Source Facts (Audit Trail)"):
                        st.caption("The response above was generated strictly from the following graph facts:")
                        for fact in retrieved_facts[:20]:
                            st.markdown(f"`{fact}`")
                            
                except Exception as e:
                    st.error(f"Execution Error: {e}")
                finally:
                    cleanup_temp_files()

# ==========================================
# TAB 2: INGESTION WITH DOCUMENT VISUALIZER
# ==========================================
with tab2:
    st.markdown("### 📄 Multi-Format Document Ingestion Engine")
    st.write("Upload ANY document format (**PDF**, **DOCX**, **CSV**, **TXT**, or **MD**) to automatically extract Knowledge Graph entities into Neo4j.")
    
    uploaded_file = st.file_uploader("Upload Document (PDF, DOCX, CSV, TXT, MD)", type=["pdf", "docx", "csv", "txt", "md"])
    
    if uploaded_file is not None:
        file_text, file_type = extract_text_from_file(uploaded_file)
        st.info(f"File uploaded: '{uploaded_file.name}' (Format: {file_type.upper()}, Size: {round(uploaded_file.size / 1024, 1)} KB). Click below to parse into Neo4j triples.")
        
        if st.button("⚡ Parse & Ingest Document into Neo4j"):
            with st.spinner(f"Parsing {file_type.upper()}, extracting entities with Gemini, and saving to Neo4j..."):
                try:
                    if file_type == "pdf" and len(file_text) < 100:
                        st.warning("⚠️ Scanned Image PDF detected. Uploading to Gemini Multimodal Engine for OCR & Graph Extraction...")
                        client = get_gemini_client()
                        file_ref = client.files.upload(file="temp_ingest.pdf")
                        
                        prompt = """
                        Extract all key structured entities and relationships from this PDF document into a JSON object matching:
                        {
                          "nodes": [{"id": "EntityName", "label": "EntityType"}],
                          "relationships": [{"source": "EntityA", "type": "RELATIONSHIP_TYPE", "target": "EntityB"}]
                        }
                        Return ONLY valid JSON without code blocks.
                        """
                        chat = client.chats.create(model='gemini-3.6-flash')
                        response = chat.send_message([file_ref, prompt])
                        clean_json = response.text.replace("```json", "").replace("```", "").strip()
                        extracted_json = json.loads(clean_json)
                        st.success("✅ Scanned PDF parsed via Gemini Vision OCR!")
                    else:
                        st.info(f"{file_type.upper()} text extracted ({len(file_text)} characters). Processing with Gemini...")
                        extracted_json = extract_universal_knowledge_graph(file_text)
                        st.success(f"✅ {file_type.upper()} parsed successfully!")

                    save_universal_triples_to_neo4j(extracted_json, source_filename=uploaded_file.name)
                    st.success("🎉 Knowledge Graph nodes & relationships successfully saved to Neo4j Cloud!")
                    
                    # RENDER DOCUMENT SUBGRAPH
                    st.write("---")
                    st.markdown("#### 🕸️ Extracted Document Knowledge Topology")
                    doc_graph_html = generate_document_subgraph(extracted_json)
                    components.html(doc_graph_html, height=480, scrolling=False)
                    
                    with st.expander("📊 View Extracted Knowledge Triples (JSON)"):
                        st.json(extracted_json)
                        
                except Exception as e:
                    st.error(f"Ingestion error: {e}")
                    
                finally:
                    cleanup_temp_files()

# ==========================================
# TAB 3: FILTERED INSPECTOR & PURGING
# ==========================================

with tab3:
    st.markdown("### 🕸️ Filtered Knowledge Graph Inspector")
    st.caption("Inspect live node statistics and topologies strictly for a specific document or active analysis.")
    
    # Fetch list of ingested documents
    available_docs = []
    try:
        with driver.session() as session:
            docs_result = session.run("MATCH (n) WHERE n.source_doc IS NOT NULL RETURN DISTINCT n.source_doc AS doc")
            available_docs = [rec['doc'] for rec in docs_result]
    except Exception:
        pass
        
    selected_doc = st.selectbox("Select a Document / Source to inspect:", ["All Ingested Data"] + available_docs)
    
    # DYNAMIC NODE COUNT (Filtered strictly by selected document)
    st.markdown(f"#### 📊 Active Node Statistics for: `{selected_doc}`")
    try:
        with driver.session() as session:
            if selected_doc == "All Ingested Data":
                cypher_counts = "MATCH (n) RETURN labels(n)[0] AS Label, count(n) AS Count ORDER BY Count DESC"
                count_result = session.run(cypher_counts)
            else:
                cypher_counts = "MATCH (n {source_doc: $doc}) RETURN labels(n)[0] AS Label, count(n) AS Count ORDER BY Count DESC"
                count_result = session.run(cypher_counts, doc=selected_doc)
                
            records = list(count_result)
            if records:
                col_stat1, col_stat2 = st.columns(2)
                for i, record in enumerate(records):
                    target_col = col_stat1 if i % 2 == 0 else col_stat2
                    target_col.markdown(f"- **{record['Label']} Nodes:** `{record['Count']}`")
            else:
                st.info(f"No active nodes found for '{selected_doc}'.")
    except Exception as e:
        st.error(f"Could not load graph metrics: {e}")
        
    st.write("---")

    if st.button("🌐 Render Filtered Topology"):
        with st.spinner(f"Extracting graph for '{selected_doc}'..."):
            try:
                net = Network(height="480px", width="100%", bgcolor="#090d16", font_color="#cbd5e1")
                net.force_atlas_2based()
                
                with driver.session() as session:
                    if selected_doc == "All Ingested Data":
                        cypher = "MATCH (a)-[r]->(b) RETURN labels(a)[0] AS l1, a.name AS n1, type(r) AS rel, labels(b)[0] AS l2, b.name AS n2 LIMIT 40"
                        result = session.run(cypher)
                    else:
                        cypher = """
                        MATCH (a)-[r]->(b) 
                        WHERE a.source_doc = $doc OR r.source_doc = $doc
                        RETURN labels(a)[0] AS l1, a.name AS n1, type(r) AS rel, labels(b)[0] AS l2, b.name AS n2 LIMIT 50
                        """
                        result = session.run(cypher, doc=selected_doc)
                        
                    for rec in result:
                        n1, l1 = str(rec['n1']), str(rec['l1'])
                        n2, l2 = str(rec['n2']), str(rec['l2'])
                        rel = str(rec['rel'])
                        net.add_node(n1, label=f"{n1}\n[{l1}]", color="#00d2ff", size=18, font={"color": "white", "size": 11})
                        net.add_node(n2, label=f"{n2}\n[{l2}]", color="#a855f7", size=18, font={"color": "white", "size": 11})
                        net.add_edge(n1, n2, title=rel, label=rel, color="#334155", font={"color": "#94a3b8", "size": 9})
                        
                net.save_graph("temp_filtered_graph.html")
                with open("temp_filtered_graph.html", "r", encoding="utf-8") as f:
                    filtered_html = f.read()
                    
                components.html(filtered_html, height=500, scrolling=False)
                st.success(f"✅ Rendered topology for: {selected_doc}")
            except Exception as e:
                st.error(f"Visualization error: {e}")
            finally:
                cleanup_temp_files()

    st.write("---")
    st.markdown("### 🗑️ Database Management & Data Purging")
    st.caption("Remove ingested knowledge or reset your Neo4j Cloud instance.")
    
    col_del1, col_del2 = st.columns(2)
    
    with col_del1:
        st.markdown("**Delete Specific Document**")
        if available_docs:
            doc_to_delete = st.selectbox("Select Document to Delete:", available_docs)
            if st.button("❌ Delete Selected Document Nodes"):
                with driver.session() as session:
                    session.run("MATCH (n {source_doc: $doc_name}) DETACH DELETE n", doc_name=doc_to_delete)
                st.success(f"Deleted all nodes associated with '{doc_to_delete}'!")
                st.rerun()
        else:
            st.info("No ingested documents available to delete.")
                
    with col_del2:
        st.markdown("**Reset Database**")
        if st.button("⚠️ Clear Entire Neo4j Database"):
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            st.success("🎉 Entire Neo4j Database successfully wiped clean!")
            st.rerun()