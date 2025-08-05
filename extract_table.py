import streamlit as st
import pandas as pd
import os
import fitz  # PyMuPDF
import google.generativeai as genai
import json
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Table Extractor with Gemini",
    page_icon="📄",
    layout="wide"
)

# --- Gemini API Configuration ---
# Using Streamlit's secrets management
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=API_KEY)
    st.sidebar.success("API key loaded successfully!")
except (KeyError, FileNotFoundError):
    st.error("🚨 GOOGLE_API_KEY not found in Streamlit secrets.")
    st.info("Please add your API key to the .streamlit/secrets.toml file and restart the app.")
    st.stop()


# --- Core Functions (from the original script) ---

def extract_text_from_pdf(pdf_stream, page_number=0):
    """
    Extracts raw text and/or an image from a specific page of a PDF stream.
    
    Returns:
        A dictionary with 'text', 'image', and 'error' data.
    """
    try:
        # Open PDF from a byte stream
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
    except Exception as e:
        return {"error": f"Error opening PDF stream: {e}"}
        
    if page_number >= len(doc):
        return {"error": f"Invalid page number: {page_number}. PDF has only {len(doc)} pages."}
        
    page = doc.load_page(page_number)
    
    text = page.get_text("text")
    image = None
    
    # Heuristic to detect if the page is primarily an image
    if len(text.strip()) < 150: 
        st.info("Page has minimal text. Rendering as an image for multimodal analysis.")
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
    else:
        st.info("Successfully extracted text directly from PDF.")

    return {"text": text, "image": image, "error": None}

def extract_table_with_gemini(data):
    """
    Uses a modern Gemini model to extract a table from text or an image.
    """
    prompt_template = """
    You are an expert data extraction assistant. Your task is to extract tabular data from the provided text or image.
    Analyze the input and identify the main table. Extract all rows accurately.
    The desired output format is a clean JSON array where each object represents a row.
    Use the table's column headers as the keys for each JSON object.
    IMPORTANT INSTRUCTIONS:
    - If a cell is empty, represent its value as an empty string "".
    - Handle multi-line text within a single cell correctly.
    - Do not include any text outside of the table (like titles or footnotes).
    - Do not include the table headers as a data row.
    - If you cannot find a table, return an empty JSON array [].
    - Respond ONLY with the JSON array, without any additional explanation or markdown formatting.
    """
    model_name = "gemini-1.5-flash-latest"
    model_input = [prompt_template, data['text']]

    if data.get("image"):
        model_input.append(data['image'])

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            model_input,
            generation_config={"response_mime_type": "application/json"}
        )
        table_data = json.loads(response.text)
        return table_data
    except Exception as e:
        return {"error": f"An error occurred while processing with the model: {e}"}


# --- Streamlit App UI ---

st.title("📄 PDF Table Extractor with Gemini")
st.write("Upload a PDF, select a page, and let AI extract the tables for you.")

# --- Sidebar for Instructions and Upload ---
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1.  Make sure your `GOOGLE_API_KEY` is set in your Streamlit secrets (`.streamlit/secrets.toml`).
    2.  Upload the PDF file containing the table you want to extract.
    3.  Enter the page number where the table is located.
    4.  Click the **'Extract Table'** button.
    5.  View the results and download the data as a CSV or Excel file.
    """)
    
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    page_number = st.number_input(
        "Enter Page Number", 
        min_value=1, 
        value=1, 
        help="Enter the page number containing the table (starting from 1)."
    )

# --- Main App Logic ---
if uploaded_file is not None:
    # Initialize session state to store results
    if 'result' not in st.session_state:
        st.session_state.result = None

    if st.button("✨ Extract Table", type="primary"):
        with st.spinner("Processing... Reading PDF and calling Gemini..."):
            pdf_stream = uploaded_file.read()
            
            # 1. Extract raw data from the PDF
            extracted_data = extract_text_from_pdf(pdf_stream, page_number - 1) # Adjust to 0-index
            
            if extracted_data.get("error"):
                st.error(extracted_data["error"])
                st.session_state.result = None
            elif extracted_data['text'] or extracted_data['image']:
                # 2. Use Gemini to extract and structure the table
                structured_table = extract_table_with_gemini(extracted_data)
                st.session_state.result = structured_table # Store result in session state
            else:
                st.warning("Could not extract any meaningful data from the PDF page.")
                st.session_state.result = None

    # --- Display Results ---
    if st.session_state.result:
        result = st.session_state.result
        st.success("Table extracted successfully!")
        
        if isinstance(result, dict) and "error" in result:
            st.error(f"An error occurred: {result['error']}")
        elif isinstance(result, list) and len(result) > 0:
            st.subheader("Extracted Data (as Table)")
            df = pd.DataFrame(result)
            st.dataframe(df)

            # --- Download Buttons ---
            col1, col2 = st.columns(2)
            
            # Convert DataFrame to CSV
            csv = df.to_csv(index=False).encode('utf-8')
            col1.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_page_{page_number}.csv",
                mime="text/csv",
            )

            # Convert DataFrame to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            excel_data = output.getvalue()
            col2.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_page_{page_number}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            
            st.subheader("Raw JSON Output")
            st.json(result)
        else:
            st.warning("The model did not find a table on the specified page or the result was empty.")
else:
    st.info("Please upload a PDF file to get started.")

