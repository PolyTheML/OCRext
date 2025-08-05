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
    Extracts raw text and a correctly oriented image from a specific page of a PDF stream.
    This approach is robust against rotated pages and complex layouts.
    
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
    
    # --- Enhanced Rotation Handling ---
    # We will now always render the page to an image to ensure the layout is preserved,
    # which is critical for rotated pages and complex tables.
    
    rotation_angle = page.rotation
    
    if rotation_angle != 0:
        st.info(f"Detected page rotation of {rotation_angle} degrees. Correcting orientation...")
        # Create a matrix to rotate the page to be upright
        rotation_matrix = fitz.Matrix(1, 0, 0, 1, 0, 0).prerotate(rotation_angle)
    else:
        rotation_matrix = fitz.Identity # No rotation needed
    
    # Render the pixmap using the rotation matrix for a correctly oriented image
    pix = page.get_pixmap(dpi=300, matrix=rotation_matrix)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    st.info("Rendered page as an image to ensure accurate layout analysis.")

    # Also extract text to provide additional context to the model
    text = page.get_text("text")
    if not text.strip():
        st.info("No text layer found on this page. Relying solely on image analysis.")

    return {"text": text, "image": image, "error": None}


def extract_table_with_gemini(data, model_name="gemini-1.5-flash-latest"):
    """
    Uses a selected Gemini model to extract a table from text or an image.
    """
    prompt_template = """Creates the ultimate prompt for financial table extraction based on your requirements."""
    return """
You are a specialized financial document parser. Extract ALL tables from this page with complete accuracy.

## Core Requirements
- **EXTRACT EVERYTHING**: Every table, every row, every column, every cell
- **PRESERVE EXACTLY**: All numbers, formatting, punctuation, and text as shown
- **NO INTERPRETATION**: Extract exactly what you see, don't convert or standardize

## Critical Rules

### 1. Complete Extraction
- Scan entire page systematically (left-to-right, top-to-bottom)
- Extract tables regardless of orientation (portrait/landscape)
- Include ALL rows: headers, data, subtotals, totals, footnotes within table structure
- Count and verify: output row count MUST match source

### 2. Value Preservation
- **Numbers**: Keep exact formatting: "1,361,196", "(2,207)", "2.5%", "-"
- **Text**: Preserve case, spacing, special characters
- **Empty cells**: Use empty string "" (not null, "0", or "-" unless actually shown)
- **Merged cells**: Repeat the value for all positions it spans

### 3. Header Processing
- **Detection**: First row with text/labels = headers
- **Cleaning**: Trim whitespace, preserve original language and case
- **Duplicates**: Append "_2", "_3" etc: "Amount", "Amount_2", "Amount_3"
- **Missing**: Generate "Column_1", "Column_2" if no clear headers

### 4. Multi-language Support
- **Mixed content**: Extract exactly as shown (don't translate)
- **Bank names**: Keep full original text including English/Khmer
- **Special characters**: Preserve all Unicode characters
- **Numbers in text**: Keep as part of the string

### 5. Complex Structure Handling
- **Multi-level headers**: Combine with underscore: "2024_Deposits", "2023_Loans"
- **Rotated tables**: Read following the text orientation
- **Split tables**: If table continues across sections, treat as separate tables
- **Nested data**: Extract hierarchical info maintaining structure

## Output Format - CRITICAL
Return ONLY a valid JSON object with this exact structure:
```json
{
    "table_data": [
        ["Header1", "Header2", "Header3"],
        ["Row1Col1", "Row1Col2", "Row1Col3"],
        ["Row2Col1", "Row2Col2", "Row2Col3"]
    ],
    "total_rows_extracted": 52,
    "confidence_score": 0.95,
    "extraction_notes": "Extracted complete banking table with all financial metrics"
}
```

## Validation Checklist
Before returning results, verify:
- [ ] Every visible table identified
- [ ] Row counts match exactly
- [ ] All numbers preserved with original formatting
- [ ] No data invented or modified
- [ ] Headers are appropriate
- [ ] All text readable and preserved

## Example Transformation
**Source row**: `Advanced Bank of Asia Limited | 43,051,336 | 34,281,391 | 79.6%`
**JSON output**: 
```json
["Advanced Bank of Asia Limited", "43,051,336", "34,281,391", "79.6%"]
```

**CRITICAL INSTRUCTIONS**:
1. Return ONLY the JSON object - no markdown, no explanations, no code blocks
2. Extract EVERY visible row in the table
3. Maintain exact numerical formatting
4. Financial accuracy is critical - double-check all numerical values
5. If you see 50+ rows, extract all 50+ rows

Begin extraction now.
"""
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
st.write("Upload a PDF, select a page and model, and let AI extract the tables for you.")

# --- Sidebar for Instructions and Upload ---
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1.  Make sure your `GOOGLE_API_KEY` is set in your Streamlit secrets.
    2.  Upload the PDF file.
    3.  Select the AI model to use.
    4.  Enter the page number of the table.
    5.  Click **'Extract Table'**.
    """)
    
    st.header("Settings")
    
    # --- NEW: Model Selector ---
    model_choice = st.selectbox(
        "Choose AI Model",
        ("gemini-2.5-pro", "gemini-2.0-flash-exp"),
        help="Flash is faster and cheaper, while Pro is more powerful for complex tables."
    )
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
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
        with st.spinner(f"Processing with {model_choice}... Reading PDF and calling Gemini..."):
            pdf_stream = uploaded_file.read()
            
            # 1. Extract raw data from the PDF
            extracted_data = extract_text_from_pdf(pdf_stream, page_number - 1) # Adjust to 0-index
            
            if extracted_data.get("error"):
                st.error(extracted_data["error"])
                st.session_state.result = None
            elif extracted_data['text'] or extracted_data['image']:
                # 2. Use Gemini to extract and structure the table, passing the selected model
                structured_table = extract_table_with_gemini(extracted_data, model_name=model_choice)
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
