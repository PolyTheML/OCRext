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


def extract_table_with_gemini(data, model_name="gemini-1.5-flash-latest", large_table_mode=False):
    """
    Uses a selected Gemini model to extract a table from text or an image.
    """
    if large_table_mode:
        prompt_template = """
You are a specialized document parser. Extract ALL tables from this page with complete accuracy.

## LARGE TABLE MODE - Optimized for 50+ rows

### Core Requirements
- **EXTRACT EVERYTHING**: Every row, every column - COMPLETENESS is critical
- **PRESERVE NUMBERS**: Keep exact formatting but remove commas if needed to save space
- **ABBREVIATE HEADERS**: Use short, clear column names

### Space-Optimized Rules
- Remove unnecessary spaces and line breaks from data
- Abbreviate long column headers: "Agriculture,Forestry andFishing" → "Agriculture"
- Keep numbers exact but remove thousand separators if space-critical
- Use "-" for empty cells consistently

### Output Format - CRITICAL
Return ONLY a JSON object (NOT an array) with this exact structure:
{
    "table_data": [
        ["No", "Bank", "Total", "Agriculture", "Mining", "Manufacturing", "Utilities", "Construction", "Wholesale", "Retail", "Accommodation", "Arts", "Transport", "Information", "RealEstate1", "RealEstate2", "Education", "Health", "Households", "Other"],
        ["1", "ACLEDA Bank", "27934459", "5722614", "201477", "740254", "94281", "1446649", "1785014", "7502966", "1699046", "55445", "1269974", "11621", "171547", "1065967", "60991", "293008", "4145764", "1667641"]
    ],
    "rows": 52
}

CRITICAL: 
- Return ONLY the JSON object, no arrays around it
- Extract ALL visible rows
- Priority: COMPLETENESS over formatting details
"""
    else:
        prompt_template = """
You are a specialized document parser. Extract ALL tables from this page with complete accuracy.

## Standard Mode Requirements
- **EXTRACT EVERYTHING**: Every table, every row, every column, every cell
- **PRESERVE EXACTLY**: All numbers, formatting, punctuation, and text as shown
- **NO INTERPRETATION**: Extract exactly what you see, don't convert or standardize

## Output Format - CRITICAL
Return ONLY a JSON object (NOT an array) with this exact structure:
{
    "table_data": [
        ["Header1", "Header2", "Header3"],
        ["Row1Col1", "Row1Col2", "Row1Col3"]
    ],
    "total_rows_extracted": 2,
    "confidence_score": 0.95,
    "extraction_notes": "Complete extraction",
    "tables_found": 1
}

CRITICAL: Return ONLY the JSON object, no arrays around it.
Begin extraction now.
"""

    model_input = [prompt_template]
    
    # Add text if available
    if data.get('text') and data['text'].strip():
        model_input.append(f"Text content from page:\n{data['text']}")
    
    # Add image if available
    if data.get("image"):
        model_input.append(data['image'])

    try:
        model = genai.GenerativeModel(model_name)
        
        # Enhanced generation config based on mode
        if large_table_mode:
            generation_config = {
                "response_mime_type": "application/json",
                "max_output_tokens": 8192,
                "temperature": 0.0  # Most deterministic for large tables
            }
        else:
            generation_config = {
                "response_mime_type": "application/json",
                "max_output_tokens": 4096,
                "temperature": 0.1
            }
        
        response = model.generate_content(
            model_input,
            generation_config=generation_config
        )
        
        # Check if response was truncated
        response_text = response.text.strip()
        is_truncated = False
        
        # Fix malformed JSON from repair attempts
        if response_text.count('}') > response_text.count('{'):
            st.warning("⚠️ Detected malformed JSON, cleaning up...")
            # Remove extra closing brackets
            while response_text.endswith('}}') and response_text.count('}') > response_text.count('{'):
                response_text = response_text[:-1]
        
        if not response_text.endswith('}') or response_text.count('{') != response_text.count('}'):
            st.warning("⚠️ Response appears to be truncated. Attempting to repair JSON...")
            is_truncated = True
            
            # Attempt to repair truncated JSON
            try:
                # Remove any partial repair attempts first
                if ',"rows":' in response_text:
                    response_text = response_text.split(',"rows":')[0]
                
                # Find the last complete row by looking for the last complete array
                lines = response_text.split('\n')
                last_complete_row = -1
                
                for i in range(len(lines) - 1, -1, -1):
                    line = lines[i].strip()
                    if line.endswith('],') or line.endswith(']'):
                        # Found a complete row
                        last_complete_row = i
                        break
                
                if last_complete_row > 0:
                    # Reconstruct JSON up to last complete row
                    reconstructed_lines = lines[:last_complete_row + 1]
                    
                    # Fix the last line - remove trailing comma if present
                    if reconstructed_lines[-1].strip().endswith('],'):
                        reconstructed_lines[-1] = reconstructed_lines[-1].replace('],', ']')
                    
                    base_json = '\n'.join(reconstructed_lines)
                    
                    # Count actual data rows (excluding header)
                    row_count = base_json.count('[') - 2  # Subtract 2: one for main array, one for header
                    if row_count < 0:
                        row_count = 0
                    
                    # Close the JSON properly
                    if large_table_mode:
                        response_text = base_json + f'\n],"rows":{row_count + 1}}}'  # +1 to include header
                    else:
                        response_text = base_json + f'\n],"total_rows_extracted":{row_count + 1},"confidence_score":0.8,"extraction_notes":"Partially extracted due to size limits","tables_found":1}}'
                        
                    st.info(f"✅ Repaired JSON - extracted {row_count + 1} rows (including header)")
                else:
                    st.error("Could not find any complete rows to repair JSON")
                    return {"error": "Response was truncated and could not be repaired"}
                    
            except Exception as repair_error:
                st.error(f"Could not repair truncated JSON: {repair_error}")
                return {"error": f"Response was truncated and could not be repaired: {repair_error}"}
        
        # Debug info
        if large_table_mode:
            st.write(f"Debug - Response length: {len(response_text)} chars")
            st.write("Debug - Response start:", response_text[:300])
            st.write("Debug - Response end:", response_text[-200:])
        
        # Parse JSON response
        try:
            table_data = json.loads(response_text)
            
            # Handle case where model returns array instead of object
            if isinstance(table_data, list) and len(table_data) > 0:
                if isinstance(table_data[0], dict) and "table_data" in table_data[0]:
                    table_data = table_data[0]  # Extract the first object from array
                    st.info("Fixed: Model returned array instead of object")
                
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract just the object part
            st.warning("JSON parsing failed, attempting to extract object...")
            
            # Look for the first { and last } to extract the object
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            
            if first_brace >= 0 and last_brace >= 0 and last_brace > first_brace:
                try:
                    object_json = response_text[first_brace:last_brace + 1]
                    table_data = json.loads(object_json)
                    st.info("Successfully extracted object from malformed JSON")
                except:
                    st.error(f"JSON parsing error: {e}")
                    st.write("Raw response that failed to parse:", response_text[:1000])
                    return {"error": f"Failed to parse model response as JSON: {e}"}
            else:
                st.error(f"JSON parsing error: {e}")
                st.write("Raw response that failed to parse:", response_text[:1000])
                return {"error": f"Failed to parse model response as JSON: {e}"}
        
        # Validate the response structure
        if not isinstance(table_data, dict):
            return {"error": "Invalid response format from model"}
        
        # Check if we have the expected structure
        if "table_data" not in table_data:
            return {"error": "Missing 'table_data' in model response"}
        
        # Add metadata if missing (for compact format)
        if "total_rows_extracted" not in table_data:
            table_data["total_rows_extracted"] = len(table_data["table_data"])
        if "confidence_score" not in table_data:
            table_data["confidence_score"] = 0.8 if is_truncated else 0.9
        if "extraction_notes" not in table_data:
            if is_truncated:
                table_data["extraction_notes"] = f"Extracted {len(table_data['table_data'])} rows (truncated response repaired)"
            else:
                table_data["extraction_notes"] = f"Extracted {len(table_data['table_data'])} rows"
        if "tables_found" not in table_data:
            table_data["tables_found"] = 1
        
        return table_data
        
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        st.write("Raw response that failed to parse:", response_text[:1000] if 'response_text' in locals() else response.text[:1000])
        return {"error": f"Failed to parse model response as JSON: {e}"}
    except Exception as e:
        st.error(f"Model error: {e}")
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
    
    # --- Model Selector ---
    model_choice = st.selectbox(
        "Choose AI Model",
        ("gemini-1.5-pro-latest", "gemini-2.0-flash-exp", "gemini-1.5-flash-latest"),
        help="Pro has higher token limits for large tables. Flash is faster but may truncate large tables."
    )
    
    # --- Large Table Handling ---
    handle_large_tables = st.checkbox(
        "Large Table Mode", 
        value=True,
        help="Use optimized settings for tables with 50+ rows"
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
                # 2. Use Gemini to extract and structure the table
                if handle_large_tables:
                    st.info("🔄 Large table mode enabled - using optimized extraction...")
                
                structured_table = extract_table_with_gemini(
                    extracted_data, 
                    model_name=model_choice,
                    large_table_mode=handle_large_tables
                )
                st.session_state.result = structured_table
            else:
                st.warning("Could not extract any meaningful data from the PDF page.")
                st.session_state.result = None

    # --- Display Results ---
    if st.session_state.result:
        result = st.session_state.result
        
        # Check for errors first
        if isinstance(result, dict) and "error" in result:
            st.error(f"An error occurred: {result['error']}")
        elif isinstance(result, dict) and "table_data" in result:
            # Handle the new JSON structure
            table_data = result["table_data"]
            
            if not table_data or len(table_data) == 0:
                st.warning("No tables found on the specified page.")
                st.info("Extraction details:")
                st.json(result)
            else:
                st.success("Table extracted successfully!")
                
                # Display extraction metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tables Found", result.get("tables_found", 1))
                with col2:
                    st.metric("Rows Extracted", result.get("total_rows_extracted", len(table_data)))
                with col3:
                    st.metric("Confidence", f"{result.get('confidence_score', 0):.1%}")
                
                if result.get("extraction_notes"):
                    st.info(f"Notes: {result['extraction_notes']}")
                
                # Display the table
                st.subheader("Extracted Data")
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)

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
                
                # Show raw JSON for debugging
                with st.expander("Show Raw JSON Response"):
                    st.json(result)
        else:
            st.warning("Unexpected response format from the model.")
            st.json(result)
else:
    st.info("Please upload a PDF file to get started.")