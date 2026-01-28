"""
Form Field Detection with Sobel Edge Filter
This script applies Sobel magnitude filter to the PDF image before
sending it to the LLM for improved field detection.
"""

import os 
import time
import base64
import json
import fitz  # PyMuPDF
import asyncio
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor


# Load environment variables
load_dotenv()


def pdf_to_image(pdf_path, page_num=0, dpi=300):
    """
    Convert a PDF page to a PIL Image.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Get PDF dimensions for coordinate conversion
    pdf_width, pdf_height = page.rect.width, page.rect.height
    
    doc.close()
    return img, pdf_width, pdf_height


# =============================================================================
# Image Preprocessing Functions
# =============================================================================

def convert_to_grayscale(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to grayscale numpy array.
    
    Args:
        image: PIL Image (RGB or grayscale)
        
    Returns:
        Grayscale numpy array
    """
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return img_array


def apply_gaussian_blur(gray_image: np.ndarray, kernel_size: tuple = (7, 7)) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise in the image.
    
    Args:
        gray_image: Grayscale numpy array
        kernel_size: Size of the Gaussian kernel (must be odd numbers)
        
    Returns:
        Blurred grayscale numpy array
    """
    return cv2.GaussianBlur(gray_image, kernel_size, 0)


# =============================================================================
# Edge Detection Functions
# =============================================================================

def compute_sobel_magnitude(blurred_image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply Sobel filter in X and Y directions and compute gradient magnitude.
    
    Args:
        blurred_image: Preprocessed grayscale numpy array
        ksize: Size of the Sobel kernel
        
    Returns:
        Normalized Sobel magnitude image (0-255 range, uint8)
    """
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=ksize)
    
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    return cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def threshold_to_binary(magnitude_image: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Convert magnitude image to binary for contour detection.
    
    Args:
        magnitude_image: Sobel magnitude image (uint8)
        threshold: Threshold value for binarization
        
    Returns:
        Binary image (0 or 255 values)
    """
    _, binary = cv2.threshold(magnitude_image, threshold, 255, cv2.THRESH_BINARY)
    return binary


# =============================================================================
# Contour Detection Functions
# =============================================================================

def find_rectangular_contours(
    binary_image: np.ndarray, 
    min_area: int = 50, 
    max_area: int = 500000,
    epsilon_factor: float = 0.02
) -> list:
    """
    Find contours that approximate to quadrilaterals (4 vertices).
    
    Args:
        binary_image: Binary image for contour detection
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
        epsilon_factor: Factor for contour approximation (smaller = more precise)
        
    Returns:
        List of bounding rectangles as tuples (x, y, w, h)
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    rectangles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
        
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            rectangles.append((x, y, w, h))
    
    print(f"Found {len(contours)} total contours")
    print(f"Found {len(rectangles)} rectangular shapes (4 edges)")
    
    return rectangles


def find_horizontal_lines(
    binary_image: np.ndarray, 
    min_width: int = 80, 
    max_height: int = 12,
    min_aspect_ratio: float = 10.0
) -> list:
    """
    Find elongated horizontal contours (potential fillable underlines).
    
    Args:
        binary_image: Binary image for contour detection
        min_width: Minimum width for a line to be considered
        max_height: Maximum height (thickness) for a line
        min_aspect_ratio: Minimum width/height ratio (must be very elongated)
        
    Returns:
        List of bounding rectangles as tuples (x, y, w, h)
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if w < min_width:
            continue
        if h > max_height:
            continue
        
        aspect_ratio = w / max(1, h)
        if aspect_ratio < min_aspect_ratio:
            continue
        
        lines.append((x, y, w, h))
    
    print(f"Found {len(lines)} horizontal lines (potential fillable underlines)")
    
    return lines


# =============================================================================
# Visualization Functions
# =============================================================================

def draw_detected_regions(
    base_image: np.ndarray, 
    rectangles: list, 
    lines: list,
    rect_color: tuple = (0, 0, 255),
    line_color: tuple = (0, 255, 0),
    thickness: int = 8
) -> np.ndarray:
    """
    Draw bounding boxes on the image for detected regions.
    
    Args:
        base_image: Grayscale image to draw on (will be converted to BGR)
        rectangles: List of rectangle tuples (x, y, w, h) to draw in rect_color
        lines: List of line tuples (x, y, w, h) to draw in line_color
        rect_color: BGR color for rectangles (default: red)
        line_color: BGR color for lines (default: green)
        thickness: Line thickness for drawing
        
    Returns:
        BGR image with annotations
    """
    output = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    
    for (x, y, w, h) in rectangles:
        cv2.rectangle(output, (x, y), (x + w, y + h), rect_color, thickness)
    
    for (x, y, w, h) in lines:
        cv2.rectangle(output, (x, y), (x + w, y + h), line_color, thickness)
    
    return output


# =============================================================================
# Orchestrator Function
# =============================================================================

def detect_form_fields(
    image: Image.Image, 
    min_area: int = 50, 
    max_area: int = 500000, 
    detect_lines: bool = True
) -> Image.Image:
    """
    Main orchestrator function that detects form fields in an image.
    
    Applies Sobel magnitude filter to detect edges, finds rectangular contours
    (potential text fields/checkboxes) and horizontal lines (potential underlines),
    then returns an annotated image.
    
    Args:
        image: PIL Image to process
        min_area: Minimum contour area for rectangle detection
        max_area: Maximum contour area for rectangle detection
        detect_lines: Whether to detect horizontal lines
        
    Returns:
        PIL Image with Sobel magnitude filter, red rectangular boxes, and green underlines
    """
    # Step 1: Preprocessing
    gray = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    
    # Step 2: Edge detection
    sobel_magnitude = compute_sobel_magnitude(blurred)
    binary = threshold_to_binary(sobel_magnitude)
    
    # Step 3: Contour detection
    rectangles = find_rectangular_contours(binary, min_area, max_area)
    lines = find_horizontal_lines(binary) if detect_lines else []
    
    # Step 4: Visualization
    annotated = draw_detected_regions(sobel_magnitude, rectangles, lines)
    
    # Convert BGR to RGB before returning as PIL Image
    output_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output_rgb)


# Legacy function name for backward compatibility
def apply_sobel_magnitude(image, min_area=50, max_area=500000, detect_lines=True):
    """
    Legacy wrapper for detect_form_fields.
    Maintained for backward compatibility.
    """
    return detect_form_fields(image, min_area, max_area, detect_lines)


def image_to_base64(image):
    """
    Convert PIL Image to base64 string.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# =============================================================================
# API Client Functions
# =============================================================================

def create_llm_client():
    """
    Create and configure the LLM client from environment variables.
    
    Returns:
        tuple: (client, model_name) or (None, None) if credentials are missing
        
    Raises:
        ValueError: If required environment variables are not set
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    base_url = os.getenv("GOOGLE_API_BASE_URL")
    model_name = os.getenv("MODEL")
    
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable")
    if not base_url:
        raise ValueError("Missing GOOGLE_API_BASE_URL environment variable")
    if not model_name:
        raise ValueError("Missing MODEL environment variable")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name


def parse_pdf_coordinates_with_filter(pdf_path, page_num=0, output_prefix=None, client=None, model_name=None):
    """
    Converts PDF page to image, applies Sobel magnitude filter,
    and asks Gemini for bounding box coordinates.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to process (0-indexed)
        output_prefix: Prefix for the filtered image filename (e.g., "document_page_0")
        client: OpenAI client instance (optional, will create if not provided)
        model_name: Model name to use (optional, will get from env if not provided)
    """
    # Create client if not provided
    if client is None or model_name is None:
        try:
            client, model_name = create_llm_client()
        except ValueError as e:
            print(f"Error: {e}")
            return None

    # Convert PDF to image
    try:
        print(f"Converting PDF page {page_num} to image...")
        dpi = 300  # Use 300 DPI for better image quality
        img, pdf_width, pdf_height = pdf_to_image(pdf_path, page_num=page_num, dpi=dpi)
        print(f"Original image size: {img.size}")
        print(f"PDF dimensions: {pdf_width} x {pdf_height} points")
        
        # Apply Sobel magnitude filter (using refactored detect_form_fields)
        print("Applying Sobel magnitude filter...")
        filtered_img = detect_form_fields(img)
        
        # Save filtered image for debugging with unique name
        if output_prefix:
            filtered_filename = f"output/{output_prefix}_filter_applied.png"
        else:
            filtered_filename = "filter_applied_for_llm.png"
        filtered_img.save(filtered_filename)
        print(f"Saved filtered image: {filtered_filename}")
        
        # Convert to base64
        base64_image = image_to_base64(filtered_img)
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

    prompt = """
    You are an expert at analyzing forms.
    I have converted the first page of a PDF into an edge-detected image using Sobel filter.
  
    
    Please identify the bounding boxes for:
    1. Text input fields (rectangular boxes where a user would type text).
    2. Checkboxes (small square boxes where a user would tick/check).
    
    Look for:
    - Rectangular outlines that indicate text input fields
    - Small square outlines that indicate checkboxes
    - The edges are highlighted in white against a dark background
    - If there are red outlines, those indicate detected rectangles from the Sobel filter
    - If there are green lines, those indicate detected horizontal lines (potential fillable underlines)
    
    Return a valid JSON object with the following structure:
    {
      "fields": [
        {
          "color": "red" or "green",
          "type": "text_field" or "checkbox",
          "label": "inferred label text or description",
          "bbox": [ymin, xmin, ymax, xmax]
        }
      ]
    }
    
    - bbox should be normalized coordinates (0-1000 scale).
    - Format: [ymin, xmin, ymax, xmax] where (0,0) is top-left and (1000,1000) is bottom-right.
    - Be precise and identify all visible form fields.
    """

    try:
        print(f"Sending filtered image to model: {model_name}")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        content = response.choices[0].message.content
        print("\nResponse from Gemini (Coordinates):")
        print("-" * 20)
        print(content)
        print("-" * 20)
        
        # Parse JSON from response
        try:
            start_index = content.find('{')
            end_index = content.rfind('}')
            if start_index != -1 and end_index != -1:
                json_str = content[start_index:end_index+1]
                data = json.loads(json_str)
                print("Successfully parsed JSON.")
                
                # Apply scaling: convert normalized (0-1000) to PDF points
                print("Applying coordinate scaling...")
                for field in data.get('fields', []):
                    bbox = field['bbox']
                    if len(bbox) == 4:
                        ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
                        
                        # Scale normalized to PDF points
                        field['bbox'] = [
                            (xmin / 1000) * pdf_width,   # x0 in PDF points
                            (ymin / 1000) * pdf_height,  # y0 in PDF points
                            (xmax / 1000) * pdf_width,   # x1 in PDF points
                            (ymax / 1000) * pdf_height,  # y1 in PDF points
                        ]
                
                return data
            else:
                print("No JSON object found in response.")
                return content
        except json.JSONDecodeError:
            print("Failed to parse JSON from response.")
            return content

    except Exception as e:
        print(f"Error invoking Gemini: {e}")
        return None


def create_acroform_from_coordinates(pdf_path, json_path, output_path, output_filename=None):
    """
    Reads coordinates from JSON and adds AcroForm fields to the PDF.
    Coordinates in JSON are already in PDF points (no scaling needed).
    Removes existing AcroForms before adding new ones.
    Supports multi-page PDFs by using the 'page' field in JSON.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    # Check for existing AcroForms on all pages and remove them
    total_existing = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        existing_widgets = list(page.widgets())
        if existing_widgets:
            total_existing += len(existing_widgets)
            for widget in existing_widgets:
                page.delete_widget(widget)
    
    if total_existing > 0:
        print(f"Found {total_existing} existing form fields. Removed them.")
    else:
        print("No existing form fields found.")

    print(f"Adding {len(data['fields'])} new fields to PDF...")
    print("Coordinates are already in PDF points (no scaling needed)")

    for field in data['fields']:
        bbox = field['bbox']
        color = field.get('color', 'red').lower()
        page_num = field.get('page', 0)  # Default to page 0 if not specified
        
        # Get the correct page
        if page_num >= len(doc):
            print(f"Warning: Page {page_num} does not exist, skipping field {field.get('label', 'unknown')}")
            continue
        page = doc[page_num]
        
        if color == 'green':
            # Green lines: Create text_field with expanded height (18pt for text)
            x0, y0, x1, y1 = bbox
            
            # Expand height to accommodate text (minimum 18 points)
            min_height = 18
            current_height = y1 - y0
            # if current_height < min_height:
            #     y0 = y1 - min_height  # Expand upward from the line

            if current_height < min_height:
                mid = (y0 + y1) / 2
                y0 = mid - min_height / 2
                y1 = mid + min_height / 2
            
            rect = fitz.Rect(x0, y0, x1, y1)
            
            widget = fitz.Widget()
            widget.rect = rect
            widget.field_name = field['label']
            widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
            page.add_widget(widget)
            
        else:
            # Red (default): Keep current workflow
            rect = fitz.Rect(bbox)
            
            if field['type'] == 'text_field':
                widget = fitz.Widget()
                widget.rect = rect
                widget.field_name = field['label']
                widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                page.add_widget(widget)
            elif field['type'] == 'checkbox':
                widget = fitz.Widget()
                widget.rect = rect
                widget.field_name = field['label']
                widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
                widget.field_value = False
                page.add_widget(widget)

    # Use provided filename or default
    if output_filename:
        output_file = os.path.join(output_path, output_filename)
    else:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = os.path.join(output_path, f"{base_name}_interactive.pdf")
    
    doc.save(output_file)
    doc.close()
    print(f"Saved interactive PDF to: {output_file}")

# =============================================================================
# Thread pool for CPU-bound operations (OpenCV, PDF processing)
executor = ThreadPoolExecutor(max_workers=4)


def process_pdf_sync(
    pdf_path: str,
    output_dir: str,
    client,
    model_name: str
) -> str:
    """
    Synchronous function to process a single PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        client: OpenAI client instance
        model_name: Model name to use
        
    Returns:
        Path to the generated interactive PDF
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Get number of pages
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()
    
    print(f"Processing PDF: {base_name} ({num_pages} pages)")
    
    all_fields = {"fields": []}
    
    # Process each page
    for page_num in range(num_pages):
        print(f"Processing page {page_num + 1} of {num_pages}")
        output_prefix = f"{base_name}_page_{page_num}"
        
        data = parse_pdf_coordinates_with_filter(
            pdf_path,
            page_num=page_num,
            output_prefix=output_prefix,
            client=client,
            model_name=model_name
        )
        
        if isinstance(data, dict):
            for field in data.get('fields', []):
                field['page'] = page_num
            all_fields['fields'].extend(data.get('fields', []))
    
    # Save JSON
    json_file = os.path.join(output_dir, f"{base_name}.json")
    with open(json_file, 'w') as f:
        json.dump(all_fields, f, indent=2)
    print(f"Saved field coordinates to: {json_file}")
    
    # Create interactive PDF
    create_acroform_from_coordinates(pdf_path, json_file, output_dir)
    
    # Return path to interactive PDF
    interactive_pdf_path = os.path.join(output_dir, f"{base_name}_interactive.pdf")
    print(f"Created interactive PDF: {interactive_pdf_path}")
    
    return interactive_pdf_path


async def process_pdf_async(
    pdf_path: str,
    output_dir: str = "output/"
) -> str:
    """
    Async wrapper for PDF processing.
    Creates LLM client on-demand and runs processing in thread pool.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        
    Returns:
        Path to the generated interactive PDF
    """
    # Create LLM client on-demand (only when endpoint is called)
    print("Creating LLM client...")
    client, model_name = create_llm_client()
    print(f"Using model: {model_name}")
    
    # Run CPU-bound processing in thread pool
    loop = asyncio.get_event_loop()
    interactive_pdf_path = await loop.run_in_executor(
        executor,
        process_pdf_sync,
        pdf_path,
        output_dir,
        client,
        model_name
    )
    
    return interactive_pdf_path

# =============================================================================


if __name__ == "__main__":
    data_dir = r"data/"
    output_dir = r"output/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Form Field Detection with Sobel Edge Filter")
    print("Processing all PDF files in data directory")
    print("=" * 50)
    
    # Initialize LLM client once for all files (more efficient)
    try:
        client, model_name = create_llm_client()
        print(f"Initialized LLM client with model: {model_name}")
    except ValueError as e:
        print(f"Error initializing LLM client: {e}")
        exit(1)
    
    # Get all PDF files in data directory
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    print(f"\nFound {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        
        print(f"\n{'='*50}")
        print(f"Processing: {pdf_file}")
        print(f"{'='*50}")
        
        try:
            # Get the number of pages in the PDF
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()
            print(f"PDF has {num_pages} page(s)")
            
            # Process each page
            all_fields = {"fields": []}
            for page_num in range(num_pages):
                print(f"\n--- Processing page {page_num + 1} of {num_pages} ---")
                
                # Create unique output prefix for this PDF and page
                output_prefix = f"{base_name}_page_{page_num}"
                
                # Run full workflow: parse coordinates with filter + LLM
                # Pass the pre-initialized client and model_name
                data = parse_pdf_coordinates_with_filter(
                    pdf_path, 
                    page_num=page_num, 
                    output_prefix=output_prefix,
                    client=client,
                    model_name=model_name
                )
                
                if isinstance(data, dict):
                    # Add page number to each field for reference
                    for field in data.get('fields', []):
                        field['page'] = page_num
                    all_fields['fields'].extend(data.get('fields', []))
                else:
                    print(f"Failed to get valid JSON data for {pdf_file} page {page_num}")
            
            # Save all coordinates to JSON
            if all_fields['fields']:
                json_file = os.path.join(data_dir, f"{base_name}.json")
                with open(json_file, 'w') as f:
                    json.dump(all_fields, f, indent=2)
                print(f"Saved coordinates to {json_file}")
                
                # Create interactive PDF
                create_acroform_from_coordinates(pdf_path, json_file, output_dir)
            else:
                print(f"No fields detected for {pdf_file}")
            
            print("Waiting for 3 seconds...")
            time.sleep(3)
            print("Done waiting!")
                        
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    
    print("\n" + "=" * 50)
    print("Done processing all files!")
    print("=" * 50)
