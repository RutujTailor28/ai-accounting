from typing import List, BinaryIO
import PyPDF2
import openpyxl
import xlrd
import pandas as pd
from docx import Document


class DocumentParser:
    """Unified document parser for multiple file formats."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.csv', '.docx'}
    
    @staticmethod
    def parse(file_content: BinaryIO, filename: str) -> str:
        """
        Parse document and extract text based on file extension.
        
        Args:
            file_content: Binary file content
            filename: Name of the file including extension
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file type is not supported
        """
        extension = filename.lower().split('.')[-1]
        extension_with_dot = f'.{extension}'
        
        if extension_with_dot not in DocumentParser.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}. Supported types: {DocumentParser.SUPPORTED_EXTENSIONS}")
        
        print(f"[INFO] Parsing document: {filename} (type: {extension})")
        
        try:
            if extension == 'pdf':
                return DocumentParser._parse_pdf(file_content)
            elif extension in ['xlsx', 'xls']:
                return DocumentParser._parse_excel(file_content, extension)
            elif extension == 'csv':
                return DocumentParser._parse_csv(file_content)
            elif extension == 'docx':
                return DocumentParser._parse_docx(file_content)
        except Exception as e:
            print(f"[ERROR] Error parsing {filename}: {str(e)}")
            raise
    
    @staticmethod
    def _parse_pdf(file_content: BinaryIO) -> str:
        """Extract text from PDF file."""
        pdf_reader = PyPDF2.PdfReader(file_content)
        text_parts = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                text_parts.append(text)
        
        full_text = '\n\n'.join(text_parts)
        print(f"[INFO] Extracted {len(full_text)} characters from PDF ({len(pdf_reader.pages)} pages)")
        return full_text
    
    @staticmethod
    def _parse_excel(file_content: BinaryIO, extension: str) -> str:
        """Extract text from Excel file (.xlsx or .xls)."""
        text_parts = []
        
        if extension == 'xlsx':
            workbook = openpyxl.load_workbook(file_content, data_only=True)
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text_parts.append(f"Sheet: {sheet_name}")
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = '\t'.join([str(cell) if cell is not None else '' for cell in row])
                    if row_text.strip():
                        text_parts.append(row_text)
        else:  # .xls
            workbook = xlrd.open_workbook(file_contents=file_content.read())
            for sheet in workbook.sheets():
                text_parts.append(f"Sheet: {sheet.name}")
                
                for row_idx in range(sheet.nrows):
                    row = sheet.row_values(row_idx)
                    row_text = '\t'.join([str(cell) for cell in row])
                    if row_text.strip():
                        text_parts.append(row_text)
        
        full_text = '\n'.join(text_parts)
        print(f"[INFO] Extracted {len(full_text)} characters from Excel")
        return full_text
    
    @staticmethod
    def _parse_csv(file_content: BinaryIO) -> str:
        """Extract text from CSV file."""
        df = pd.read_csv(file_content)
        
        # Convert DataFrame to text representation
        text_parts = [f"Columns: {', '.join(df.columns.tolist())}"]
        
        for _, row in df.iterrows():
            row_text = '\t'.join([str(val) for val in row.values])
            text_parts.append(row_text)
        
        full_text = '\n'.join(text_parts)
        print(f"[INFO] Extracted {len(full_text)} characters from CSV ({len(df)} rows)")
        return full_text
    
    @staticmethod
    def _parse_docx(file_content: BinaryIO) -> str:
        """Extract text from Word document."""
        doc = Document(file_content)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = '\t'.join([cell.text for cell in row.cells])
                if row_text.strip():
                    text_parts.append(row_text)
        
        full_text = '\n\n'.join(text_parts)
        print(f"[INFO] Extracted {len(full_text)} characters from Word document")
        return full_text
