import pandas as pd 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from pymavlink import mavutil
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import Dict, List, Optional, Any
import re
from collections import defaultdict
import requests

executor = ThreadPoolExecutor(max_workers=4)

def read_data(file_path, msg_types):    
    try:
        mlog = mavutil.mavlink_connection(file_path)
        data_collectors = defaultdict(list)
        
        while True:
            msg = mlog.recv_match(type = msg_types)
            if msg is None:
                break
                
            msg_type = msg.get_type()
            data_collectors[msg_type].append(msg.to_dict())
            
        return json.dumps(data_collectors["GPS"])
        
    except Exception as e:
        print(f"Error reading MAVLink file: {str(e)}")
        return ""
    
def convert_role(langchain_role):
    role_mapping = {"human": "user",
                    "ai": "assistant", 
                    "system": "system",
                    "user": "user",  
                    "assistant": "assistant" }
    
    return role_mapping.get(langchain_role, "user")  

class DynamicTableParser:
    def __init__(self, url: str):
        self.url = url
        self.soup = None
        self.extracted_data = {}
    
    def fetch_page(self) -> bool:
        """Fetch the webpage content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(self.url, headers=headers, timeout=30)
            response.raise_for_status()
            self.soup = BeautifulSoup(response.content, 'html.parser')
            return True
        except requests.RequestException as e:
            print(f"Error fetching page: {e}")
            return False
    
    def find_all_tables(self) -> List[Dict[str, Any]]:
        """Find and catalog all tables on the page with their context"""
        if not self.soup:
            return []
        
        tables_info = []
        tables = self.soup.find_all('table')
        
        for i, table in enumerate(tables):
            table_info = {
                'index': i,
                'table_element': table,
                'context': self._get_table_context(table),
                'structure': self._analyze_table_structure(table),
                'data': None
            }
            tables_info.append(table_info)
        
        return tables_info
    
    def _get_table_context(self, table) -> Dict[str, Any]:
        """Extract context around the table (headers, captions, nearby text)"""
        context = {
            'preceding_headers': [],
            'following_descriptions': [],
            'caption': None,
            'nearby_text': [],
            'section_title': None,
            'table_description': None,
            'table_id': None,
            'table_class': None
        }
        
        # Get table attributes
        context['table_id'] = table.get('id')
        context['table_class'] = table.get('class')
        
        # Get caption
        caption = table.find('caption')
        if caption:
            context['caption'] = caption.get_text(strip=True)
        
        # Look for preceding headers (h1-h6) and context
        current = table.previous_sibling
        header_distance = 0
        while current and header_distance < 5:
            if hasattr(current, 'name'):
                if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    header_text = current.get_text(strip=True)
                    if header_text:
                        context['preceding_headers'].append({
                            'level': current.name,
                            'text': header_text,
                            'distance': header_distance
                        })
                        if header_distance <= 2 and not context['section_title']:
                            context['section_title'] = header_text
                elif current.name in ['p', 'div', 'span']:
                    text = current.get_text(strip=True)
                    if text and len(text) < 200:
                        context['nearby_text'].append(text)
            current = current.previous_sibling
            header_distance += 1
        
        # Look for following descriptions/explanations
        current = table.next_sibling
        desc_distance = 0
        while current and desc_distance < 5:
            if hasattr(current, 'name'):
                if current.name in ['p', 'div', 'span', 'small', 'em', 'i']:
                    text = current.get_text(strip=True)
                    if text and len(text) > 10:  # Meaningful description
                        context['following_descriptions'].append({
                            'text': text,
                            'distance': desc_distance,
                            'tag': current.name
                        })
                        # Use the first substantial description as the main description
                        if not context['table_description'] and len(text) > 20:
                            context['table_description'] = text
                elif current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table']:
                    # Stop if we hit another major element
                    break
            current = current.next_sibling
            desc_distance += 1
        
        # Also check for descriptions in parent/sibling containers
        parent = table.parent
        if parent:
            # Look for description in same container after the table
            for sibling in parent.find_all(['p', 'div', 'span'], recursive=False):
                if sibling.get_text(strip=True) and len(sibling.get_text(strip=True)) > 20:
                    # Check if this comes after the table
                    try:
                        if table in sibling.previous_siblings or table.parent in sibling.previous_siblings:
                            continue
                        text = sibling.get_text(strip=True)
                        if text not in [desc['text'] for desc in context['following_descriptions']]:
                            context['following_descriptions'].append({
                                'text': text,
                                'distance': 999,  # Mark as container-level
                                'tag': sibling.name
                            })
                            if not context['table_description']:
                                context['table_description'] = text
                    except:
                        pass
        
        return context
    
    def _analyze_table_structure(self, table) -> Dict[str, Any]:
        """Analyze the structure of the table"""
        rows = table.find_all('tr')
        if not rows:
            return {'valid': False, 'reason': 'No rows found'}
        
        structure = {
            'valid': True,
            'total_rows': len(rows),
            'has_header': False,
            'header_row_index': None,
            'columns': [],
            'column_count': 0,
            'data_types': {},
            'sample_data': []
        }
        
        # Analyze potential header row
        for i, row in enumerate(rows[:3]):  # Check first 3 rows for headers
            cells = row.find_all(['th', 'td'])
            if not cells:
                continue
                
            # Check if this row looks like a header
            is_header = self._is_likely_header_row(row, cells)
            if is_header and not structure['has_header']:
                structure['has_header'] = True
                structure['header_row_index'] = i
                structure['columns'] = [cell.get_text(strip=True) for cell in cells]
                structure['column_count'] = len(cells)
                break
        
        # If no clear header found, use first row
        if not structure['has_header'] and rows:
            first_row_cells = rows[0].find_all(['th', 'td'])
            if first_row_cells:
                structure['columns'] = [f"column_{j+1}" for j in range(len(first_row_cells))]
                structure['column_count'] = len(first_row_cells)
        
        # Analyze data types and get samples
        data_start_index = (structure['header_row_index'] + 1) if structure['has_header'] else 0
        structure['sample_data'] = self._extract_sample_data(rows[data_start_index:], structure['column_count'])
        structure['data_types'] = self._infer_column_types(structure['sample_data'], structure['columns'])
        
        return structure
    
    def _is_likely_header_row(self, row, cells) -> bool:
        """Determine if a row is likely a header row"""
        # Check for th tags
        if row.find_all('th'):
            return True
        
        # Check if text looks like headers (short, descriptive)
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        # Headers usually don't contain only numbers
        numeric_cells = sum(1 for text in cell_texts if self._is_numeric(text))
        if numeric_cells == len(cell_texts) and len(cell_texts) > 1:
            return False
        
        # Headers are usually shorter
        avg_length = sum(len(text) for text in cell_texts) / len(cell_texts) if cell_texts else 0
        if avg_length > 50:
            return False
        
        # Check for common header patterns
        header_patterns = ['time', 'id', 'value', 'data', 'name', 'type', 'status', 'lat', 'lng', 'alt']
        matching_patterns = sum(1 for text in cell_texts 
                              for pattern in header_patterns 
                              if pattern in text.lower())
        
        return matching_patterns > 0 or avg_length < 20
    
    def _extract_sample_data(self, data_rows: List, column_count: int, sample_size: int = 5) -> List[List[str]]:
        """Extract sample data rows for analysis"""
        samples = []
        for row in data_rows[:sample_size]:
            cells = row.find_all(['td', 'th'])
            if len(cells) == column_count:
                row_data = [cell.get_text(strip=True) for cell in cells]
                samples.append(row_data)
        return samples
    
    def _infer_column_types(self, sample_data: List[List[str]], columns: List[str]) -> Dict[str, str]:
        """Infer data types for each column"""
        if not sample_data or not columns:
            return {}
        
        column_types = {}
        
        for col_idx, col_name in enumerate(columns):
            values = [row[col_idx] for row in sample_data if col_idx < len(row)]
            column_types[col_name] = self._infer_single_column_type(values)
        
        return column_types
    
    def _infer_single_column_type(self, values: List[str]) -> str:
        """Infer the data type of a single column"""
        if not values:
            return 'unknown'
        
        non_empty_values = [v for v in values if v.strip()]
        if not non_empty_values:
            return 'empty'
        
        # Check for numeric types
        numeric_count = sum(1 for v in non_empty_values if self._is_numeric(v))
        float_count = sum(1 for v in non_empty_values if self._is_float(v))
        
        if numeric_count == len(non_empty_values):
            return 'float' if float_count > 0 else 'integer'
        
        # Check for datetime
        datetime_count = sum(1 for v in non_empty_values if self._is_datetime_like(v))
        if datetime_count > len(non_empty_values) * 0.8:
            return 'datetime'
        
        # Check for boolean
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'on', 'off'}
        boolean_count = sum(1 for v in non_empty_values if v.lower() in boolean_values)
        if boolean_count == len(non_empty_values):
            return 'boolean'
        
        return 'text'
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a string represents a number"""
        try:
            float(value.replace(',', ''))
            return True
        except (ValueError, AttributeError):
            return False
    
    def _is_float(self, value: str) -> bool:
        """Check if a string represents a float"""
        return '.' in value and self._is_numeric(value)
    
    def _is_datetime_like(self, value: str) -> bool:
        """Check if a string looks like a datetime"""
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}:\d{2}:\d{2}',  # HH:MM:SS
            r'\d{13,}',            # Timestamp (microseconds)
        ]
        return any(re.search(pattern, value) for pattern in datetime_patterns)
    
    def parse_table(self, table_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse a single table into structured data"""
        table = table_info['table_element']
        structure = table_info['structure']
        
        if not structure['valid']:
            return []
        
        rows = table.find_all('tr')
        data_start = (structure['header_row_index'] + 1) if structure['has_header'] else 0
        
        parsed_data = []
        for row in rows[data_start:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) == structure['column_count']:
                row_dict = {}
                for i, cell in enumerate(cells):
                    column_name = structure['columns'][i] if i < len(structure['columns']) else f"column_{i+1}"
                    cell_value = cell.get_text(strip=True)
                    
                    # Convert based on inferred type
                    if column_name in structure['data_types']:
                        cell_value = self._convert_value(cell_value, structure['data_types'][column_name])
                    
                    row_dict[column_name] = cell_value
                parsed_data.append(row_dict)
        
        return parsed_data
    
    def _convert_value(self, value: str, data_type: str) -> Any:
        """Convert string value to appropriate type"""
        if not value.strip():
            return None
        
        try:
            if data_type == 'integer':
                return int(float(value.replace(',', '')))
            elif data_type == 'float':
                return float(value.replace(',', ''))
            elif data_type == 'boolean':
                return value.lower() in {'true', '1', 'yes', 'on'}
            else:
                return value
        except (ValueError, AttributeError):
            return value
    
    def extract_all_data(self) -> Dict[str, Any]:
        """Extract all table data from the webpage"""
        if not self.soup:
            if not self.fetch_page():
                return {}
        
        tables_info = self.find_all_tables()
        extracted_data = {}
        
        for table_info in tables_info:
            # Generate a meaningful name for the table
            table_name = self._generate_table_name(table_info)
            
            # Parse the table data
            table_data = self.parse_table(table_info)
            
            if table_data:
                extracted_data[table_name] = {
                    'data': table_data,
                    'metadata': {
                        'context': table_info['context'],
                        'structure': table_info['structure'],
                        'row_count': len(table_data),
                        'column_count': table_info['structure']['column_count']
                    }
                }
        
        self.extracted_data = extracted_data
        return extracted_data
    
    def _generate_table_name(self, table_info: Dict[str, Any]) -> str:
        """Generate a meaningful name for the table"""
        context = table_info['context']
        
        # Priority order for naming
        if context['section_title']:
            name = context['section_title']
        elif context['caption']:
            name = context['caption']
        elif context['table_id']:
            name = context['table_id']
        elif context['preceding_headers']:
            name = context['preceding_headers'][0]['text']
        else:
            name = f"table_{table_info['index'] + 1}"
        
        # Clean the name
        name = re.sub(r'[^\w\s-]', '', name).strip()
        name = re.sub(r'\s+', '_', name)
        return name[:50]  # Limit length
    
    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Convert extracted data to pandas DataFrames"""
        if not self.extracted_data:
            self.extract_all_data()
        
        dataframes = {}
        for table_name, table_info in self.extracted_data.items():
            if table_info['data']:
                df = pd.DataFrame(table_info['data'])
                dataframes[table_name] = df
        
        return dataframes
    
    def save_to_json(self, filename: str):
        """Save all extracted data to JSON"""
        if not self.extracted_data:
            self.extract_all_data()
        
        # Prepare data for JSON serialization
        json_data = {}
        for table_name, table_info in self.extracted_data.items():
            json_data[table_name] = {
                'data': table_info['data'],
                'metadata': {
                    'row_count': table_info['metadata']['row_count'],
                    'column_count': table_info['metadata']['column_count'],
                    'section_title': table_info['metadata']['context']['section_title'],
                    'table_description': table_info['metadata']['context']['table_description'],
                    'columns': table_info['metadata']['structure']['columns'],
                    'data_types': table_info['metadata']['structure']['data_types'],
                    'all_descriptions': table_info['metadata']['context']['following_descriptions'],
                    'full_context': table_info['metadata']['context']
                }
            }
        
        with open(f"output/{filename}", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Data saved to {filename}")
    
    def get_table_with_description(self, table_name: str) -> Dict[str, Any]:
        """Get a specific table with its full description and context"""
        if table_name not in self.extracted_data:
            return {}
        
        table_info = self.extracted_data[table_name]
        return {
            'name': table_name,
            'description': table_info['metadata']['context']['table_description'],
            'section_title': table_info['metadata']['context']['section_title'],
            'all_descriptions': table_info['metadata']['context']['following_descriptions'],
            'data': table_info['data'],
            'columns': table_info['metadata']['structure']['columns'],
            'data_types': table_info['metadata']['structure']['data_types']
        }
    
    def get_all_descriptions(self) -> Dict[str, List[str]]:
        """Get all descriptions for all tables"""
        if not self.extracted_data:
            self.extract_all_data()
        
        descriptions = {}
        for table_name, table_info in self.extracted_data.items():
            table_descriptions = []
            
            # Add main description
            if table_info['metadata']['context']['table_description']:
                table_descriptions.append(table_info['metadata']['context']['table_description'])
            
            # Add all following descriptions
            for desc in table_info['metadata']['context']['following_descriptions']:
                if desc['text'] not in table_descriptions:
                    table_descriptions.append(desc['text'])
            
            descriptions[table_name] = table_descriptions
        
        return descriptions
    
    def save_to_csv(self, base_filename: str):
        """Save each table to separate CSV files"""
        dataframes = self.to_dataframes()
        
        for table_name, df in dataframes.items():
            # Clean filename
            clean_name = re.sub(r'[^\w\s-]', '', table_name)
            clean_name = re.sub(r'\s+', '_', clean_name)
            filename = f"{base_filename}_{clean_name}.csv"
            
            df.to_csv(f"output/{filename}", index=False)
            print(f"Saved {table_name} to {filename}")
    
    def print_summary(self):
        """Print a summary of extracted data"""
        if not self.extracted_data:
            self.extract_all_data()
        
        print(f"\nExtracted {len(self.extracted_data)} tables:")
        print("-" * 60)
        
        for table_name, table_info in self.extracted_data.items():
            metadata = table_info['metadata']
            print(f"\nTable: {table_name}")
            print(f"  Rows: {metadata['row_count']}")
            print(f"  Columns: {metadata['column_count']}")
            
            if metadata['context']['section_title']:
                print(f"  Section: {metadata['context']['section_title']}")
            
            if metadata['context']['table_description']:
                desc = metadata['context']['table_description']
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                print(f"  Description: {desc}")
            
            if metadata['structure']['columns']:
                cols = metadata['structure']['columns'][:5]  # Show first 5 columns
                col_str = ', '.join(cols)
                if len(metadata['structure']['columns']) > 5:
                    col_str += '...'
                print(f"  Columns: {col_str}")
            
            if metadata['structure']['data_types']:
                types = list(set(metadata['structure']['data_types'].values()))
                print(f"  Data types: {', '.join(types)}")
            
            # Show following descriptions if any
            if metadata['context']['following_descriptions']:
                print(f"  Additional descriptions: {len(metadata['context']['following_descriptions'])} found")

def find_url(text):
    url_pattern = re.compile(
        r'(?:(?:http|ftp)s?://|www\.)[^\s/$.?#].[^\s]*', re.IGNORECASE)
    match = url_pattern.search(text)
    return match.group(0) if match else None






