import json
import pandas as pd
from pathlib import Path
import glob
import re
from tqdm import tqdm
from datetime import datetime
from groq import Groq
import time

class EfficientDirectoryParser:
    """
    Hybrid approach: 95% regex, 5% LLM for edge cases
    Processes 40MB in ~5-10 minutes
    """
    
    def __init__(self, groq_api_key: str = None):
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.stats = {"regex_parsed": 0, "llm_parsed": 0, "failed": 0}
        
    def extract_entries(self, text: str) -> list:
        """Extract individual entries from directory text"""
        # Split by newlines and identify entries
        entries = []
        current = ""
        
        for line in text.split('\n'):
            line = line.strip()
            if not line or len(line) > 200:  # Skip empty or too long
                continue
                
            # Skip obvious headers/ads
            if any(word in line.upper() for word in ['DIRECTORY', 'PAGE', 'ADVERTISEMENT', 'TELEPHONE']):
                continue
            
            # New entry pattern: Lastname Firstname or "Firstname
            if re.match(r'^[A-Z][a-z]+\s+[A-Z]|^"[A-Z]', line):
                if 10 < len(current) < 300:  # Valid entry length
                    entries.append(current.strip())
                current = line
            elif current:  # Continuation
                current += " " + line
                
        if current and 10 < len(current) < 300:
            entries.append(current.strip())
            
        return entries
    
    def parse_with_regex(self, entry: str, last_name: str = None) -> dict:
        """
        Regex parser that handles 90%+ of cases
        Returns None if confidence is low
        """
        result = {
            "FirstName": None,
            "LastName": last_name,
            "Spouse": None,
            "Occupation": None,
            "CompanyName": None,
            "HomeAddress": {
                "StreetNumber": None,
                "StreetName": None,
                "ResidenceIndicator": None
            }
        }
        
        # Clean entry
        entry = re.sub(r'\s+', ' ', entry.strip())
        
        # Pattern 1: Continuation entry ("FirstName ...)
        if entry.startswith('"'):
            match = re.match(r'^"([A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s*(.*)', entry)
            if match:
                result["FirstName"] = match.group(1)
                rest = match.group(2)
            else:
                return None  # Can't parse, need LLM
        else:
            # Pattern 2: Full entry (LastName FirstName ...)
            match = re.match(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s*(.*)', entry)
            if match:
                result["LastName"] = match.group(1)
                result["FirstName"] = match.group(2)
                rest = match.group(3)
            else:
                return None  # Can't parse, need LLM
        
        # Extract spouse (in parentheses)
        spouse_match = re.search(r'\(([^)]+)\)', rest)
        if spouse_match:
            spouse_info = spouse_match.group(1)
            # Remove "wid" (widow/widower of)
            spouse_info = re.sub(r'\bwid\s+', '', spouse_info)
            result["Spouse"] = spouse_info.strip()
            # Remove spouse from rest
            rest = rest[:spouse_match.start()] + rest[spouse_match.end():]
        
        # Find residence pattern (h/r/b/rms followed by address)
        res_match = re.search(r'\b(h|r|b|rms)\s+(\d+)\s+([^,\.]+)', rest)
        if res_match:
            result["HomeAddress"]["ResidenceIndicator"] = res_match.group(1)
            result["HomeAddress"]["StreetNumber"] = res_match.group(2)
            result["HomeAddress"]["StreetName"] = res_match.group(3).strip()
            
            # Everything before residence is likely occupation/company
            occ_text = rest[:res_match.start()].strip(' ,.')
            if occ_text:
                # Common occupation patterns
                occ_parts = occ_text.split(' ', 1)
                if occ_parts:
                    result["Occupation"] = occ_parts[0]
                    if len(occ_parts) > 1:
                        result["CompanyName"] = occ_parts[1].strip()
        
        # Confidence check - must have name and either address or occupation
        has_name = result["FirstName"] or result["LastName"]
        has_data = result["HomeAddress"]["StreetName"] or result["Occupation"]
        
        if has_name and has_data:
            self.stats["regex_parsed"] += 1
            return result
        else:
            return None  # Low confidence, use LLM
    
    def parse_with_llm(self, entry: str, last_name: str = None) -> dict:
        """Use LLM only for entries regex couldn't handle"""
        if not self.groq_client:
            self.stats["failed"] += 1
            return self.create_empty_entry()
            
        prompt = f"""Parse this directory entry into JSON. Be precise.
Entry: {entry}
Last name if continuation: {last_name}

Output only JSON:
{{"FirstName": "", "LastName": "", "Spouse": "", "Occupation": "", "CompanyName": "", "HomeAddress": {{"StreetNumber": "", "StreetName": "", "ResidenceIndicator": ""}}}}"""

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=<model_of_your_choice>,  # Fastest current model
                temperature=0,
                max_tokens=200
            )
            
            text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                self.stats["llm_parsed"] += 1
                return json.loads(json_match.group())
        except Exception as e:
            print(f"LLM error: {e}")
            
        self.stats["failed"] += 1
        return self.create_empty_entry()
    
    def create_empty_entry(self) -> dict:
        return {
            "FirstName": None,
            "LastName": None,
            "Spouse": None,
            "Occupation": None,
            "CompanyName": None,
            "HomeAddress": {
                "StreetNumber": None,
                "StreetName": None,
                "ResidenceIndicator": None
            }
        }
    
    def process_all_files(self, input_folder: str, output_file: str):
        """Process all JSON files efficiently"""
        json_files = glob.glob(f"{input_folder}/*.json")
        print(f"Found {len(json_files)} files to process")
        
        all_results = []
        total_entries = 0
        
        for json_file in json_files:
            print(f"\nProcessing {Path(json_file).name}...")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get('text', '')
            
            # Extract entries
            entries = self.extract_entries(text)
            print(f"Found {len(entries)} entries")
            total_entries += len(entries)
            
            # Track last names for continuation entries
            current_last_name = None
            llm_queue = []  # Batch LLM requests
            
            for entry in tqdm(entries, desc="Parsing"):
                # Update last name tracking
                if not entry.startswith('"'):
                    name_match = re.match(r'^([A-Z][a-z]+)', entry)
                    if name_match:
                        current_last_name = name_match.group(1)
                
                # Try regex first
                result = self.parse_with_regex(entry, current_last_name)
                
                if result:
                    # Regex succeeded
                    result["source_file"] = Path(json_file).name
                    result["raw_entry"] = entry[:100] + "..." if len(entry) > 100 else entry
                    all_results.append(result)
                else:
                    # Queue for LLM
                    llm_queue.append((entry, current_last_name, Path(json_file).name))
            
            # Process LLM queue in batches
            if llm_queue and self.groq_client:
                print(f"Processing {len(llm_queue)} complex entries with LLM...")
                for entry, last_name, source in tqdm(llm_queue, desc="LLM parsing"):
                    result = self.parse_with_llm(entry, last_name)
                    result["source_file"] = source
                    result["raw_entry"] = entry[:100] + "..." if len(entry) > 100 else entry
                    all_results.append(result)
                    time.sleep(0.1)  # Rate limiting
        
        # Save results
        print(f"\nTotal entries processed: {total_entries}")
        print(f"Regex parsed: {self.stats['regex_parsed']} ({self.stats['regex_parsed']/total_entries*100:.1f}%)")
        print(f"LLM parsed: {self.stats['llm_parsed']} ({self.stats['llm_parsed']/total_entries*100:.1f}%)")
        print(f"Failed: {self.stats['failed']} ({self.stats['failed']/total_entries*100:.1f}%)")
        
        # Save JSON
        Path(output_file).parent.mkdir(exist_ok=True)
        output_data = {
            "metadata": {
                "total_entries": len(all_results),
                "processed_date": datetime.now().isoformat(),
                "stats": self.stats
            },
            "entries": all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save CSV
        df = pd.DataFrame(all_results)
        # Flatten nested address
        if 'HomeAddress' in df.columns:
            df['StreetNumber'] = df['HomeAddress'].apply(lambda x: x.get('StreetNumber') if isinstance(x, dict) else None)
            df['StreetName'] = df['HomeAddress'].apply(lambda x: x.get('StreetName') if isinstance(x, dict) else None)
            df['ResidenceIndicator'] = df['HomeAddress'].apply(lambda x: x.get('ResidenceIndicator') if isinstance(x, dict) else None)
            df = df.drop('HomeAddress', axis=1)
        
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"\nSaved to:")
        print(f"- {output_file}")
        print(f"- {csv_file}")
        
        return df


# Usage
if __name__ == "__main__":
    # Option 1: Pure regex (2-5 minutes for 40MB)

    #Use your api key
 
    df = parser.process_all_files(
        input_folder="simplified_outputs",
        output_file="structured_output/efficient_parse.json"
    )
    
    # Quick quality check
    print("\nQuality Check:")
    print(f"Entries with names: {df['LastName'].notna().sum()}")
    print(f"Entries with addresses: {df['StreetName'].notna().sum()}")
    print(f"Entries with occupations: {df['Occupation'].notna().sum()}")
    
    # Show samples
    print("\nSample entries:")
    print(df[['LastName', 'FirstName', 'Occupation', 'StreetName']].head(20))
