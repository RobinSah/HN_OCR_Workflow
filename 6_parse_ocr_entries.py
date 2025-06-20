import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import re
from tqdm import tqdm
import os
from datetime import datetime
import asyncio
from openai import AsyncOpenAI

# ------------ Async Batch GPT Integration ------------
PROMPT_TEMPLATE_SIMPLE = """
Extract the following fields from the directory entry. If a field is missing, use null.

**Preferred Output Format:**
{
  "FirstName": "Peter D",
  "LastName": "Aadland",
  "Spouse": "Pearl R",
  "Occupation": "Salesman",
  "CompanyName": "Lifetime Sls",
  "HomeAddress": {
    "StreetNumber": "2103",
    "StreetName": "Bryant av S",
    "ApartmentOrUnit": "apt 1",
    "ResidenceIndicator": "h"
  },
  "WorkAddress": null,
  "Telephone": null,
  "DirectoryName": "Minneapolis 1900",
  "PageNumber": 32
}

**Entry:**
{entry}

**Instructions:**
- Parse the entry and fill the JSON fields as accurately as possible.
- If a field is not present, use null.
- Output ONLY the JSON.
"""

async def async_parse_batch(entries, batch_size=20, max_concurrent_batches=5, model="gpt-3.5-turbo"):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent_batches)
    results = []

    async def process_batch(batch):
        async with semaphore:
            prompts = [
                PROMPT_TEMPLATE_SIMPLE.format(entry=entry)
                for entry in batch
            ]
            tasks = [
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                for prompt in prompts
            ]
            responses = await asyncio.gather(*tasks)
            batch_results = []
            for response in responses:
                try:
                    json_str = response.choices[0].message.content.strip()
                    json_obj = json.loads(json_str)
                    batch_results.append(json_obj)
                except Exception as e:
                    print(f"Error parsing batch response: {e}\nResponse: {json_str}")
                    batch_results.append(None)
            return batch_results

    batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]
    all_tasks = [process_batch(batch) for batch in batches]
    all_results = await asyncio.gather(*all_tasks)
    for batch in all_results:
        results.extend(batch)
    return results

# ---------------- Directory Parser Class ----------------
class DirectoryParserOpenAI:
    def __init__(self, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=500
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=3072
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            separators=["\n", '"', ".", ",", " "]
        )
        self.vector_store = None
        self.directory_year = 1910

    def create_extraction_prompt(self) -> PromptTemplate:
        template = """You are an expert at parsing historical directory entries from the 1910 Minneapolis City Directory.
TASK: Extract structured information from the directory entry below.

CONTEXT FROM SIMILAR ENTRIES:
{context}

ENTRY TO PARSE:
{entry}

PARSING RULES:
1. If entry starts with " (quote), it continues from the previous entry with last name: {last_name}
2. Entries follow pattern: [Last] [First] [occupation/details] [residence_indicator] [address]
3. Spouse info appears in parentheses, usually with "wid" meaning widow/widower
4. Business addresses often come after occupation

COMMON ABBREVIATIONS:
- Residence: h=house(owns), r=resides, b=boards, rms=rooms
- Jobs: clk=clerk, carp=carpenter, mach=machinist, tmstr=teamster, mngr=manager
- Streets: av=avenue, pl=place, N/S/E/W=North/South/East/West
- Other: wid=widow/widower, tel opr=telephone operator, trav agt=traveling agent

OUTPUT FORMAT (JSON only):
{{
  "FirstName": "Peter D",
  "LastName": "Aadland",
  "Spouse": "Pearl R",
  "Occupation": "Salesman",
  "CompanyName": "Lifetime Sls",
  "HomeAddress": {{
    "StreetNumber": "2103",
    "StreetName": "Bryant av S",
    "ApartmentOrUnit": "apt 1",
    "ResidenceIndicator": "h"
  }},
  "WorkAddress": null,
  "Telephone": null,
  "DirectoryName": "Minneapolis 1900",
  "PageNumber": 32
}}

IMPORTANT: Return ONLY the JSON object, no explanation or markdown."""
        return PromptTemplate(
            template=template,
            input_variables=["context", "entry", "last_name"]
        )

    def load_and_index_directory(self, json_files: List[str]):
        print("Loading directory files...")
        all_documents = []
        for json_file in tqdm(json_files, desc="Loading files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = data.get('text', '')
                    if not text:
                        continue
                    chunks = self.text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": Path(json_file).name,
                                "chunk_id": i,
                                "batch_name": data.get("batch_name", "unknown")
                            }
                        )
                        all_documents.append(doc)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        print("Creating vector index...")
        batch_size = 100
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(all_documents) + batch_size - 1)//batch_size}...")
            if i == 0:
                self.vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                self.vector_store.add_documents(batch)
        print(f"Created vector store with {len(all_documents)} chunks")
        self.vector_store.save_local("faiss_index")
        print("Vector store saved to 'faiss_index' directory")

    def extract_entries_from_text(self, text: str) -> List[str]:
        lines = text.split('\n')
        entries = []
        current_entry = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(skip in line.upper() for skip in ['DIRECTORY', 'MINNEAPOLIS', 'PAGE', 'ADVERTISEMENT']):
                continue
            if re.match(r'^[A-Z][a-z]+\s+[A-Z]|^"[A-Z]', line):
                if current_entry and len(current_entry) > 10:
                    entries.append(current_entry.strip())
                current_entry = line
            else:
                if current_entry:
                    current_entry += " " + line
        if current_entry and len(current_entry) > 10:
            entries.append(current_entry.strip())
        return entries

    def parse_single_entry(self, entry: str, last_name: str = None) -> Dict:
        """Parse a single directory entry using GPT-4o-mini and map output."""
        try:
            context_docs = self.vector_store.similarity_search(entry, k=2)
            context = "\n".join([doc.page_content[:200] for doc in context_docs])
            prompt = self.create_extraction_prompt()
            formatted_prompt = prompt.format(
                context=context,
                entry=entry,
                last_name=last_name or "Unknown"
            )
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return self.remap_fields(result)
            else:
                print(f"No JSON found in response for entry: {entry[:50]}...")
                return self.create_empty_entry()
        except Exception as e:
            print(f"Error parsing entry '{entry[:50]}...': {e}")
            return self.create_empty_entry()

    def remap_fields(self, result: Dict) -> Dict:
        """Map model fields to the standard output format."""
        # Helper to parse or pass HomeAddress
        def parse_home_address(addr):
            if isinstance(addr, dict):
                return {
                    "StreetNumber": addr.get("StreetNumber"),
                    "StreetName": addr.get("StreetName"),
                    "ApartmentOrUnit": addr.get("ApartmentOrUnit"),
                    "ResidenceIndicator": addr.get("ResidenceIndicator"),
                }
            elif isinstance(addr, str) and addr:
                # Very naive parse: split by space
                m = re.match(r'(\d+)\s+(.*)', addr)
                return {
                    "StreetNumber": m.group(1) if m else None,
                    "StreetName": m.group(2) if m else addr,
                    "ApartmentOrUnit": None,
                    "ResidenceIndicator": None
                }
            else:
                return {
                    "StreetNumber": None,
                    "StreetName": None,
                    "ApartmentOrUnit": None,
                    "ResidenceIndicator": None
                }

        return {
            "FirstName": result.get("FirstName") or result.get("first_name"),
            "LastName": result.get("LastName") or result.get("last_name"),
            "Spouse": result.get("Spouse") or result.get("spouse_name"),
            "Occupation": result.get("Occupation") or result.get("occupation"),
            "CompanyName": result.get("CompanyName") or result.get("employer_name"),
            "HomeAddress": parse_home_address(result.get("HomeAddress") or result.get("home_address")),
            "WorkAddress": result.get("WorkAddress"),
            "Telephone": result.get("Telephone"),
            "DirectoryName": result.get("DirectoryName"),
            "PageNumber": result.get("PageNumber")
        }

    def create_empty_entry(self) -> Dict:
        """Create an empty entry structure"""
        return {
            "FirstName": None,
            "LastName": None,
            "Spouse": None,
            "Occupation": None,
            "CompanyName": None,
            "HomeAddress": {
                "StreetNumber": None,
                "StreetName": None,
                "ApartmentOrUnit": None,
                "ResidenceIndicator": None
            },
            "WorkAddress": None,
            "Telephone": None,
            "DirectoryName": None,
            "PageNumber": None
        }

    def process_all_files(self, input_folder: str, output_file: str, sample_size: Optional[int] = None):
        json_files = glob.glob(f"{input_folder}/*.json")
        if not json_files:
            print(f"No JSON files found in {input_folder}")
            return
        print(f"Found {len(json_files)} files to process")
        if Path("faiss_index").exists():
            print("Loading existing vector index...")
            self.vector_store = FAISS.load_local(
                "faiss_index", 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.load_and_index_directory(json_files)
        all_entries = []
        total_processed = 0
        print("\nProcessing entries...")
        for json_file in json_files:
            if sample_size and total_processed >= sample_size:
                break
            print(f"\nProcessing {Path(json_file).name}...")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = data.get('text', '')
                entries = self.extract_entries_from_text(text)
                print(f"Found {len(entries)} entries in this file")
                last_name = None
                file_entries = []
                for entry in tqdm(entries, desc="Parsing entries"):
                    if sample_size and total_processed >= sample_size:
                        break
                    if not entry.startswith('"'):
                        name_match = re.match(r'^([A-Z][a-z]+)', entry)
                        if name_match:
                            last_name = name_match.group(1)
                    parsed = self.parse_single_entry(entry, last_name)
                    parsed['source_file'] = Path(json_file).name
                    parsed['raw_entry'] = entry
                    file_entries.append(parsed)
                    total_processed += 1
                all_entries.extend(file_entries)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        print(f"\nSaving {len(all_entries)} parsed entries...")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "metadata": {
                "total_entries": len(all_entries),
                "processed_date": datetime.now().isoformat(),
                "directory_year": self.directory_year,
                "model_used": "gpt-3.5-turbo",
                "embedding_model": "text-embedding-3-large"
            },
            "entries": all_entries
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        df = pd.DataFrame(all_entries)
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8')
        summary_file = output_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Directory Parsing Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Total entries processed: {len(all_entries)}\n")
            f.write(f"Processing date: {datetime.now()}\n")
            f.write(f"Files processed: {len(json_files)}\n\n")
            df_clean = df.dropna(subset=['LastName'])
            f.write(f"Entries with last names: {len(df_clean)}\n")
            f.write(f"Unique last names: {df_clean['LastName'].nunique()}\n")
            f.write(f"Entries with occupations: {df['Occupation'].notna().sum()}\n")
            f.write(f"Entries with addresses: {df['HomeAddress'].notna().sum()}\n")
        print(f"\nProcessing complete!")
        print(f"JSON output: {output_file}")
        print(f"CSV output: {csv_file}")
        print(f"Summary: {summary_file}")
        return df

# ------------- Example usage (sync version) -------------
if __name__ == "__main__":
    # Initialize parser with your OpenAI API key
    api_key = "OPENAI_API_KEY"
    parser = DirectoryParserOpenAI(api_key=api_key)
    # Process a sample first to test
    print("Testing with sample entries...")
    df_sample = parser.process_all_files(
        input_folder="simplified_outputs",
        output_file="structured_output/minneapolis_1910_sample.json",
        sample_size=50  # Process only 50 entries for testing
    )
    if df_sample is not None and len(df_sample) > 0:
        print("\nSample results:")
        print(df_sample.head(10))
        # If sample looks good, process all
        user_input = input("\nSample looks good? Process all entries? (y/n): ")
        if user_input.lower() == 'y':
            df_full = parser.process_all_files(
                input_folder="simplified_outputs",
                output_file="structured_output/minneapolis_1910_full.json"
            )
    else:
        print("No results from sample processing. Check your API key and data.")
