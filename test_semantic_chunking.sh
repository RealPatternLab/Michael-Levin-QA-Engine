#!/bin/bash

echo "🧪 Testing semantic chunking for s42005-025-02101-5.pdf"
echo "=================================================="

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "📋 Checking current status..."
python -c "
import json
from pathlib import Path
from scripts.simple_pipeline import load_metadata

metadata = load_metadata()
paper_name = 's42005-025-02101-5.pdf'

if paper_name in metadata['papers']:
    steps = metadata['papers'][paper_name].get('steps', {})
    print(f'Current steps: {list(steps.keys())}')
    
    if 'semantic_chunking' in steps:
        print(f'Semantic chunking status: {steps[\"semantic_chunking\"]}')
    else:
        print('No semantic chunking step found')
else:
    print('Paper not found in metadata')
"

echo ""
echo "🔍 Checking if consensus file exists..."
if [ -f "outputs/extracted_texts/s42005-025-02101-5/consensus.txt" ]; then
    echo "✅ Consensus file exists"
    echo "📏 Consensus file size: $(wc -c < outputs/extracted_texts/s42005-025-02101-5/consensus.txt) bytes"
else
    echo "❌ Consensus file not found"
    exit 1
fi

echo ""
echo "🚀 Running semantic chunking test..."
echo "=================================================="

python -c "
import sys
import time
from pathlib import Path
sys.path.append(str(Path('.')))

from scripts.simple_pipeline import process_semantic_chunking, load_metadata

print('Starting semantic chunking test...')
start_time = time.time()

metadata = load_metadata()
pdf_path = Path('inputs/raw_papers/s42005-025-02101-5.pdf')

print(f'Processing: {pdf_path.name}')
print(f'Start time: {time.strftime(\"%H:%M:%S\")}')

try:
    result = process_semantic_chunking(pdf_path, metadata)
    end_time = time.time()
    duration = end_time - start_time
    
    print(f'\\nResults:')
    print(f'✅ Success: {result}')
    print(f'⏱️ Duration: {duration:.1f} seconds')
    print(f'End time: {time.strftime(\"%H:%M:%S\")}')
    
    if result:
        # Check if chunks file was created
        chunks_file = Path('outputs/extracted_texts/s42005-025-02101-5/semantic_chunks.json')
        if chunks_file.exists():
            import json
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
            print(f'📊 Created {len(chunks)} semantic chunks')
        else:
            print('❌ Chunks file not created')
    
    # Check metadata
    metadata = load_metadata()
    if 'semantic_chunking' in metadata['papers']['s42005-025-02101-5.pdf']['steps']:
        print('✅ Metadata updated successfully')
    else:
        print('❌ Metadata not updated')
        
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🔍 Final status check..."
python -c "
import json
from pathlib import Path
from scripts.simple_pipeline import load_metadata

metadata = load_metadata()
paper_name = 's42005-025-02101-5.pdf'

if paper_name in metadata['papers']:
    steps = metadata['papers'][paper_name].get('steps', {})
    print(f'Final steps: {list(steps.keys())}')
    
    if 'semantic_chunking' in steps:
        print(f'Final semantic chunking status: {steps[\"semantic_chunking\"]}')
    else:
        print('Still no semantic chunking step')
else:
    print('Paper not found in metadata')

# Check if chunks file exists
chunks_file = Path('outputs/extracted_texts/s42005-025-02101-5/semantic_chunks.json')
if chunks_file.exists():
    print(f'✅ Chunks file exists: {chunks_file}')
    print(f'📏 File size: {chunks_file.stat().st_size} bytes')
else:
    print('❌ Chunks file not found')
"

echo ""
echo "✅ Test completed!" 