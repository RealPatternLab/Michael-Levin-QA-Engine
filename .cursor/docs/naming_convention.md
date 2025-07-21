# PDF Naming Convention for Levin RAG System

## 🎯 Naming Goals

1. **Informative**: Contains key identifying information
2. **Filesystem-safe**: No special characters, spaces, or invalid characters
3. **Sortable**: Chronological and logical ordering
4. **Consistent**: Same format for all papers
5. **Searchable**: Easy to find specific papers
6. **Unique**: Each paper has a distinct filename

## 📝 Naming Format

```
levin_[topic]_[year]_[journal]_[identifier].pdf
```

### Format Breakdown
- `levin_` - Author prefix for easy identification
- `[topic]` - Key concept or topic area (2-3 words max)
- `[year]` - Publication year
- `[journal]` - Journal abbreviation (2-4 letters)
- `[identifier]` - Short unique identifier (1-2 words)

## 🏷️ Topic Categories

### Core Research Areas
- `bioelectric` - Bioelectric signaling and fields
- `morphogenetic` - Morphogenetic intelligence and collective behavior
- `regeneration` - Regenerative biology and development
- `computation` - Computational approaches and modeling
- `development` - Developmental biology and embryology
- `cancer` - Cancer applications and research

### Specific Concepts
- `voltage_gradients` - Voltage gradient control
- `collective_intelligence` - Collective problem solving
- `multicellular` - Multicellular organization
- `self_boundary` - Computational boundaries of self
- `ml_hypothesis` - Machine learning applications

## 📚 Journal Abbreviations

### Common Journals
- `annrev` - Annual Review of Biomedical Engineering
- `frontiers` - Frontiers in Psychology
- `bioelectricity` - Bioelectricity journal
- `sci_rep` - Scientific Reports
- `nature` - Nature journals
- `cell` - Cell journals
- `plos` - PLOS journals
- `pnas` - Proceedings of the National Academy of Sciences
- `jcb` - Journal of Cell Biology
- `dev_bio` - Developmental Biology

### Special Cases
- `arxiv` - arXiv preprints
- `preprint` - Other preprints
- `book` - Book chapters
- `review` - Review papers

## 📚 Example Naming

### Original Filenames (from websites)
- `s41598-024-79087-7.pdf` → `levin_multicellular_2024_sci_rep_adaptation.pdf`
- `bioe.2024.0006.pdf` → `levin_bioelectric_2024_bioelectricity_applications.pdf`
- `fpsyg-12-765605.pdf` → `levin_self_boundary_2021_frontiers_computation.pdf`

### Our Target Papers
1. **"Endogenous Bioelectric Signaling Networks"** (2017, Annual Review)
   - `levin_bioelectric_2017_annrev_signaling.pdf`

2. **"The Computational Boundary of a 'Self'"** (2021, Frontiers)
   - `levin_self_boundary_2021_frontiers_computation.pdf`

3. **"Morphogenetic Intelligence"** (2023, Bioelectricity)
   - `levin_morphogenetic_2023_bioelectricity_intelligence.pdf`

4. **"Bioelectricity in Development, Regeneration, and Cancers"** (2024, Bioelectricity)
   - `levin_bioelectric_2024_bioelectricity_applications.pdf`

5. **"Machine Learning for Hypothesis Generation"** (2024, Digital Discovery)
   - `levin_ml_hypothesis_2024_digital_discovery.pdf`

## 🔧 Renaming Process

### Step 1: Identify Paper
- Extract title, year, and journal from PDF metadata or filename
- Determine primary topic category
- Choose appropriate journal abbreviation
- Create unique identifier

### Step 2: Create New Name
- Apply naming format: `levin_[topic]_[year]_[journal]_[identifier].pdf`
- Ensure no special characters or spaces
- Use underscores for word separation
- Verify uniqueness

### Step 3: Validate
- Check filename is unique within the collection
- Verify filesystem compatibility
- Test sorting and searching

## 📋 Renaming Checklist

For each downloaded PDF:
- [ ] Extract paper title, year, and journal
- [ ] Identify primary research topic
- [ ] Choose appropriate topic category
- [ ] Determine journal abbreviation
- [ ] Create unique identifier
- [ ] Apply naming format
- [ ] Verify uniqueness
- [ ] Rename file
- [ ] Update documentation

## 🎯 Benefits

1. **Unique identification**: Journal + identifier prevents conflicts
2. **Easy identification**: `levin_bioelectric_` prefix makes all papers findable
3. **Chronological sorting**: Year in filename enables time-based organization
4. **Topic grouping**: Topic categories enable concept-based filtering
5. **Journal tracking**: Can filter by publication venue
6. **Filesystem safe**: No spaces, special characters, or invalid symbols
7. **Consistent format**: Same structure for all papers

## 📁 Directory Structure

```
data/raw_papers/
├── levin_bioelectric_2017_annrev_signaling.pdf
├── levin_self_boundary_2021_frontiers_computation.pdf
├── levin_morphogenetic_2023_bioelectricity_intelligence.pdf
├── levin_bioelectric_2024_bioelectricity_applications.pdf
└── levin_ml_hypothesis_2024_digital_discovery.pdf
```

## 🔍 Search Examples

```bash
# Find all bioelectric papers
ls data/raw_papers/levin_bioelectric_*.pdf

# Find papers from specific year
ls data/raw_papers/levin_*_2024_*.pdf

# Find papers by journal
ls data/raw_papers/levin_*_*_bioelectricity_*.pdf

# Find papers by topic and year
ls data/raw_papers/levin_bioelectric_2024_*.pdf
```

## 🚨 Conflict Resolution

If naming conflicts still occur:
1. **Add month**: `levin_topic_2024_journal_month_identifier.pdf`
2. **Add co-authors**: `levin_topic_2024_journal_coauthor_identifier.pdf`
3. **Add DOI suffix**: `levin_topic_2024_journal_identifier_doi.pdf`
4. **Add sequence number**: `levin_topic_2024_journal_identifier_01.pdf` 