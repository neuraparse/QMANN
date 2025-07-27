#!/usr/bin/env python3
"""
arXiv metadata processor for QMANN paper.

This script processes meta.yaml and generates arXiv-compatible metadata.
"""

import argparse
import yaml
import json
import sys
from pathlib import Path
from datetime import datetime
import re

def validate_orcid(orcid):
    """Validate ORCID format."""
    pattern = r'^0000-\d{4}-\d{4}-\d{3}[\dX]$'
    return re.match(pattern, orcid) is not None

def validate_arxiv_category(category):
    """Validate arXiv category format."""
    valid_categories = {
        'cs.LG', 'cs.AI', 'cs.CL', 'cs.CV', 'cs.DC', 'cs.DS', 'cs.IR', 'cs.IT',
        'quant-ph', 'physics.comp-ph', 'physics.data-an',
        'math.OC', 'math.ST', 'stat.ML', 'stat.ME'
    }
    return category in valid_categories

def process_authors(authors):
    """Process and validate author information."""
    processed_authors = []
    
    for i, author in enumerate(authors):
        if 'name' not in author:
            raise ValueError(f"Author {i+1} missing required 'name' field")
        
        processed_author = {
            'name': author['name'],
            'affiliation': author.get('affiliation', ''),
            'email': author.get('email', '')
        }
        
        # Validate ORCID if provided
        if 'orcid' in author:
            orcid = author['orcid']
            if not validate_orcid(orcid):
                raise ValueError(f"Invalid ORCID format for {author['name']}: {orcid}")
            processed_author['orcid'] = orcid
        
        processed_authors.append(processed_author)
    
    return processed_authors

def process_categories(categories):
    """Process and validate arXiv categories."""
    if not categories:
        raise ValueError("At least one arXiv category is required")
    
    processed_categories = []
    for category in categories:
        if not validate_arxiv_category(category):
            print(f"Warning: '{category}' may not be a valid arXiv category")
        processed_categories.append(category)
    
    return processed_categories

def generate_arxiv_metadata(meta_data):
    """Generate arXiv-compatible metadata."""
    
    # Required fields
    required_fields = ['title', 'authors', 'abstract', 'categories']
    for field in required_fields:
        if field not in meta_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Process components
    authors = process_authors(meta_data['authors'])
    categories = process_categories(meta_data['categories'])
    
    # Build arXiv metadata
    arxiv_meta = {
        'title': meta_data['title'].strip(),
        'authors': authors,
        'abstract': meta_data['abstract'].strip(),
        'categories': categories,
        'primary_category': categories[0],
        'comments': meta_data.get('comments', ''),
        'license': meta_data.get('license', 'CC-BY-4.0'),
        'submission_date': datetime.now().isoformat(),
    }
    
    # Optional fields
    if 'keywords' in meta_data:
        arxiv_meta['keywords'] = meta_data['keywords']
    
    if 'journal' in meta_data:
        arxiv_meta['journal_ref'] = meta_data['journal']
    
    if 'doi' in meta_data:
        arxiv_meta['doi'] = meta_data['doi']
    
    return arxiv_meta

def generate_latex_metadata(meta_data):
    """Generate LaTeX metadata commands."""
    
    latex_commands = []
    
    # Title
    latex_commands.append(f"\\title{{{meta_data['title']}}}")
    
    # Authors
    for author in meta_data['authors']:
        latex_commands.append(f"\\author{{{author['name']}}}")
        if author.get('email'):
            latex_commands.append(f"\\email{{{author['email']}}}")
        if author.get('affiliation'):
            latex_commands.append(f"\\affiliation{{{author['affiliation']}}}")
    
    # Date
    latex_commands.append(f"\\date{{\\today}}")
    
    # Abstract (as comment)
    abstract_lines = meta_data['abstract'].split('\n')
    latex_commands.append("% Abstract:")
    for line in abstract_lines:
        latex_commands.append(f"% {line.strip()}")
    
    # Keywords
    if 'keywords' in meta_data:
        keywords_str = ', '.join(meta_data['keywords'])
        latex_commands.append(f"\\keywords{{{keywords_str}}}")
    
    return '\n'.join(latex_commands)

def validate_metadata(meta_data):
    """Validate metadata for common issues."""
    
    issues = []
    warnings = []
    
    # Check title length
    title = meta_data.get('title', '')
    if len(title) > 200:
        warnings.append(f"Title is very long ({len(title)} chars). Consider shortening.")
    
    # Check abstract length
    abstract = meta_data.get('abstract', '')
    if len(abstract) < 100:
        warnings.append("Abstract is quite short. Consider expanding.")
    elif len(abstract) > 2000:
        warnings.append("Abstract is very long. Consider shortening.")
    
    # Check author information
    authors = meta_data.get('authors', [])
    if not authors:
        issues.append("No authors specified")
    
    for i, author in enumerate(authors):
        if not author.get('name'):
            issues.append(f"Author {i+1} missing name")
        if not author.get('affiliation'):
            warnings.append(f"Author {i+1} ({author.get('name', 'Unknown')}) missing affiliation")
    
    # Check categories
    categories = meta_data.get('categories', [])
    if not categories:
        issues.append("No arXiv categories specified")
    elif len(categories) > 5:
        warnings.append(f"Many categories specified ({len(categories)}). Consider reducing.")
    
    return issues, warnings

def main():
    parser = argparse.ArgumentParser(description='Process arXiv metadata for QMANN paper')
    parser.add_argument('--input', '-i', default='paper/meta.yaml', 
                       help='Input metadata YAML file')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--latex', action='store_true', 
                       help='Generate LaTeX metadata commands')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate metadata only')
    parser.add_argument('--check', action='store_true', 
                       help='Check for common issues')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)
    
    # Load metadata
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            meta_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Validate metadata
    if args.validate or args.check:
        try:
            issues, warnings = validate_metadata(meta_data)
            
            if issues:
                print("Issues found:")
                for issue in issues:
                    print(f"  ❌ {issue}")
            
            if warnings:
                print("Warnings:")
                for warning in warnings:
                    print(f"  ⚠️  {warning}")
            
            if not issues and not warnings:
                print("✅ Metadata validation passed")
            
            if args.validate:
                sys.exit(1 if issues else 0)
        
        except Exception as e:
            print(f"Validation error: {e}")
            sys.exit(1)
    
    # Generate arXiv metadata
    try:
        arxiv_meta = generate_arxiv_metadata(meta_data)
        
        if args.latex:
            # Generate LaTeX commands
            latex_meta = generate_latex_metadata(meta_data)
            print(latex_meta)
        else:
            # Output JSON
            if args.output:
                output_path = Path(args.output)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(arxiv_meta, f, indent=2, ensure_ascii=False)
                print(f"arXiv metadata written to {output_path}")
            else:
                print(json.dumps(arxiv_meta, indent=2, ensure_ascii=False))
    
    except Exception as e:
        print(f"Error generating metadata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
