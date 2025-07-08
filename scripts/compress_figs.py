#!/usr/bin/env python3
"""
Compress figures for arXiv submission.

This script optimizes figures to meet arXiv size requirements while maintaining quality.
"""

import argparse
import subprocess
import shutil
from pathlib import Path
import os
from PIL import Image
import sys

def get_file_size_mb(filepath):
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)

def compress_pdf(input_path, output_path, quality='prepress'):
    """Compress PDF using Ghostscript."""
    
    gs_command = [
        'gs',
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        f'-dPDFSETTINGS=/{quality}',
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        f'-sOutputFile={output_path}',
        str(input_path)
    ]
    
    try:
        subprocess.run(gs_command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error compressing PDF {input_path}: {e}")
        return False
    except FileNotFoundError:
        print("Ghostscript not found. Please install ghostscript.")
        return False

def compress_png(input_path, output_path, quality=85):
    """Compress PNG using PIL."""
    
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            # Save with optimization
            img.save(output_path, 'PNG', optimize=True, quality=quality)
        return True
    except Exception as e:
        print(f"Error compressing PNG {input_path}: {e}")
        return False

def compress_jpg(input_path, output_path, quality=85):
    """Compress JPEG using PIL."""
    
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save with optimization
            img.save(output_path, 'JPEG', optimize=True, quality=quality)
        return True
    except Exception as e:
        print(f"Error compressing JPEG {input_path}: {e}")
        return False

def optimize_png_with_optipng(filepath):
    """Optimize PNG using optipng if available."""
    
    try:
        subprocess.run(['optipng', '-o7', str(filepath)], 
                      check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def optimize_jpg_with_jpegoptim(filepath, quality=85):
    """Optimize JPEG using jpegoptim if available."""
    
    try:
        subprocess.run(['jpegoptim', f'--max={quality}', str(filepath)], 
                      check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def compress_figure(input_path, output_path=None, max_size_mb=10, quality=85):
    """Compress a single figure file."""
    
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    
    # Check if compression is needed
    original_size = get_file_size_mb(input_path)
    if original_size <= max_size_mb:
        print(f"✓ {input_path.name}: {original_size:.2f} MB (no compression needed)")
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return True
    
    print(f"Compressing {input_path.name}: {original_size:.2f} MB -> ", end="")
    
    # Create backup if compressing in place
    backup_path = None
    if input_path == output_path:
        backup_path = input_path.with_suffix(input_path.suffix + '.backup')
        shutil.copy2(input_path, backup_path)
    
    success = False
    
    # Compress based on file type
    if input_path.suffix.lower() == '.pdf':
        success = compress_pdf(input_path, output_path, quality='ebook')
        if not success or get_file_size_mb(output_path) > max_size_mb:
            success = compress_pdf(input_path, output_path, quality='screen')
    
    elif input_path.suffix.lower() == '.png':
        success = compress_png(input_path, output_path, quality=quality)
        if success:
            # Try additional optimization
            optimize_png_with_optipng(output_path)
    
    elif input_path.suffix.lower() in ['.jpg', '.jpeg']:
        success = compress_jpg(input_path, output_path, quality=quality)
        if success:
            # Try additional optimization
            optimize_jpg_with_jpegoptim(output_path, quality=quality)
    
    else:
        print(f"Unsupported format: {input_path.suffix}")
        return False
    
    if success:
        new_size = get_file_size_mb(output_path)
        compression_ratio = (1 - new_size / original_size) * 100
        
        if new_size <= max_size_mb:
            print(f"{new_size:.2f} MB ({compression_ratio:.1f}% reduction) ✓")
            # Remove backup if successful
            if backup_path and backup_path.exists():
                backup_path.unlink()
            return True
        else:
            print(f"{new_size:.2f} MB (still too large) ✗")
            # Restore backup if compression failed
            if backup_path and backup_path.exists():
                shutil.move(backup_path, input_path)
            return False
    else:
        print("compression failed ✗")
        # Restore backup if compression failed
        if backup_path and backup_path.exists():
            shutil.move(backup_path, input_path)
        return False

def check_figure_sizes(figures_dir, max_size_mb=10):
    """Check all figures in directory for size compliance."""
    
    figures_dir = Path(figures_dir)
    if not figures_dir.exists():
        print(f"Directory {figures_dir} does not exist")
        return False
    
    oversized_files = []
    total_size = 0
    
    # Supported formats
    formats = ['.pdf', '.png', '.jpg', '.jpeg', '.eps', '.svg']
    
    print(f"Checking figures in {figures_dir}...")
    print(f"Size limit: {max_size_mb} MB per file")
    print("-" * 50)
    
    for filepath in figures_dir.iterdir():
        if filepath.is_file() and filepath.suffix.lower() in formats:
            size_mb = get_file_size_mb(filepath)
            total_size += size_mb
            
            status = "✓" if size_mb <= max_size_mb else "✗"
            print(f"{status} {filepath.name}: {size_mb:.2f} MB")
            
            if size_mb > max_size_mb:
                oversized_files.append(filepath)
    
    print("-" * 50)
    print(f"Total size: {total_size:.2f} MB")
    print(f"Oversized files: {len(oversized_files)}")
    
    if oversized_files:
        print("\nOversized files:")
        for filepath in oversized_files:
            print(f"  - {filepath.name}: {get_file_size_mb(filepath):.2f} MB")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Compress figures for arXiv submission')
    parser.add_argument('input', nargs='?', help='Input file or directory')
    parser.add_argument('--output', '-o', help='Output file or directory')
    parser.add_argument('--max-size', type=float, default=10.0, 
                       help='Maximum file size in MB (default: 10)')
    parser.add_argument('--quality', type=int, default=85, 
                       help='Compression quality 1-100 (default: 85)')
    parser.add_argument('--check-only', action='store_true', 
                       help='Only check sizes, do not compress')
    parser.add_argument('--recursive', '-r', action='store_true', 
                       help='Process directories recursively')
    
    args = parser.parse_args()
    
    # Default to paper/figs if no input specified
    if args.input is None:
        args.input = 'paper/figs'
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)
    
    if input_path.is_file():
        # Single file
        output_path = Path(args.output) if args.output else input_path
        
        if args.check_only:
            size_mb = get_file_size_mb(input_path)
            status = "✓" if size_mb <= args.max_size else "✗"
            print(f"{status} {input_path.name}: {size_mb:.2f} MB")
            sys.exit(0 if size_mb <= args.max_size else 1)
        
        success = compress_figure(input_path, output_path, args.max_size, args.quality)
        sys.exit(0 if success else 1)
    
    elif input_path.is_dir():
        # Directory
        if args.check_only:
            success = check_figure_sizes(input_path, args.max_size)
            sys.exit(0 if success else 1)
        
        # Compress all figures in directory
        formats = ['.pdf', '.png', '.jpg', '.jpeg']
        files_to_process = []
        
        if args.recursive:
            for fmt in formats:
                files_to_process.extend(input_path.rglob(f'*{fmt}'))
        else:
            for fmt in formats:
                files_to_process.extend(input_path.glob(f'*{fmt}'))
        
        if not files_to_process:
            print(f"No supported image files found in {input_path}")
            sys.exit(0)
        
        print(f"Processing {len(files_to_process)} files...")
        
        success_count = 0
        for filepath in files_to_process:
            if compress_figure(filepath, max_size_mb=args.max_size, quality=args.quality):
                success_count += 1
        
        print(f"\nProcessed {success_count}/{len(files_to_process)} files successfully")
        
        # Final size check
        print("\nFinal size check:")
        all_good = check_figure_sizes(input_path, args.max_size)
        sys.exit(0 if all_good else 1)
    
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
