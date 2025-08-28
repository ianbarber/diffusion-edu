#!/usr/bin/env python3
"""
Cleanup script to organize and archive old experiment files.
Keeps the repository clean while preserving important results.
"""

import os
import shutil
from pathlib import Path
import argparse
import json
from datetime import datetime

def cleanup_old_files(dry_run=False):
    """Clean up and organize old experiment files."""
    
    print("🧹 Starting cleanup process...")
    
    # Define what to clean up
    cleanup_items = {
        # Old scripts to remove
        'scripts': [
            'train_all_mnist.sh',
            'train_iso_param.sh', 
            'train_iso_dit_mmdit.sh',
            'train_large_iso.sh',
            'archive_old_configs.sh',
            'find_iso_params.py',
            'find_larger_iso_params.py',
            'compare_iso_params.py',
            'estimate_training_time.py',
            'benchmark_performance.py',
            'test_setup.py',
            'quick_start.py'
        ],
        
        # Old experiment directories to archive
        'experiment_dirs': [
            'experiments_mnist',
            'experiments_mnist_old_backup'
        ],
        
        # Generated files to archive
        'generated_files': [
            'comparison.png',
            'comparison_data.json',
            'comparison_report.md',
            'generation_speed.png',
            'iso_param_analysis.png',
            'iso_param_comparison.png',
            'performance_benchmark.png',
            'comparison_samples'
        ]
    }
    
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = Path(f'archive_{timestamp}')
    
    if not dry_run:
        archive_dir.mkdir(exist_ok=True)
        print(f"📁 Created archive directory: {archive_dir}")
    else:
        print(f"📁 Would create archive directory: {archive_dir}")
    
    # Process each category
    removed_count = 0
    archived_count = 0
    
    # Remove old scripts
    print("\n📜 Cleaning up old scripts...")
    for script in cleanup_items['scripts']:
        script_path = Path(script)
        if script_path.exists():
            if dry_run:
                print(f"  Would remove: {script}")
            else:
                script_path.unlink()
                print(f"  ✓ Removed: {script}")
            removed_count += 1
    
    # Archive experiment directories
    print("\n📊 Archiving old experiment directories...")
    for exp_dir in cleanup_items['experiment_dirs']:
        exp_path = Path(exp_dir)
        if exp_path.exists():
            archive_exp_path = archive_dir / exp_dir
            if dry_run:
                print(f"  Would archive: {exp_dir} -> {archive_exp_path}")
            else:
                shutil.move(str(exp_path), str(archive_exp_path))
                print(f"  ✓ Archived: {exp_dir} -> {archive_exp_path}")
            archived_count += 1
    
    # Archive generated files
    print("\n🖼️ Archiving generated files...")
    for gen_file in cleanup_items['generated_files']:
        gen_path = Path(gen_file)
        if gen_path.exists():
            archive_gen_path = archive_dir / gen_file
            if dry_run:
                print(f"  Would archive: {gen_file} -> {archive_gen_path}")
            else:
                if gen_path.is_dir():
                    shutil.move(str(gen_path), str(archive_gen_path))
                else:
                    shutil.move(str(gen_path), str(archive_gen_path))
                print(f"  ✓ Archived: {gen_file} -> {archive_gen_path}")
            archived_count += 1
    
    # Create new experiments directory structure
    print("\n📂 Setting up clean directory structure...")
    new_dirs = [
        'experiments',
        'experiments/unified_sd1_unet_eps',
        'experiments/unified_sd2_unet_v', 
        'experiments/unified_sd3_mmdit_flow',
        'experiments/reference',
        'outputs',
        'outputs/samples',
        'outputs/comparisons'
    ]
    
    for new_dir in new_dirs:
        dir_path = Path(new_dir)
        if not dry_run:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {new_dir}")
        else:
            print(f"  Would create: {new_dir}")
    
    # Summary
    print("\n" + "="*50)
    print("✅ Cleanup Summary:")
    print(f"  - Removed {removed_count} old scripts")
    print(f"  - Archived {archived_count} items to {archive_dir}")
    print(f"  - Created clean directory structure")
    
    if dry_run:
        print("\n⚠️  This was a DRY RUN - no actual changes were made")
        print("Run without --dry-run to perform actual cleanup")
    else:
        print("\n✨ Cleanup complete! Repository is now organized.")
        
        # Create a manifest of what was archived
        manifest = {
            'timestamp': timestamp,
            'removed_scripts': cleanup_items['scripts'],
            'archived_items': cleanup_items['experiment_dirs'] + cleanup_items['generated_files'],
            'removed_count': removed_count,
            'archived_count': archived_count
        }
        
        manifest_path = archive_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\n📋 Archive manifest saved to: {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description='Clean up old experiment files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be cleaned up without actually doing it')
    parser.add_argument('--keep-experiments', action='store_true',
                       help='Keep old experiment directories (only remove scripts)')
    args = parser.parse_args()
    
    cleanup_old_files(dry_run=args.dry_run)

if __name__ == "__main__":
    main()