#!/usr/bin/env python3
"""
HAR Batch Experiment Runner
Runs multiple experiments from a list of YAML configurations.
"""

import argparse
import yaml
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def run_single_experiment(config_file, max_workers=1, dry_run=False):
    """Run a single experiment from config file"""
    
    print(f"Loading configuration from {config_file}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build command using config_builder
    from config_builder import build_command_from_config
    command = build_command_from_config(config_file)
    
    if dry_run:
        print(f"DRY RUN: Would execute: {command}")
        return True, "Dry run completed"
    
    print(f"Executing: {command}")
    start_time = time.time()
    
    try:
        # Change to repo directory
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            command.split(),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {config_file} completed successfully in {duration:.1f}s")
            return True, f"Completed in {duration:.1f}s"
        else:
            print(f"✗ {config_file} failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, f"Failed with return code {result.returncode}"
            
    except subprocess.TimeoutExpired:
        print(f"✗ {config_file} timed out after 1 hour")
        return False, "Timed out"
    except Exception as e:
        print(f"✗ {config_file} failed with error: {e}")
        return False, f"Error: {e}"

def run_batch_experiments(config_files, max_workers=1, dry_run=False, stop_on_failure=False):
    """Run multiple experiments"""
    
    print(f"Running {len(config_files)} experiments with max_workers={max_workers}")
    if dry_run:
        print("DRY RUN MODE - No actual experiments will be executed")
    
    results = []
    start_time = time.time()
    
    if max_workers == 1:
        # Sequential execution
        for config_file in config_files:
            success, message = run_single_experiment(config_file, max_workers, dry_run)
            results.append((config_file, success, message))
            
            if not success and stop_on_failure:
                print(f"Stopping batch due to failure in {config_file}")
                break
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(run_single_experiment, config_file, 1, dry_run): config_file 
                for config_file in config_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_config):
                config_file = future_to_config[future]
                try:
                    success, message = future.result()
                    results.append((config_file, success, message))
                    
                    if not success and stop_on_failure:
                        print(f"Stopping batch due to failure in {config_file}")
                        # Cancel remaining futures
                        for f in future_to_config:
                            f.cancel()
                        break
                        
                except Exception as e:
                    print(f"✗ {config_file} generated exception: {e}")
                    results.append((config_file, False, f"Exception: {e}"))
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"\n{'='*50}")
    print(f"BATCH SUMMARY")
    print(f"{'='*50}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per experiment: {total_time/len(results):.1f}s")
    
    if failed > 0:
        print(f"\nFailed experiments:")
        for config_file, success, message in results:
            if not success:
                print(f"  - {config_file}: {message}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run multiple HAR experiments from YAML configs")
    parser.add_argument("config_files", nargs="+", help="YAML configuration files")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be executed without running")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop batch if any experiment fails")
    parser.add_argument("--config-list", help="File containing list of config files (one per line)")
    
    args = parser.parse_args()
    
    # Get config files
    config_files = args.config_files
    
    if args.config_list:
        with open(args.config_list, 'r') as f:
            config_files.extend([line.strip() for line in f if line.strip()])
    
    # Validate config files exist
    missing_files = [f for f in config_files if not Path(f).exists()]
    if missing_files:
        print(f"Error: Configuration files not found: {missing_files}")
        sys.exit(1)
    
    # Run experiments
    results = run_batch_experiments(
        config_files, 
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        stop_on_failure=args.stop_on_failure
    )
    
    # Exit with error code if any failed
    if any(not success for _, success, _ in results):
        sys.exit(1)

if __name__ == "__main__":
    main()
