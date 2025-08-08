#!/usr/bin/env python3
"""
Test file for Lineage Chain to Event Mapper

This module contains the main function and utilities to test the lineage mapper
by processing all JSON files from the output folder and generating lineage events.
"""

import json
import os
import uuid
from src.analyzer.lineage_mapper import LineageEventMapper


def save_events_to_file(events, output_path):
    """Save lineage events to a JSON file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(events, f, indent=2)
    print(f"  Saved {len(events)} lineage events to {output_path}")


def main():
    """Main function to process all JSON files from output folder."""
    mapper = LineageEventMapper()
    
    # Define input and output directories
    input_dir = "output"
    events_output_dir = "output/events"
    
    try:
        # Get all JSON files from the output directory (exclude existing events files)
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} does not exist")
            return
            
        json_files = [f for f in os.listdir(input_dir) 
                     if f.endswith('.json') and not f.endswith('_events.json')]
        
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return
            
        print(f"Found {len(json_files)} JSON files in {input_dir}")
        print(f"Events will be saved to: {events_output_dir}")
        print("-" * 50)
        
        total_events_processed = 0
        
        # Process each JSON file
        for json_file in sorted(json_files):
            input_file_path = os.path.join(input_dir, json_file)
            
            # Create output filename based on input filename
            base_name = os.path.splitext(json_file)[0]
            output_file_path = os.path.join(events_output_dir, f"{base_name}_events.json")
            
            print(f"\nProcessing: {json_file}")
            
            try:
                # Read JSON file content
                with open(input_file_path, 'r') as f:
                    json_content = f.read()
                
                # Reset processed mappings and metadata cache for each file
                mapper.processed_mappings.clear()
                mapper.entity_metadata_cache.clear()
                
                # Map lineage chain to events with default association parameters
                events = mapper.map_lineage_chain_to_events(
                    json_content, 
                    tenant_id=str(uuid.uuid4()),
                    association_type="DATAMAP",
                    association_id=str(uuid.uuid4()),
                    query_id=str(uuid.uuid4())
                )
                
                if events:
                    # Print summary for this file
                    print(f"  Generated {len(events)} lineage events")
                    
                    # Save to individual output file
                    save_events_to_file(events, output_file_path)
                    total_events_processed += len(events)
                else:
                    print(f"  No lineage events generated (empty chains or no dependencies)")
                    
            except json.JSONDecodeError as e:
                print(f"  Error: Invalid JSON format in {json_file} - {e}")
                continue
            except Exception as e:
                print(f"  Error processing {json_file}: {e}")
                continue
        
        print(f"\n" + "=" * 50)
        print(f"Processing completed!")
        print(f"Total files processed: {len(json_files)}")
        print(f"Total lineage events generated: {total_events_processed}")
        print(f"Output events saved to: {events_output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()