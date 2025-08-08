"""
Lineage Chain to Event Mapper

This module maps lineage chain JSON data to lineage event format by traversing
dependency chains and creating source-target mappings.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import os
from .utils.logging_config import get_logger

class LineageEventMapper:
    """Maps lineage chain data to lineage event format."""
    
    def __init__(self):
        self.event_version = "1.0.0"
        self.processed_mappings: Set[str] = set()
        self.dialect: str = ""
        self.chain_type: str = ""
        self.entity_metadata_cache: Dict[str, Dict[str, Any]] = {}  # Track metadata for each entity across paths
        self.logger = get_logger("lineage_mapper")
    
    def _find_entity_metadata_in_chains(self, entity_name: str, chains: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recursively search for entity metadata in nested dependency chains."""
        self.logger.debug(f"Searching for metadata for entity: {entity_name}")
        
        if not chains:
            self.logger.warning(f"No chains provided for entity metadata search: {entity_name}")
            return None
        
        # Return cached merged metadata if available
        if entity_name in self.entity_metadata_cache:
            self.logger.debug(f"Found cached metadata for entity: {entity_name}")
            return self.entity_metadata_cache[entity_name]
            
        # Search in top-level chains first
        for chain_name, chain_data in chains.items():
            if chain_name == entity_name:
                metadata = chain_data.get("metadata", {})
                self._cache_entity_metadata(entity_name, metadata)
                self.logger.debug(f"Found metadata for entity {entity_name} in top-level chains")
                return metadata
            
            # Recursively search in dependencies
            dependencies = chain_data.get("dependencies", [])
            for dep in dependencies:
                if dep.get("entity") == entity_name:
                    metadata = dep.get("metadata", {})
                    self._cache_entity_metadata(entity_name, metadata)
                    self.logger.debug(f"Found metadata for entity {entity_name} in dependencies of {chain_name}")
                    return metadata
                
                # Recursively search deeper
                nested_result = self._search_in_dependencies(entity_name, [dep])
                if nested_result:
                    self._cache_entity_metadata(entity_name, nested_result)
                    self.logger.debug(f"Found metadata for entity {entity_name} in nested dependencies")
                    return nested_result
        
        self.logger.warning(f"Could not find metadata for entity: {entity_name}")
        return None
    
    def _search_in_dependencies(self, entity_name: str, dependencies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Helper method to search recursively in dependency list."""
        self.logger.debug(f"Searching in dependencies for entity: {entity_name}")
        
        for dep in dependencies:
            if dep.get("entity") == entity_name:
                self.logger.debug(f"Found entity {entity_name} in current dependency level")
                return dep.get("metadata", {})
            
            # Search deeper in nested dependencies
            nested_deps = dep.get("dependencies", [])
            if nested_deps:
                nested_result = self._search_in_dependencies(entity_name, nested_deps)
                if nested_result:
                    return nested_result
        
        self.logger.debug(f"Entity {entity_name} not found in dependency tree")
        return None
    
    def _cache_entity_metadata(self, entity_name: str, metadata: Dict[str, Any]) -> None:
        """Cache and merge metadata for an entity across multiple paths."""
        if entity_name not in self.entity_metadata_cache:
            self.entity_metadata_cache[entity_name] = metadata.copy()
            self.logger.debug(f"Cached metadata for entity: {entity_name}")
        else:
            # Merge metadata from multiple paths
            self.entity_metadata_cache[entity_name] = self._merge_metadata(
                self.entity_metadata_cache[entity_name], 
                metadata
            )
            self.logger.debug(f"Merged metadata for entity: {entity_name}")
    
    def _merge_metadata(self, existing_metadata: Dict[str, Any], new_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two metadata dictionaries, combining columns and preserving transformations."""
        try:
            self.logger.debug("Merging metadata from multiple paths")
            merged = existing_metadata.copy()
            
            # Merge table columns - combine and deduplicate by column name
            existing_columns = merged.get("table_columns", [])
            new_columns = new_metadata.get("table_columns", [])
            
            self.logger.debug(f"Merging {len(existing_columns)} existing columns with {len(new_columns)} new columns")
            
            # Create a map of existing columns by name
            column_map = {col.get("name"): col for col in existing_columns if col.get("name")}
            
            # Add new columns or merge existing ones
            merged_columns = 0
            new_columns_added = 0
            for new_col in new_columns:
                col_name = new_col.get("name")
                if not col_name:
                    continue
                    
                if col_name in column_map:
                    # Column exists, merge upstream sources and transformations
                    existing_col = column_map[col_name]
                    
                    # Merge upstream sources
                    existing_upstream = set(existing_col.get("upstream", []))
                    new_upstream = set(new_col.get("upstream", []))
                    merged_upstream = list(existing_upstream.union(new_upstream))
                    if merged_upstream:
                        existing_col["upstream"] = merged_upstream
                    
                    # Preserve transformation if it exists
                    if new_col.get("transformation") and not existing_col.get("transformation"):
                        existing_col["transformation"] = new_col["transformation"]
                    
                    merged_columns += 1
                else:
                    # New column, add it
                    column_map[col_name] = new_col.copy()
                    new_columns_added += 1
            
            self.logger.debug(f"Merged {merged_columns} columns, added {new_columns_added} new columns")
            
            # Update merged metadata with combined columns
            merged["table_columns"] = list(column_map.values())
            
            # Merge other metadata properties (is_cte, table_type, etc.)
            for key, value in new_metadata.items():
                if key != "table_columns" and key not in merged:
                    merged[key] = value
            
            self.logger.debug("Successfully merged metadata")
            return merged
        
        except Exception as e:
            self.logger.error(f"Failed to merge metadata: {str(e)}")
            raise
    
    def _cache_entity_transformations(self, entity_name: str, transformations: List[Dict[str, Any]]) -> None:
        """Cache and merge transformations for an entity across multiple paths."""
        cache_key = f"{entity_name}_transformations"
        
        if cache_key not in self.entity_metadata_cache:
            self.entity_metadata_cache[cache_key] = transformations.copy()
        else:
            # Merge transformations from multiple paths
            existing_transformations = self.entity_metadata_cache[cache_key]
            merged_transformations = self._merge_transformations(existing_transformations, transformations)
            self.entity_metadata_cache[cache_key] = merged_transformations
    
    def _merge_transformations(self, existing_transformations: List[Dict[str, Any]], new_transformations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge transformation arrays from multiple paths, combining unique transformations."""
        merged = []
        
        # Create a set to track unique transformations
        seen_transformations = set()
        
        # Process existing transformations first
        for transform in existing_transformations:
            transform_key = self._get_transformation_key(transform)
            if transform_key not in seen_transformations:
                merged.append(transform.copy())
                seen_transformations.add(transform_key)
        
        # Add new unique transformations
        for transform in new_transformations:
            transform_key = self._get_transformation_key(transform)
            if transform_key not in seen_transformations:
                merged.append(transform.copy())
                seen_transformations.add(transform_key)
            else:
                # Same transformation exists, merge specific properties like joins, filters
                self._merge_duplicate_transformation(merged, transform, transform_key)
        
        return merged
    
    def _get_transformation_key(self, transformation: Dict[str, Any]) -> str:
        """Generate a unique key for a transformation based on its core properties."""
        transform_type = transformation.get("type", "")
        source_table = transformation.get("source_table", "")
        target_table = transformation.get("target_table", "")
        return f"{transform_type}:{source_table}->{target_table}"
    
    def _merge_duplicate_transformation(self, merged_list: List[Dict[str, Any]], new_transform: Dict[str, Any], transform_key: str) -> None:
        """Merge properties of duplicate transformations (same source->target)."""
        # Find the existing transformation to merge with
        for existing_transform in merged_list:
            if self._get_transformation_key(existing_transform) == transform_key:
                # Merge joins
                existing_joins = existing_transform.get("joins", [])
                new_joins = new_transform.get("joins", [])
                merged_joins = self._merge_joins(existing_joins, new_joins)
                if merged_joins:
                    existing_transform["joins"] = merged_joins
                
                # Merge filter conditions
                existing_filters = existing_transform.get("filter_conditions", [])
                new_filters = new_transform.get("filter_conditions", [])
                merged_filters = self._merge_filter_conditions(existing_filters, new_filters)
                if merged_filters:
                    existing_transform["filter_conditions"] = merged_filters
                
                # Merge group by columns
                existing_group_by = set(existing_transform.get("group_by_columns", []))
                new_group_by = set(new_transform.get("group_by_columns", []))
                merged_group_by = list(existing_group_by.union(new_group_by))
                if merged_group_by:
                    existing_transform["group_by_columns"] = merged_group_by
                
                # Merge order by columns
                existing_order_by = existing_transform.get("order_by_columns", [])
                new_order_by = new_transform.get("order_by_columns", [])
                if new_order_by and new_order_by not in [existing_order_by]:
                    # Combine order by columns (preserve order)
                    combined_order_by = existing_order_by + [col for col in new_order_by if col not in existing_order_by]
                    existing_transform["order_by_columns"] = combined_order_by
                
                break
    
    def _merge_joins(self, existing_joins: List[Dict[str, Any]], new_joins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge join conditions, avoiding duplicates."""
        merged = existing_joins.copy()
        
        for new_join in new_joins:
            # Check if this join already exists
            join_key = f"{new_join.get('join_type', '')}:{new_join.get('right_table', '')}"
            existing_keys = [f"{j.get('join_type', '')}:{j.get('right_table', '')}" for j in merged]
            
            if join_key not in existing_keys:
                merged.append(new_join.copy())
        
        return merged
    
    def _merge_filter_conditions(self, existing_filters: List[Dict[str, Any]], new_filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge filter conditions, avoiding duplicates."""
        merged = existing_filters.copy()
        
        for new_filter in new_filters:
            # Simple deduplication based on filter type and content
            filter_str = json.dumps(new_filter, sort_keys=True)
            existing_strs = [json.dumps(f, sort_keys=True) for f in merged]
            
            if filter_str not in existing_strs:
                merged.append(new_filter.copy())
        
        return merged
    
    def create_lineage_event(
        self, 
        source_table: str,
        source_type: str,
        target_table: str, 
        target_type: str,
        metadata: Dict[str, Any],
        tenant_id: str = "",
        association_type: str = "",
        association_id: str = "",
        query_id: str = ""
    ) -> Dict[str, Any]:
        """Create a single lineage event."""
        self.logger.debug(f"Creating lineage event: {source_table} -> {target_table}")
        
        event_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        event = {
            "event_id": event_id,
            "event_version": self.event_version,
            "timekey": timestamp[:10].replace("-", ""),  # YYYYMMDD format
            "timestamp": timestamp,
            "tenant_id": tenant_id,
            "workspace_id": tenant_id, 
            "user_id": "",
            "query_id": query_id,
            "source_table": source_table,
            "source_type": source_type,
            "source_id": f"{source_type.lower()}_{source_table.replace('.', '_')}",
            "source_path": "",
            "target_table": target_table,
            "target_type": target_type,
            "target_id": f"{target_type.lower()}_{target_table.replace('.', '_')}",
            "target_path": "",
            "metadata": metadata,
            "associated_type": association_type,
            "associated_id": association_id,
            "state": "active"
        }
        
        self.logger.debug(f"Successfully created lineage event with ID: {event_id}")
        return event
    
    def traverse_dependencies(
        self, 
        entity_data: Dict[str, Any], 
        parent_entity: str,
        parent_type: str,
        events: List[Dict[str, Any]],
        tenant_id: str,
        association_type: str,
        association_id: str,
        query_id: str,
        chains: Dict[str, Any] = None,
        visited: Set[str] = None
    ) -> None:
        """Recursively traverse dependency chains to create lineage events."""
        try:
            if visited is None:
                visited = set()
                
            entity_name = entity_data.get("entity", "")
            entity_type = entity_data.get("entity_type", "table")
            
            self.logger.debug(f"Traversing dependency: {parent_entity} -> {entity_name}")
            
            # Avoid infinite loops
            mapping_key = f"{parent_entity}->{entity_name}"
            if mapping_key in visited or mapping_key in self.processed_mappings:
                self.logger.debug(f"Skipping already processed mapping: {mapping_key}")
                return
                
            visited.add(mapping_key)
            self.processed_mappings.add(mapping_key)
        
            # Count columns in target (current entity)
            target_metadata = entity_data.get("metadata", {})
            target_columns = target_metadata.get("table_columns", [])
            target_column_count = len(target_columns)
            target_depth = entity_data.get("depth", 0)
            
            self.logger.debug(f"Target entity {entity_name} has {target_column_count} columns")
            
            # Count columns in source (parent entity) by looking in chains
            source_column_count = 0
            source_metadata = {}
            source_depth = 0
            if chains and parent_entity in chains:
                # Parent entity is in top-level chains
                source_metadata = chains[parent_entity].get("metadata", {})
                parent_columns = source_metadata.get("table_columns", [])
                source_column_count = len(parent_columns)
                source_depth = chains[parent_entity].get("depth", 0)
                self.logger.debug(f"Found source entity {parent_entity} in top-level chains with {source_column_count} columns")
            else:
                # Search for parent entity in nested dependencies
                found_metadata = self._find_entity_metadata_in_chains(parent_entity, chains)
                if found_metadata:
                    source_metadata = found_metadata
                    parent_columns = source_metadata.get("table_columns", [])
                    source_column_count = len(parent_columns)
                    # For nested entities, we need to find depth by searching the dependency tree
                    # For now, set depth to 0 as it's complex to determine from nested structure
                    source_depth = 0
                    self.logger.debug(f"Found source entity {parent_entity} in nested dependencies with {source_column_count} columns")
                else:
                    # Could not get source columns from chains
                    self.logger.warning(f"Could not get source columns for {parent_entity} -> {entity_name}")
                    print(f"  Debug: Could not get source columns for {parent_entity} -> {entity_name}")
        
            # Get transformations from the entity data
            transformations = entity_data.get("transformations", [])
            
            # Cache transformations for this entity to support merging across paths
            if transformations:
                self._cache_entity_transformations(entity_name, transformations)
                self.logger.debug(f"Cached {len(transformations)} transformations for entity {entity_name}")
            
            # Get merged transformations for this entity (if multiple paths reach it)
            transformation_cache_key = f"{entity_name}_transformations"
            merged_transformations = self.entity_metadata_cache.get(transformation_cache_key, transformations)
            
            # Create metadata for the relationship
            metadata = {
                "source_column_count": source_column_count,
                "target_column_count": target_column_count,
                "source_depth": source_depth,
                "target_depth": target_depth,
                "dialect": self.dialect,
                "chain_type": self.chain_type,
                "source_metadata": source_metadata,
                "target_metadata": target_metadata,
                "transformations": merged_transformations
            }
            
            # Create lineage event from parent to current entity
            if parent_entity != entity_name:  # Avoid self-references
                event = self.create_lineage_event(
                    source_table=parent_entity,
                    source_type=parent_type.upper(),
                    target_table=entity_name,
                    target_type=entity_type.upper(),
                    metadata=metadata,
                    tenant_id=tenant_id,
                    association_type=association_type,
                    association_id=association_id,
                    query_id=query_id
                )
                events.append(event)
                self.logger.debug(f"Added lineage event: {parent_entity} -> {entity_name}")
            
            # Recursively process dependencies
            dependencies = entity_data.get("dependencies", [])
            self.logger.debug(f"Processing {len(dependencies)} dependencies for entity {entity_name}")
            
            for dep in dependencies:
                self.traverse_dependencies(
                    dep, 
                    entity_name,
                    entity_type, 
                    events,
                    tenant_id,
                    association_type,  # Pass association parameters down
                    association_id,
                    query_id,
                    chains,  # Pass chains down
                    visited  # Pass same visited set to prevent infinite loops
                )
                
        except Exception as e:
            self.logger.error(f"Failed to traverse dependencies for {parent_entity} -> {entity_name}: {str(e)}")
            raise
    
    def map_lineage_chain_to_events(
        self, 
        lineage_json_string: str, 
        tenant_id: str = "",
        association_type: str = "", 
        association_id: str = "",
        query_id: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Map a lineage chain JSON string to lineage events.
        
        Args:
            lineage_json_string: JSON string containing lineage chain data
            tenant_id: Tenant ID for the lineage events
            association_type: Type of association for the lineage events
            association_id: ID of association for the lineage events
            query_id: Query ID for the lineage events
            
        Returns:
            List of lineage event dictionaries
        """
        try:
            self.logger.info("Starting lineage chain to events mapping")
            
            lineage_data = json.loads(lineage_json_string)
            
            events = []
            chains = lineage_data.get("chains", {})
            
            self.logger.debug(f"Processing {len(chains)} chains with query_id: {query_id}")
            
            # Reset processed mappings, metadata cache, and set global properties for each file
            self.processed_mappings.clear()
            self.entity_metadata_cache.clear()
            self.dialect = lineage_data.get("dialect", "")
            self.chain_type = lineage_data.get("chain_type", "")
            
            self.logger.debug(f"Set dialect: {self.dialect}, chain_type: {self.chain_type}")
            
            # Process each base entity and its dependency chain
            for base_entity_name, chain_data in chains.items():
                self.logger.debug(f"Processing base entity: {base_entity_name}")
                
                self.traverse_dependencies(
                    chain_data,
                    base_entity_name,
                    chain_data.get("entity_type", "table"),
                    events,
                    tenant_id,
                    association_type,  # Pass association parameters
                    association_id,
                    query_id,
                    chains,  # Pass chains for source column lookup
                    None  # visited set (will be initialized in method)
                )
            
            self.logger.info(f"Successfully generated {len(events)} lineage events")
            return events
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON input: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to map lineage chain to events: {str(e)}")
            raise
    
    
    def print_events_summary(self, events: List[Dict[str, Any]]) -> None:
        """Print a summary of the generated events."""
        self.logger.info(f"Printing summary for {len(events)} lineage events")
        
        print(f"\nGenerated {len(events)} lineage events:")
        print("-" * 50)
        
        for i, event in enumerate(events, 1):
            print(f"{i}. {event['source_table']} ({event['source_type']}) -> "
                  f"{event['target_table']} ({event['target_type']})")
        
        # Count by relationship types
        relationships = {}
        for event in events:
            rel_type = f"{event['source_type']}->{event['target_type']}"
            relationships[rel_type] = relationships.get(rel_type, 0) + 1
        
        print(f"\nRelationship types:")
        for rel_type, count in relationships.items():
            print(f"  {rel_type}: {count}")
            
        self.logger.info("Successfully printed events summary")
