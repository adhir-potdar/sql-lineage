"""
Utility functions for processing lineage chains.

This module provides utilities for:
- Calculating chain depths
- Detecting transformations and metadata in chains
- Integrating column transformations
- Chain structure manipulation
"""

from typing import Dict, Any, List, Optional


def calculate_max_depth(chains: Dict[str, Any]) -> int:
    """Calculate the maximum depth in the chains."""
    max_depth = 0
    
    def get_chain_depth(chain_entity: Dict[str, Any]) -> int:
        current_depth = chain_entity.get('depth', 0)
        dependencies = chain_entity.get('dependencies', [])
        
        if not dependencies:
            return current_depth
        
        max_dep_depth = max(get_chain_depth(dep) for dep in dependencies)
        return max(current_depth, max_dep_depth)
    
    for chain in chains.values():
        chain_depth = get_chain_depth(chain)
        max_depth = max(max_depth, chain_depth)
    
    return max_depth


def detect_transformations_in_chains(chains: Dict) -> bool:
    """Check if there are transformations present in the chains."""
    if not chains:
        return False
    
    def has_transformations(entity_data: Dict) -> bool:
        # Check dependencies for transformations
        for dep in entity_data.get('dependencies', []):
            if dep.get('transformations'):
                return True
            # Recursively check nested dependencies
            if has_transformations(dep):
                return True
        return False
    
    for entity_data in chains.values():
        if has_transformations(entity_data):
            return True
    
    return False


def detect_metadata_in_chains(chains: Dict) -> bool:
    """Check if there is metadata present in the chains."""
    if not chains:
        return False
    
    def has_metadata(entity_data: Dict) -> bool:
        # Check if entity has metadata
        if entity_data.get('metadata'):
            return True
        
        # Check dependencies for metadata
        for dep in entity_data.get('dependencies', []):
            if dep.get('metadata'):
                return True
            # Recursively check nested dependencies
            if has_metadata(dep):
                return True
        return False
    
    for entity_data in chains.values():
        if has_metadata(entity_data):
            return True
    
    return False


def integrate_column_transformations(chains: Dict, sql: str = None) -> None:
    """Integrate column transformations into the chains."""
    # This is a placeholder for column transformation integration logic
    # The actual implementation would depend on specific requirements
    # for how column transformations should be integrated into the chain structure
    
    if not chains or not sql:
        return
    
    # Implementation would go here
    # For now, this is a no-op to maintain compatibility
    pass


def find_entity_in_chains(chains: Dict[str, Any], entity_name: str) -> Optional[Dict[str, Any]]:
    """Find a specific entity in the chains structure."""
    if entity_name in chains:
        return chains[entity_name]
    
    # Search in nested dependencies
    for entity_data in chains.values():
        result = _find_entity_in_dependencies(entity_data.get('dependencies', []), entity_name)
        if result:
            return result
    
    return None


def _find_entity_in_dependencies(dependencies: List[Dict[str, Any]], entity_name: str) -> Optional[Dict[str, Any]]:
    """Helper function to search for entity in dependencies recursively."""
    for dep in dependencies:
        if dep.get('entity') == entity_name:
            return dep
        
        # Search nested dependencies
        nested_result = _find_entity_in_dependencies(dep.get('dependencies', []), entity_name)
        if nested_result:
            return nested_result
    
    return None


def get_all_entities_at_depth(chains: Dict[str, Any], target_depth: int) -> List[str]:
    """Get all entity names at a specific depth level."""
    entities = []
    
    def collect_entities_at_depth(entity_data: Dict[str, Any], current_depth: int = 0):
        if current_depth == target_depth:
            entity_name = entity_data.get('entity')
            if entity_name and entity_name not in entities:
                entities.append(entity_name)
        
        # Search in dependencies
        for dep in entity_data.get('dependencies', []):
            collect_entities_at_depth(dep, current_depth + 1)
    
    for entity_data in chains.values():
        collect_entities_at_depth(entity_data)
    
    return entities


def count_total_entities(chains: Dict[str, Any]) -> int:
    """Count total number of unique entities in all chains."""
    unique_entities = set()
    
    def collect_entities(entity_data: Dict[str, Any]):
        entity_name = entity_data.get('entity')
        if entity_name:
            unique_entities.add(entity_name)
        
        # Collect from dependencies
        for dep in entity_data.get('dependencies', []):
            collect_entities(dep)
    
    for entity_data in chains.values():
        collect_entities(entity_data)
    
    return len(unique_entities)


def get_chain_summary(chains: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of the chains structure."""
    return {
        'total_entities': count_total_entities(chains),
        'max_depth': calculate_max_depth(chains),
        'has_transformations': detect_transformations_in_chains(chains),
        'has_metadata': detect_metadata_in_chains(chains),
        'chain_count': len(chains)
    }