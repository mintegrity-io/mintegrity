#!/usr/bin/env python3
"""
CLI Utils
Extracted from RocketPoolGroupsAnalyzer for reuse across all protocol analyzers
"""

import sys
from pathlib import Path

def print_startup_info(protocol_name: str):
    """
    Print startup information for the analyzer
    
    Args:
        protocol_name (str): Name of the protocol being analyzed
    """
    print(f"🚀 Starting {protocol_name} Groups Analysis")
    print(f"Working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).resolve()}")

def parse_graph_file_argument(protocol_name: str):
    """
    Parse and validate graph file argument from command line
    
    Args:
        protocol_name (str): Name of the protocol for error messages
        
    Returns:
        str: Path to the graph file
        
    Exits:
        System exit with code 1 if arguments are invalid or file doesn't exist
    """
    # Check for required argument
    if len(sys.argv) < 2:
        print(f"❌ Usage: python {protocol_name.lower().replace(' ', '_')}_groups_analyzer.py <graph_file_path>")
        print(f"   Example: python {protocol_name.lower().replace(' ', '_')}_groups_analyzer.py files/{protocol_name.lower().replace(' ', '_')}_full_graph_90_days.json")
        sys.exit(1)
    
    graph_file_path = sys.argv[1]
    
    # Verify file exists
    if not Path(graph_file_path).exists():
        print(f"❌ Graph file not found: {graph_file_path}")
        sys.exit(1)
    
    return graph_file_path

def validate_file_exists(file_path: str, file_description: str = "File"):
    """
    Validate that a file exists
    
    Args:
        file_path (str): Path to the file to check
        file_description (str): Description of the file for error messages
        
    Returns:
        bool: True if file exists, False otherwise
    """
    if not Path(file_path).exists():
        print(f"❌ {file_description} not found: {file_path}")
        return False
    return True

def show_usage(script_name: str, protocol_name: str):
    """
    Show usage information for the script
    
    Args:
        script_name (str): Name of the script
        protocol_name (str): Name of the protocol
    """
    protocol_file_name = protocol_name.lower().replace(' ', '_')
    print(f"❌ Usage: python {script_name} <graph_file_path>")
    print(f"   Example: python {script_name} files/{protocol_file_name}_full_graph_90_days.json")

def parse_command_line_args(protocol_name: str):
    """
    Complete command line argument parsing workflow
    
    Combines startup info printing and graph file argument parsing.
    
    Args:
        protocol_name (str): Name of the protocol being analyzed
        
    Returns:
        str: Path to the graph file
        
    Exits:
        System exit with code 1 if arguments are invalid or file doesn't exist
    """
    print_startup_info(protocol_name)
    return parse_graph_file_argument(protocol_name)
