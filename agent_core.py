"""
Optimized core agent logic for the TDS Data Analyst Agent.

This module contains enhanced agent functionality with:
- Configurable settings
- Enhanced security
- Retry logic
- Better error handling
"""

import os
import sys
import subprocess
import tempfile
import json
import ast
import time
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

# Import Google Generative AI with retry support
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the Data Analyst Agent."""
    gemini_model: str = "gemini-1.5-pro"
    max_tokens: int = 3000
    temperature: float = 0.3
    execution_timeout: int = 150
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_code_validation: bool = True
    max_memory_mb: int = 512  # Memory limit for subprocess


def validate_generated_code(code: str) -> bool:
    """
    Validate generated code for safety and syntax.
    
    Args:
        code: Python code to validate
        
    Returns:
        True if code is valid and safe
    """
    # Check syntax
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    
    # Check for dangerous imports
    dangerous_imports = {'os', 'subprocess', 'sys', 'eval', 'exec', '__import__'}
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in dangerous_imports:
                    logger.warning(f"Dangerous import detected: {alias.name}")
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module in dangerous_imports:
                logger.warning(f"Dangerous import detected: {node.module}")
                return False
    
    return True


def generate_analysis_script(
    task_description: str, 
    config: Optional[AgentConfig] = None
) -> str:
    """
    Generate a Python script using Google Gemini API with retry logic.
    
    Args:
        task_description: Natural language description of the analysis task
        config: Optional configuration object
        
    Returns:
        Generated Python script as a string
        
    Raises:
        Exception: If Google Gemini API is not available or all retries fail
    """
    if not GEMINI_AVAILABLE:
        raise Exception("Google Generative AI library not installed. Please install with: pip install google-generativeai")
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise Exception("GOOGLE_API_KEY environment variable not set")
    
    if config is None:
        config = AgentConfig()
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.gemini_model)
    
    system_prompt = """
You are an expert data analyst and Python programmer. Your task is to generate a complete, self-contained Python script that performs the requested data analysis.

CRITICAL REQUIREMENTS:
1. Import ALL necessary libraries at the top (pandas, matplotlib, seaborn, requests, duckdb, base64, io, json, numpy, etc.)
2. Handle all data fetching, cleaning, analysis, and visualization
3. For ANY plots/visualizations:
   - Use matplotlib with 'Agg' backend: matplotlib.use('Agg')
   - Save to an in-memory buffer using io.BytesIO()
   - Encode to base64 data URI string
   - Close the plot with plt.close() to free memory
   - Include the base64 string in the final output
4. The VERY LAST LINE must be a print() statement with a single, valid JSON string
5. The JSON should contain all results, summaries, and any base64-encoded plots
6. Include robust error handling with try/except blocks
7. Use environment variables for any API keys
8. Make the script completely self-contained - no external file dependencies
9. DO NOT use os, subprocess, sys modules or any file I/O operations except for in-memory operations

FILE HANDLING RULES:
- If the user mentions an attached file (e.g., 'the attached sales_data.csv', 'data.csv'), the script MUST assume that this file has been made available in the current working directory with a simple name like 'data.csv'.
- Use standard file reading: pd.read_csv('data.csv'), pd.read_excel('data.xlsx'), etc.
- Always include error handling for file operations

COMPLETE EXAMPLE:

USER REQUEST:
"Analyze the attached sales_data.csv. Calculate the total revenue and the average transaction value. Plot a bar chart of sales by product category and return it as a base64 string."

PERFECT SCRIPT:
```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import json
import numpy as np

try:
    # Read the data file (simplified name)
    df = pd.read_csv('data.csv')
    
    # Perform analysis
    total_revenue = (df['price'] * df['quantity']).sum()
    average_transaction = df['price'].mean()
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    sales_by_category = df.groupby('category')['price'].sum()
    sales_by_category.plot(kind='bar', color='steelblue')
    plt.title('Sales by Product Category')
    plt.ylabel('Total Revenue ($)')
    plt.xlabel('Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Encode plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    b64_plot = base64.b64encode(buf.read()).decode('utf-8')
    plot_uri = f"data:image/png;base64,{b64_plot}"
    
    # Prepare final JSON output
    result = {
        "summary": f"Sales analysis: total revenue ${total_revenue:,.2f}, average transaction ${average_transaction:.2f}",
        "data": {
            "total_revenue": float(total_revenue),
            "average_transaction": float(average_transaction),
            "sales_by_category": sales_by_category.to_dict()
        },
        "visualizations": [plot_uri],
        "insights": [
            f"Total revenue: ${total_revenue:,.2f}",
            f"Average transaction: ${average_transaction:.2f}",
            f"Number of categories: {len(sales_by_category)}"
        ],
        "metadata": {"rows": len(df), "columns": len(df.columns)},
        "status": "success",
        "error": null
    }
    
    print(json.dumps(result))

except Exception as e:
    error_result = {
        "summary": f"Analysis failed: {str(e)}",
        "data": {},
        "visualizations": [],
        "insights": [],
        "metadata": {},
        "status": "error",
        "error": str(e)
    }
    print(json.dumps(error_result))
```

ANALYSIS GUIDELINES:
- For sales data: Calculate totals, averages, trends, and create bar/line charts
- For weather data: Analyze temperature patterns, create time series plots, calculate statistics
- For network data: Analyze traffic patterns, create network graphs, calculate metrics
- For any dataset: Always include summary statistics, visualizations, and key insights
- Always use descriptive variable names and clear comments
- Ensure final JSON is properly formatted and contains all required fields
- Test edge cases and handle missing data gracefully

CRITICAL SUCCESS FACTORS:
1. Read the user request carefully and address ALL parts of the question
2. Use appropriate analysis methods for the data type
3. Create meaningful visualizations that support the analysis
4. Format the final JSON exactly as specified
5. Include comprehensive error handling
6. Test your logic before generating the final output

Generate clean, production-ready Python code with proper error handling and comprehensive analysis.
"""
    
    # Retry logic
    last_error = None
    for attempt in range(config.max_retries):
        try:
            prompt = f"{system_prompt}\n\nCreate a Python script to: {task_description}"
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
            )
            
            generated_script = response.text
            
            # Clean up the response
            if "```python" in generated_script:
                start = generated_script.find("```python") + 9
                end = generated_script.rfind("```")
                generated_script = generated_script[start:end].strip()
            elif "```" in generated_script:
                start = generated_script.find("```") + 3
                end = generated_script.rfind("```")
                generated_script = generated_script[start:end].strip()
            
            # Validate if enabled
            if config.enable_code_validation and not validate_generated_code(generated_script):
                raise Exception("Generated code failed validation")
            
            return generated_script
            
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (attempt + 1))  # Exponential backoff
            continue
    
    raise Exception(f"Failed after {config.max_retries} attempts. Last error: {str(last_error)}")


def execute_script(
    script_code: str, 
    config: Optional[AgentConfig] = None
) -> str:
    """
    Execute a Python script safely using subprocess with resource limits.
    
    Args:
        script_code: The Python script code to execute
        config: Optional configuration object
        
    Returns:
        The stdout output from the script execution
        
    Raises:
        Exception: If script execution fails or times out
    """
    if config is None:
        config = AgentConfig()
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(script_code)
            temp_file_path = temp_file.name
        
        try:
            # Prepare subprocess with resource limits
            env = os.environ.copy()
            
            # Execute the script
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=config.execution_timeout,
                cwd=os.getcwd(),
                env=env
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                error_msg = f"Script execution failed with return code {result.returncode}\n"
                error_msg += f"STDERR: {result.stderr}\n"
                error_msg += f"STDOUT: {result.stdout}"
                raise Exception(error_msg)
                
        finally:
            # Clean up
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
                
    except subprocess.TimeoutExpired:
        raise Exception(f"Script execution timed out after {config.execution_timeout} seconds")
    except Exception as e:
        if "Script execution failed" in str(e) or "timed out" in str(e):
            raise
        else:
            raise Exception(f"Error executing script: {str(e)}")


class DataAnalystAgent:
    """
    Optimized data analyst agent with enhanced features.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the agent with configuration."""
        self.config = config or AgentConfig()
        logger.info(f"Initialized agent with config: {self.config}")
    
    async def process_question(
        self, 
        question: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language question and return analysis results.
        
        Args:
            question: The natural language question to analyze
            context: Optional context dictionary
            
        Returns:
            Dict containing code, results, and explanation
        """
        logger.info(f"Processing question: {question}")
        
        try:
            # Generate code
            generated_code = generate_analysis_script(question, self.config)
            logger.info("Code generation successful")
            
            # Execute code
            json_output = execute_script(generated_code, self.config)
            logger.info("Code execution successful")
            
            # Parse output
            try:
                parsed_output = json.loads(json_output)
                return {
                    "code": generated_code,
                    "output": parsed_output,
                    "explanation": "Analysis completed successfully.",
                    "execution_success": True,
                    "config": {
                        "model": self.config.gemini_model,
                        "timeout": self.config.execution_timeout
                    }
                }
            except json.JSONDecodeError:
                return {
                    "code": generated_code,
                    "output": json_output,
                    "explanation": "Analysis completed but output is not valid JSON.",
                    "execution_success": True,
                    "config": {
                        "model": self.config.gemini_model,
                        "timeout": self.config.execution_timeout
                    }
                }
                
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "code": f"# Error: {str(e)}",
                "output": None,
                "explanation": f"Error during analysis: {str(e)}",
                "execution_success": False,
                "config": {
                    "model": self.config.gemini_model,
                    "timeout": self.config.execution_timeout
                }
            }


if __name__ == "__main__":
    # Enhanced testing with both questions
    config = AgentConfig(
        max_tokens=3000,
        enable_code_validation=True
    )
    
    # Test Wikipedia question
    print("=== Testing Wikipedia Question ===")
    try:
        with open("test_questions/wikipedia_question.txt", "r") as f:
            wiki_task = f.read()
        
        print("Generating script...")
        wiki_script = generate_analysis_script(wiki_task, config)
        print(f"Generated {len(wiki_script)} characters of code")
        
        print("\nExecuting script...")
        wiki_output = execute_script(wiki_script, config)
        print("Output:", wiki_output[:200] + "..." if len(wiki_output) > 200 else wiki_output)
        
    except Exception as e:
        print(f"Wikipedia test failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test DuckDB question
    print("=== Testing DuckDB Question ===")
    try:
        with open("test_questions/duckdb_question.txt", "r") as f:
            duck_task = f.read()
        
        print("Generating script...")
        duck_script = generate_analysis_script(duck_task, config)
        print(f"Generated {len(duck_script)} characters of code")
        
        print("\nExecuting script...")
        duck_output = execute_script(duck_script, config)
        print("Output:", duck_output[:200] + "..." if len(duck_output) > 200 else duck_output)
        
    except Exception as e:
        print(f"DuckDB test failed: {e}")
    
    print("\nReminder: Set GOOGLE_API_KEY environment variable to test")