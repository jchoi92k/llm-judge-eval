# evaluation_pipeline/openai_client.py

import os
import time
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import openai
import tiktoken
from dotenv import load_dotenv, find_dotenv

from .config import Config


class OpenAIClient:
    """
    Wrapper for OpenAI API with automatic retries, cost estimation, and batch processing.
    
    Args:
        config: Configuration object containing model settings and pricing
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model_name = config.model.model_name
        
        # Load environment variables and validate API key
        load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please create a .env file with your API key."
            )
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        
        # Initialize tokenizer for cost calculation
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback for newer models
            self.tokenizer = tiktoken.get_encoding("o200k_base")
        
        # Use run_id logger to tie logs to specific run
        self.logger = logging.getLogger(self.config.run_id)
    
    # ========================================================================
    # DIRECT API CALLS
    # ========================================================================
    
    def call(self, prompt: Any, service_tier: str = "flex", retries: Optional[int] = None, timeout: Optional[float] = None) -> Any:
        """
        Call OpenAI API with automatic retry logic.
        
        Args:
            prompt: The prompt to send
            retries: Number of retry attempts (defaults to config value)
            timeout: Request timeout in seconds (defaults to config value)
            
        Returns:
            OpenAI response object
        """
        retries = retries or self.config.api_settings.max_retries
        timeout = timeout or self.config.api_settings.timeout
        retry_delay = self.config.api_settings.retry_delay
        
        for attempt in range(retries):
            try:
                response = self.client.with_options(timeout=timeout).responses.create(
                    model=self.model_name,
                    input=prompt,
                    service_tier=service_tier,
                )
                return response
            except Exception as e:
                self.logger.error(f"OpenAI API error on attempt {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"Failed to call OpenAI API after {retries} attempts: {e}")

    def create_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Create embedding with configurable rate limit delay."""
        response = self.client.embeddings.create(
            input=text,
            model=model
        )
        time.sleep(self.config.api_settings.embedding_delay)
        return response.data[0].embedding
    
    # ========================================================================
    # COST CALCULATION
    # ========================================================================
    
    def estimate_cost(
        self, 
        prompt_cached: str, 
        prompt_uncached: str, 
        expected_output_tokens: Optional[int] = None
    ) -> float:
        """
        Estimate the cost of an API call based on token counts.
        
        Args:
            prompt_cached: Portion of prompt that will be cached
            prompt_uncached: Portion of prompt that won't be cached
            expected_output_tokens: Expected number of output tokens
            
        Returns:
            Estimated cost in dollars
        """
        # Get token prices from config
        cached_price = self.config.model.cached_token_price
        uncached_price = self.config.model.input_token_price
        output_price = self.config.model.output_token_price if expected_output_tokens else 0
        
        # Count tokens
        cached_tokens = len(self.tokenizer.encode(prompt_cached))
        uncached_tokens = len(self.tokenizer.encode(prompt_uncached))
        
        # Calculate costs
        input_cost = (cached_tokens * cached_price) + (uncached_tokens * uncached_price)
        output_cost = expected_output_tokens * output_price if expected_output_tokens else 0
        total_cost = input_cost + output_cost
        
        return total_cost
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    def upload_batch_file(self, file_path: Path) -> str:
        """
        Upload a batch processing file to OpenAI.
        
        Args:
            file_path: Path to JSONL batch file
            
        Returns:
            Batch ID string
            
        Raises:
            FileNotFoundError: If batch file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Batch file not found: {file_path}")
        
        # Upload file
        file_response = self.client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )
        
        batch_input_file_id = file_response.id
        self.logger.info(f"Batch file uploaded with ID: {batch_input_file_id}")
        
        # Create batch job
        batch_response = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        
        batch_id = batch_response.id
        self.logger.info(f"Batch job created with ID: {batch_id}")
        
        return batch_id
    
    def check_batch_status(self, batch_id: str) -> str:
        """
        Check the status of a batch job.
        
        Args:
            batch_id: Batch job ID
            
        Returns:
            Status string ("validating", "in_progress", "completed", "failed", "cancelled")
        """
        batch = self.client.batches.retrieve(batch_id)
        return batch.status
    
    def retrieve_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve results from a completed batch job.
        
        Args:
            batch_id: Batch job ID
            
        Returns:
            List of result dictionaries
            
        Raises:
            ValueError: If batch is not completed
        """
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            raise ValueError(f"Batch is not completed. Current status: {batch.status}")
        
        # Download results
        file_response = self.client.files.content(batch.output_file_id)
        results = [
            json.loads(line) 
            for line in file_response.text.strip().split('\n') 
            if line
        ]
        
        self.logger.info(f"Retrieved {len(results)} results from batch {batch_id}")
        return results
    
    def wait_for_batch(
        self, 
        batch_id: str, 
        check_interval: int = 60, 
        max_wait_time: Optional[int] = None
    ) -> str:
        """
        Wait for a batch job to complete, checking status periodically.
        
        Args:
            batch_id: Batch job ID
            check_interval: Seconds between status checks
            max_wait_time: Maximum time to wait in seconds (None = unlimited)
            
        Returns:
            Final status string
        """
        start_time = time.time()
        
        while True:
            status = self.check_batch_status(batch_id)
            self.logger.info(f"Batch {batch_id} status: {status}")
            
            if status in ["completed", "failed", "cancelled"]:
                return status
            
            # Check if we've exceeded max wait time
            if max_wait_time and (time.time() - start_time) > max_wait_time:
                self.logger.warning(f"Max wait time exceeded for batch {batch_id}")
                return status
            
            # Wait before checking again
            time.sleep(check_interval)
    
    def cancel_batch(self, batch_id: str) -> None:
        """
        Cancel a batch job.
        
        Args:
            batch_id: Batch job ID
        """
        self.client.batches.cancel(batch_id)
        self.logger.info(f"Batch {batch_id} has been cancelled")