import asyncio
import os
import pandas as pd
from openai import AsyncOpenAI
from typing import List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
console = Console()

df = pd.read_parquet("data/yelp_sample_100k.parquet")

async def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text using OpenAI API"""
    embedding = await client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return embedding.data[0].embedding

async def embed_data(df: pd.DataFrame) -> List[List[float]]:
    """Get embeddings for all texts in parallel using asyncio.gather"""
    total_rows = len(df)
    embeddings = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Generating embeddings...", total=total_rows)
        
        # Process in smaller batches to show progress
        batch_size = 100
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            tasks = [get_embedding(row["text"]) for _, row in batch_df.iterrows()]
            batch_embeddings = await asyncio.gather(*tasks)
            embeddings.extend(batch_embeddings)
            
            # Update progress
            progress.update(task, advance=len(batch_embeddings))
    
    return embeddings

# Run async embedding process
embeddings = asyncio.run(embed_data(df))

# Add embeddings to dataframe
df["embedding"] = embeddings

# Save results with status message
with console.status("[green]Saving embeddings to parquet file..."):
    df.to_parquet("data/yelp_sample_100k_embeddings.parquet")
console.print("[green]✓ Embeddings saved successfully!")