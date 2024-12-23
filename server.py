from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import logging
import time
import hdbscan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('allenai/specter2_base')
pca = PCA(n_components=2)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
keywordmodel = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
try:
    logger.info("Loading BART model and tokenizer...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    keywordmodel = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

@app.post("/api/generate-keywords")
async def generate_keywords(request: Request):
    try:
        # Get the request data
        data = await request.json()
        text = data.get('text', '')
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
            
        logger.info("Tokenizing input text...")
        # Tokenize
        inputs = tokenizer(
            text, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        )
        
        logger.info("Generating summary...")
        # Generate summary
        summary_ids = keywordmodel.generate(
            inputs.input_ids,
            num_beams=4,
            max_length=130,
            min_length=30,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            num_return_sequences=1
        )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Simple keyword extraction from summary
        words = summary.lower().split()
        # Filter out short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Get unique keywords
        unique_keywords = list(dict.fromkeys(keywords))[:5]
        
        logger.info("Keywords generated successfully")
        return {"keywords": unique_keywords}
        
    except Exception as e:
        logger.error(f"Error in generate_keywords: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_theme(papers: list) -> str:
    try:
        # Validation checks
        if not papers:
            logger.error("Empty papers list provided to generate_theme")
            return "No papers in cluster"
            
        # Take only first 3 papers
        papers = papers[:3]
        
        # Check if papers contain actual text
        papers = [p for p in papers if p and isinstance(p, str) and len(p.strip()) > 0]
        if not papers:
            logger.error("No valid text content in papers")
            return "Empty content"
        
        # Papers are already strings, just join them
        combined_text = " | ".join(papers)
        logger.info(f"Combined text for theme generation: {combined_text[:500]}...")
        
        # Check if combined text is too short
        if len(combined_text.strip()) < 10:
            logger.error(f"Combined text too short: {combined_text}")
            return "Insufficient content"
            
        prompt = "Generate a single complete sentence summarizing this collection of papers: " + combined_text
        
        try:
            inputs = tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            return "Tokenization error"
            
        try:
            theme_ids = keywordmodel.generate(
                inputs.input_ids,
                num_beams=4,
                max_length=50,
                min_length=10,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                num_return_sequences=1,
                early_stopping=True,
            )
        except Exception as e:
            logger.error(f"Model generation error: {str(e)}")
            return "Generation error"
            
        try:
            raw_theme = tokenizer.decode(theme_ids[0], skip_special_tokens=True)
            logger.info(f"Raw theme generated: {raw_theme}")
            
            if not raw_theme or len(raw_theme.strip()) < 5:
                logger.error("Generated theme too short or empty")
                return "Generation failed"
                
            # Cut off at the first period
            theme = raw_theme.split('.')[0].strip().capitalize() + '.'
            
            logger.info(f"Cleaned theme: {theme}")
            logger.info(f"Theme length: {len(theme)}")
            
            # Final validation of theme
            if len(theme) < 10 or theme.lower().startswith('cluster'):
                logger.error(f"Invalid theme generated: {theme}")
                return "Theme generation failed"
                
            return theme
            
        except Exception as e:
            logger.error(f"Theme post-processing error: {str(e)}")
            return "Processing error"
            
    except Exception as e:
        logger.error(f"Critical error in generate_theme: {str(e)}")
        return "Theme generation error"
    
def clean_theme(theme: str) -> str:
    replacements = {
        "this paper": "these papers",
        "this article": "these papers",
        "this study": "these studies",
        "deals with": "explore",
        "looks at": "investigate",
    }
    
    for old, new in replacements.items():
        theme = theme.lower().replace(old, new)
    
    return theme.strip().capitalize()
    
def calculate_clustering_metrics(embeddings, clusters):
    """Calculate detailed clustering metrics"""
    if len(set(clusters)) <= 1:  # Can't calculate scores for 1 or fewer clusters
        return {
            "silhouette_score": 0,
            "cluster_silhouettes": {},
            "cluster_sizes": {0: len(clusters)}
        }
    
    # Calculate overall silhouette score
    sil_score = silhouette_score(embeddings, clusters)
    
    # Calculate per-point silhouette scores
    sample_silhouette_values = silhouette_samples(embeddings, clusters)
    
    # Calculate per-cluster metrics
    cluster_metrics = {}
    unique_clusters = np.unique(clusters)
    
    for cluster in unique_clusters:
        if cluster != -1:  # Skip noise points if using HDBSCAN
            cluster_mask = (clusters == cluster)
            cluster_metrics[int(cluster)] = {
                "size": int(np.sum(cluster_mask)),
                "avg_silhouette": float(np.mean(sample_silhouette_values[cluster_mask])),
                "min_silhouette": float(np.min(sample_silhouette_values[cluster_mask])),
                "max_silhouette": float(np.max(sample_silhouette_values[cluster_mask]))
            }
    
    return {
        "silhouette_score": float(sil_score),
        "cluster_metrics": cluster_metrics
    }

@app.post("/api/embed")
async def create_embeddings(request: Request):
    try:
        data = await request.json()
        papers = data if isinstance(data, list) else data.get('papers', [])
        clustering_method = data.get('clustering_method', 'kmeans') if isinstance(data, dict) else 'kmeans'
        
        if not papers:
            raise HTTPException(status_code=400, detail="No papers provided")
            
        texts = [f"{paper.get('title', '')} {paper.get('abstract', '')}" for paper in papers]
        logger.info(f"Processing {len(papers)} papers for clustering using {clustering_method}")
        
        try:
            embeddings = model.encode(texts)
            logger.info(f"Embeddings shape: {embeddings.shape}")
            
            coords = pca.fit_transform(embeddings)
            logger.info(f"PCA coords shape: {coords.shape}")
            
            # Calculate optimal k based on number of papers
            k = get_optimal_k(len(papers))
            
            if clustering_method == 'hdbscan':
                # Try HDBSCAN first with stricter parameters
                min_cluster_size = max(2, len(papers) // 10)
                logger.info(f"Using HDBSCAN with min_cluster_size={min_cluster_size}")
                
                normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
                cosine_distances = 1 - np.dot(normalized_embeddings, normalized_embeddings.T)
                cosine_distances = np.array(cosine_distances, dtype=np.float64)
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=1,
                    metric='precomputed',
                    cluster_selection_method='eom',
                    prediction_data=True,
                    cluster_selection_epsilon=0.2  # Adjust this to be more aggressive
                )
                
                clusters = clusterer.fit_predict(cosine_distances)
                
                # Check if HDBSCAN produced too many noise points
                noise_ratio = np.sum(clusters == -1) / len(clusters)
                unique_clusters = np.unique(clusters)
                
                # If more than 20% noise points or less than 2 real clusters, fall back to K-means
                if noise_ratio > 0.2 or len(unique_clusters[unique_clusters != -1]) < 2:
                    logger.warning(f"HDBSCAN produced too many noise points ({noise_ratio:.2%}), falling back to K-means")
                    clusterer = KMeans(n_clusters=k, random_state=42)
                    clusters = clusterer.fit_predict(embeddings)
                    clustering_method = 'kmeans'  # Update method to reflect fallback
            else:
                clusterer = KMeans(n_clusters=k, random_state=42)
                clusters = clusterer.fit_predict(embeddings)
            
            logger.info(f"Clustering complete. Unique clusters: {np.unique(clusters)}")
            
            # Generate themes for clusters (excluding noise points)
            cluster_themes = {}
            for cluster_id in np.unique(clusters):
                if cluster_id != -1:  # Skip noise points
                    cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
                    if cluster_texts:
                        logger.info(f"Generating theme for cluster {cluster_id} with {len(cluster_texts)} papers")
                        theme = generate_theme(cluster_texts)
                        cluster_themes[int(cluster_id)] = theme
            
            return {
                "coords": coords.tolist(),
                "clusters": clusters.tolist(),
                "num_clusters": len(np.unique(clusters[clusters != -1])),
                "cluster_themes": cluster_themes,
                "clustering_metrics": {
                    "method_used": clustering_method,
                    "k_value": k if clustering_method == 'kmeans' else None
                }
            }
            
        except Exception as processing_error:
            logger.error(f"Processing error: {str(processing_error)}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(processing_error)}")
            
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_optimal_k(n_papers):
    """Determine optimal K based on paper count"""
    if n_papers <= 5:
        return 2
    elif n_papers <= 10:
        return 3
    elif n_papers <= 20:
        return 4
    elif n_papers <= 50:
        return 5
    else:
        return min(8, n_papers // 10)  # Adjusted for larger datasets, max 8 clusters
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)