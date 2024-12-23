import { SemanticScholarAPI } from './api.js';

// Constants
const HF_TOKEN = "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";
const clusterColors = [
    '#440154', '#fde725', '#21908C', '#35B779', 
    '#31688E', '#443A83', '#90D743', '#E16462'
];

// State variables
let currentPlotData = null;

// Initialize components
document.addEventListener('DOMContentLoaded', () => {
    const api = new SemanticScholarAPI();
    
    // Auto-resize textarea as content changes
    const searchInput = document.getElementById('searchInput');
    searchInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Add event listeners
    const searchButton = document.getElementById('searchButton');
    if (searchButton) {
        searchButton.addEventListener('click', handleSearch);
    }
});

async function handleSearch() {
    console.log('1. Search function started'); // Add this

    const searchButton = document.getElementById('searchButton');
    const originalButtonText = searchButton.textContent;
    
    try {
        const searchInput = document.getElementById('searchInput').value;
        const limit = parseInt(document.getElementById('papersLimit').value) || 30;
        
        console.log('2. Search input:', searchInput); // Add this
        console.log('3. Paper limit:', limit); // Add this

        searchButton.innerHTML = 'Processing...';
        searchButton.disabled = true;

        const keywords = await generateKeywords(searchInput);
        console.log('4. Keywords generated:', keywords);

        const searchQuery = keywords.join(' ');
        console.log('5. Search query:', searchQuery); // Add this

        const searchResponse = await fetch(
            `https://api.semanticscholar.org/graph/v1/paper/search?query=${encodeURIComponent(searchQuery)}&fields=title,abstract,authors,year,venue,url&limit=${limit}`
        );
        const searchData = await searchResponse.json();
        const papers = searchData.data;
        
        console.log('6. Papers received:', papers?.length); // Add this

        if (!papers?.length) {
            throw new Error('No papers found');
        }

        const embedResponse = await fetch('http://localhost:8000/api/embed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                papers: papers,
                clustering_method: 'hdbscan' // or 'kmeans'
              })
        });
        
        console.log('7. Embed response status:', embedResponse.status); // Add this
        
        if (!embedResponse.ok) {
            throw new Error('Failed to get embeddings');
        }

        const embedData = await embedResponse.json();
        console.log('8. Embed data received:', embedData); // Add this
        
        embedData.papers = papers;
        
        updatePlot(embedData);
        
    } catch (error) {
        console.error('Search error:', error); // Modified this
    } finally {
        searchButton.innerHTML = originalButtonText;
        searchButton.disabled = false;
    }
}

async function generateKeywords(text) {
    try {
        const response = await fetch('http://localhost:8000/api/generate-keywords', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text })
        });
        console.log(response);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log(result);
        return result.keywords;

    } catch (error) {
        console.error('Error generating keywords:', error);
        throw new Error('Failed to generate keywords: ' + error.message);
    }
}

function updatePlot(embedData) {
    console.log('updatePlot called with data:', embedData); // Check if function is called

    currentPlotData = embedData;
    const plotData = create2DPlotData(embedData, embedData.cluster_themes);
    const layout = create2DLayout();

    // Check what data we have
    console.log('Data for evaluation:', {
        hasCoords: !!embedData.coords,
        coordsLength: embedData.coords?.length,
        hasClusters: !!embedData.clusters,
        clustersLength: embedData.clusters?.length
    });

    // Add condition check logging
    // if (embedData.coords && embedData.clusters) {
    //     console.log('Starting silhouette score calculation');
    //     const clusteringScore = calculateSilhouetteScore(embedData.coords, embedData.clusters);
    //     console.log('Silhouette Score calculated:', clusteringScore);
    //     displayEvaluationScores(null, clusteringScore, null);
    // } else {
    //     console.warn('Missing required data for evaluation');
    // }

    Plotly.newPlot('plot', plotData, layout)
        .then(() => {
            console.log('Plot successfully rendered');
            updateClusterList(embedData, embedData.cluster_themes);
        })
        .catch(err => console.error('Error rendering plot:', err));
}

function calculateSilhouetteScore(embeddings, clusters) {
    if (embeddings.length !== clusters.length) {
        throw new Error('Embeddings and clusters arrays must have the same length');
    }

    const uniqueClusters = [...new Set(clusters)];
    let totalScore = 0;
    const n = embeddings.length;

    console.log('Calculating silhouette score with:', {
        embeddingCount: n,
        clusterCount: uniqueClusters.length,
    });

    for (let i = 0; i < n; i++) {
        const clusterI = clusters[i];
        let sameClusterDistances = [];
        let differentClusterDistances = {};

        // Initialize distances for other clusters
        uniqueClusters.forEach(cluster => {
            if (cluster !== clusterI) {
                differentClusterDistances[cluster] = [];
            }
        });

        // Calculate distances
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                const distance = 1 - cosineSimilarity(embeddings[i], embeddings[j]);

                if (clusters[j] === clusterI) {
                    sameClusterDistances.push(distance);
                } else {
                    differentClusterDistances[clusters[j]].push(distance);
                }
            }
        }

        // Mean intra-cluster distance (a)
        const a = sameClusterDistances.length > 0
            ? sameClusterDistances.reduce((sum, d) => sum + d, 0) / sameClusterDistances.length
            : 0;

        // Mean nearest-cluster distance (b)
        let b = Infinity;
        Object.values(differentClusterDistances).forEach(distances => {
            if (distances.length > 0) {
                const avgDistance = distances.reduce((sum, d) => sum + d, 0) / distances.length;
                b = Math.min(b, avgDistance);
            }
        });

        // Silhouette score for this point
        const max_ab = Math.max(a, b);
        const pointScore = max_ab === 0 ? 0 : (b - a) / max_ab;
        totalScore += pointScore;
    }

    return totalScore / n;
}

function displayEvaluationScores(citationScore, clusteringScore, themeScore) {
    // Remove existing evaluation scores if present
    const existingScores = document.querySelector('.evaluation-scores');
    if (existingScores) {
        existingScores.remove();
    }

    const evaluationHtml = `
        <div class="evaluation-scores">
            <h3>Quality Metrics</h3>
            ${clusteringScore !== null ? `
                <div class="score-item">
                    <span>Clustering Quality (Silhouette):</span>
                    <span class="score">${clusteringScore.toFixed(3)}</span>
                </div>
            ` : ''}
        </div>
    `;
    
    // Insert before the plot
    const plotElement = document.getElementById('plot');
    if (plotElement) {
        plotElement.insertAdjacentHTML('beforebegin', evaluationHtml);
    } else {
        document.getElementById('cluster-list').insertAdjacentHTML('beforebegin', evaluationHtml);
    }
}

function cosineSimilarity(vec1, vec2) {
    const dotProduct = vec1.reduce((acc, val, i) => acc + val * vec2[i], 0);
    const norm1 = Math.sqrt(vec1.reduce((acc, val) => acc + val * val, 0));
    const norm2 = Math.sqrt(vec2.reduce((acc, val) => acc + val * val, 0));
    return dotProduct / (norm1 * norm2);
}

function create2DPlotData(embedData, themes) {
    const uniqueClusters = [...new Set(embedData.clusters)];
    return uniqueClusters.map(clusterId => {
        const clusterPoints = embedData.coords.filter((_, index) => 
            embedData.clusters[index] === clusterId
        );
        
        const x = clusterPoints.map(point => point[0]);
        const y = clusterPoints.map(point => point[1]);
        
        // Simplified hover text - just show basic info
        const hoverTexts = embedData.clusters
            .map((cluster, index) => cluster === clusterId ? index : -1)
            .filter(index => index !== -1)
            .map(index => {
                const paper = embedData.papers[index];
                return paper ? `${paper.title}` : '';
            });
    
        // Get theme for this cluster
        const clusterTheme = themes[clusterId] || `Cluster ${clusterId}`;
    
        return {
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            name: clusterTheme,  // This will show in the legend
            text: hoverTexts,
            hoverinfo: 'text',
            marker: {
                size: 10,
                opacity: 0.7
            }
        };
    });
}

function create2DLayout() {
    return {
        title: 'Paper Clusters',
        showlegend: true,
        hovermode: 'closest',
        xaxis: {
            title: 'Dimension 1',
            zeroline: false,
            showgrid: true,
            gridcolor: '#e0e0e0',
            autorange: true  // Enable autoscaling
        },
        yaxis: {
            title: 'Dimension 2',
            zeroline: false,
            showgrid: true,
            gridcolor: '#e0e0e0',
            autorange: true  // Enable autoscaling
        },
        legend: {
            x: 1,
            xanchor: 'right',
            y: 1
        },
        margin: {
            l: 50,
            r: 50,
            b: 50,
            t: 50,
            pad: 4
        },
        height: 600, // Match the CSS height
        // Make plot responsive
        responsive: true,
        autosize: true
    };
}

function updateClusterList(embedData, themes) {
    const clusterList = document.getElementById('cluster-list');
    clusterList.innerHTML = '';

    const containerDiv = document.createElement('div');
    containerDiv.style.display = 'flex';
    containerDiv.style.gap = '20px';

    const uniqueClusters = [...new Set(embedData.clusters)];
    
    uniqueClusters.forEach((clusterId, clusterIndex) => {
        if (clusterId === -1) return; // Skip noise points if using HDBSCAN
        
        const clusterPapers = embedData.clusters
            .map((cluster, index) => cluster === clusterId ? index : -1)
            .filter(index => index !== -1)
            .map(index => ({
                paper: embedData.papers[index],
                index: index
            }));

        const clusterDiv = document.createElement('div');
        clusterDiv.className = 'cluster-column';
        clusterDiv.style.flex = '1';

        // Create themed header with the actual theme from the API
        const headerDiv = document.createElement('div');
        headerDiv.className = 'cluster-header';
        // Directly use the theme from the API response
        headerDiv.textContent = embedData.cluster_themes[clusterId];
        headerDiv.style.backgroundColor = clusterColors[clusterIndex % clusterColors.length];
        headerDiv.style.color = 'white';
        headerDiv.style.padding = '10px';
        headerDiv.style.borderRadius = '5px';
        headerDiv.style.marginBottom = '10px';
        clusterDiv.appendChild(headerDiv);

        // Add papers
        clusterPapers.forEach(({ paper, index }) => {
            const paperDiv = document.createElement('div');
            paperDiv.className = 'paper-item';
            
            const title = paper.title || 'No title';
            const authors = paper.authors ? paper.authors.map(author => author.name).join(', ') : 'No authors';
            const venue = paper.venue || 'No venue';
            const year = paper.year || 'No year';
            const url = paper.url || '#';

            paperDiv.innerHTML = `
                <div class="paper-header">
                    <span class="paper-number">${index + 1}.</span>
                    <a href="${url}" target="_blank" class="paper-title">${title}</a>
                </div>
                <div class="paper-metadata">
                    <div class="authors">${authors}</div>
                    <div class="venue-year">
                        <span class="venue">${venue}</span>
                        <span class="year">${year}</span>
                    </div>
                </div>
            `;
            
            clusterDiv.appendChild(paperDiv);
        });

        containerDiv.appendChild(clusterDiv);
    });

    clusterList.appendChild(containerDiv);
}

function createClusterDiv(clusterId, clusterIndex, paperGroup, themes) {
    const clusterDiv = document.createElement('div');
    clusterDiv.className = 'cluster-column';
    clusterDiv.style.flex = '1';

    const headerDiv = document.createElement('div');
    headerDiv.className = 'cluster-header';
    // Use the theme if available, otherwise fall back to cluster number
    headerDiv.textContent = themes[clusterId] || `Cluster ${parseInt(clusterId) + 1}`;
    headerDiv.style.backgroundColor = clusterColors[clusterIndex % clusterColors.length];
    headerDiv.style.color = 'white';
    headerDiv.style.padding = '10px';
    headerDiv.style.borderRadius = '5px';
    headerDiv.style.marginBottom = '10px';
    clusterDiv.appendChild(headerDiv);

    paperGroup.forEach(({ paper, index }) => {
        const paperDiv = createPaperDiv(paper, index);
        clusterDiv.appendChild(paperDiv);
    });

    return clusterDiv;
}

function createPaperDiv(paper, index) {
    const paperDiv = document.createElement('div');
    paperDiv.className = 'paper-item';
    
    const title = paper.title || 'No title';
    const authors = paper.authors ? paper.authors.map(author => author.name).join(', ') : 'No authors';
    const venue = paper.venue || 'No venue';
    const year = paper.year || 'No year';
    const url = paper.url || '#';

    paperDiv.innerHTML = `
        <div class="paper-header">
            <span class="paper-number">${index + 1}.</span>
            <a href="${url}" target="_blank" class="paper-title">${title}</a>
        </div>
        <div class="paper-metadata">
            <div class="authors">${authors}</div>
            <div class="venue-year">
                <span class="venue">${venue}</span>
                <span class="year">${year}</span>
            </div>
        </div>
    `;
    
    return paperDiv;
}