export class SemanticScholarAPI {
    async searchPapers(query) {
        try {
            const response = await fetch(`https://api.semanticscholar.org/graph/v1/paper/search?query=${encodeURIComponent(query)}&fields=title,abstract&limit=10`);
            if (!response.ok) {
                throw new Error('Search failed');
            }
            const data = await response.json();
            return data.data;
        } catch (error) {
            console.error('API error:', error);
            throw error;
        }
    }
}