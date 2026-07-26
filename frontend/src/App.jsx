import React, { useState, useEffect } from 'react';
import HeroSection from './components/HeroSection';
import SearchBar from './components/SearchBar';
import Filters from './components/Filters';
import ResultCard from './components/ResultCard';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [categories, setCategories] = useState([]);
  const [severities, setSeverities] = useState([]);
  
  const [query, setQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [selectedSeverity, setSelectedSeverity] = useState('');
  
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch categories and severities on mount
    const fetchMetadata = async () => {
      try {
        const [catRes, sevRes] = await Promise.all([
          fetch(`${API_BASE_URL}/categories`),
          fetch(`${API_BASE_URL}/severities`)
        ]);
        
        if (catRes.ok) {
          const catData = await catRes.json();
          setCategories(catData.categories || []);
        }
        
        if (sevRes.ok) {
          const sevData = await sevRes.json();
          setSeverities(sevData.severities || []);
        }
      } catch (err) {
        console.error('Failed to fetch metadata:', err);
      }
    };
    
    fetchMetadata();
  }, []);

  const handleSearch = async (searchQuery = query) => {
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    setError(null);
    setQuery(searchQuery);
    setHasSearched(true);
    
    try {
      const params = new URLSearchParams({
        q: searchQuery,
        limit: 10
      });
      
      if (selectedCategory) params.append('category', selectedCategory);
      if (selectedSeverity) params.append('severity', selectedSeverity);
      
      const res = await fetch(`${API_BASE_URL}/search?${params.toString()}`);
      if (!res.ok) throw new Error('Failed to fetch results');
      
      const data = await res.json();
      setResults(data.results || []);
    } catch (err) {
      console.error(err);
      setError('An error occurred while searching. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Re-run search if filters change and we already have a query
  useEffect(() => {
    if (hasSearched && query) {
      handleSearch(query);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCategory, selectedSeverity]);

  return (
    <main>
      <HeroSection>
        <SearchBar onSearch={handleSearch} isLoading={isLoading} />
        <Filters 
          categories={categories}
          severities={severities}
          selectedCategory={selectedCategory}
          setSelectedCategory={setSelectedCategory}
          selectedSeverity={selectedSeverity}
          setSelectedSeverity={setSelectedSeverity}
        />
      </HeroSection>

      {hasSearched && (
        <section className="results-section" style={{ 
          maxWidth: '800px', 
          margin: '0 auto', 
          padding: '0 20px 80px',
          display: 'flex',
          flexDirection: 'column',
          gap: '24px'
        }}>
          {error ? (
            <div style={{ color: 'var(--severity-emergency)', textAlign: 'center', padding: '40px' }}>
              {error}
            </div>
          ) : results.length > 0 ? (
            results.map((result, index) => (
              <ResultCard key={result.id} result={result} index={index} />
            ))
          ) : !isLoading ? (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '60px 20px' }}>
              <p style={{ fontSize: '1.2rem', marginBottom: '8px' }}>No matching results found.</p>
              <p>Try adjusting your filters or refining your search query.</p>
            </div>
          ) : null}
        </section>
      )}
    </main>
  );
}

export default App;
