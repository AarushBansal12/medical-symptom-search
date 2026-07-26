import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';
import styles from './SearchBar.module.css';

export default function SearchBar({ onSearch, isLoading }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query);
    }
  };

  return (
    <form className={styles.searchForm} onSubmit={handleSubmit}>
      <div className={styles.inputWrapper}>
        <Search className={styles.searchIcon} size={24} />
        <input
          type="text"
          className={styles.searchInput}
          placeholder="Describe your symptoms (e.g., severe headache and fever)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button 
          type="submit" 
          className={styles.searchButton}
          disabled={isLoading || !query.trim()}
        >
          {isLoading ? <Loader2 className={styles.spinner} size={20} /> : 'Search'}
        </button>
      </div>
    </form>
  );
}
