import React from 'react';
import { Filter, AlertCircle } from 'lucide-react';
import styles from './Filters.module.css';

export default function Filters({ categories, severities, selectedCategory, setSelectedCategory, selectedSeverity, setSelectedSeverity }) {
  return (
    <div className={styles.filtersContainer}>
      <div className={styles.filterGroup}>
        <div className={styles.filterLabel}>
          <Filter size={16} />
          <span>Category</span>
        </div>
        <select 
          className={styles.selectInput}
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
        >
          <option value="">All Categories</option>
          {categories.map(cat => (
            <option key={cat} value={cat}>{cat}</option>
          ))}
        </select>
      </div>

      <div className={styles.filterGroup}>
        <div className={styles.filterLabel}>
          <AlertCircle size={16} />
          <span>Severity</span>
        </div>
        <select 
          className={styles.selectInput}
          value={selectedSeverity}
          onChange={(e) => setSelectedSeverity(e.target.value)}
        >
          <option value="">All Severities</option>
          {severities.map(sev => (
            <option key={sev} value={sev}>{sev}</option>
          ))}
        </select>
      </div>
    </div>
  );
}
