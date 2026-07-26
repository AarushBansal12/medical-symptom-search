import React from 'react';
import { Activity } from 'lucide-react';
import styles from './HeroSection.module.css';

export default function HeroSection({ children }) {
  return (
    <div className={styles.heroContainer}>
      <div className={styles.heroGlow}></div>
      <div className={styles.heroContent}>
        <div className={styles.logoContainer}>
          <div className={styles.logoIcon}>
            <Activity size={32} color="var(--primary)" />
          </div>
          <h1 className={styles.heroTitle}>
            Medi<span className={styles.titleHighlight}>Search</span>
          </h1>
        </div>
        <p className={styles.heroSubtitle}>
          Intelligent symptoms analysis powered by clinical data.
        </p>
        
        {/* Search Bar will be injected here via children */}
        <div className={styles.searchContainer}>
          {children}
        </div>
      </div>
    </div>
  );
}
