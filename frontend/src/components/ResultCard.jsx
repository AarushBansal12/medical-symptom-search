import React from 'react';
import { Stethoscope, AlertTriangle, Info, ArrowRight } from 'lucide-react';
import styles from './ResultCard.module.css';

const severityColors = {
  'Emergency': 'var(--severity-emergency)',
  'Severe': 'var(--severity-severe)',
  'Chronic-Severe': 'var(--severity-severe)',
  'Moderate-Severe': 'var(--severity-severe)',
  'Moderate': 'var(--severity-moderate)',
  'Mild-Moderate': 'var(--severity-moderate)',
  'Mild-Severe': 'var(--severity-moderate)',
  'Mild': 'var(--severity-mild)',
  'Chronic': 'var(--severity-mild)'
};

export default function ResultCard({ result, index }) {
  const sevColor = severityColors[result.severity] || 'var(--text-muted)';
  
  return (
    <article 
      className={styles.card}
      style={{ animationDelay: `${index * 0.1}s` }}
    >
      <div className={styles.cardHeader}>
        <h3 className={styles.title}>{result.title}</h3>
        <div className={styles.badges}>
          <span className={styles.badgeCategory}>{result.category}</span>
          <span 
            className={styles.badgeSeverity} 
            style={{ 
              borderColor: sevColor, 
              color: sevColor,
              backgroundColor: `${sevColor}15`
            }}
          >
            {result.severity === 'Emergency' && <AlertTriangle size={14} />}
            {result.severity}
          </span>
        </div>
      </div>
      
      <div className={styles.cardBody}>
        {result.symptoms && (
          <div className={styles.section}>
            <h4 className={styles.sectionTitle}>
              <Stethoscope size={16} /> Symptoms
            </h4>
            <p className={styles.sectionContent}>{result.symptoms}</p>
          </div>
        )}
        
        {result.causes && (
          <div className={styles.section}>
            <h4 className={styles.sectionTitle}>
              <Info size={16} /> Common Causes
            </h4>
            <p className={styles.sectionContent}>{result.causes}</p>
          </div>
        )}
        
        {result.what_to_do && (
          <div className={styles.actionBox}>
            <h4 className={styles.actionTitle}>What to do</h4>
            <p className={styles.actionContent}>{result.what_to_do}</p>
          </div>
        )}
      </div>
      
      <div className={styles.cardFooter}>
        <div className={styles.scoreInfo}>
          Match Score: <span className={styles.scoreValue}>{(result.rrf_score * 100).toFixed(1)}</span>
        </div>
        <button className={styles.readMoreBtn}>
          Details <ArrowRight size={16} />
        </button>
      </div>
    </article>
  );
}
