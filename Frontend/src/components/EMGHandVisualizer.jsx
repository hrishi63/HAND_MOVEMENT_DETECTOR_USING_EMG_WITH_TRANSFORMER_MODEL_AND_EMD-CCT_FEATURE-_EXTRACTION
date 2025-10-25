import React, { useState, useEffect, useRef } from 'react';

export default function EMGMonitor() {
  const [handState, setHandState] = useState('open');
  const [emgSignal, setEmgSignal] = useState(0);
  const [isRunning, setIsRunning] = useState(true);
  const [signalHistory, setSignalHistory] = useState(Array(50).fill(0));
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!isRunning) return;

    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:5000/prediction');
        const data = await response.json();
        const isClenched = data[0];
        setHandState(isClenched ? 'clenched' : 'open');
        const newSignal = isClenched ? 75 + Math.random() * 15 : 15 + Math.random() * 10;
        setEmgSignal(newSignal);
        
        setSignalHistory(prev => [...prev.slice(1), newSignal]);
      } catch (error) {
        console.error('Error fetching prediction:', error);
      }
    };

    const interval = setInterval(fetchData, 150);
    return () => clearInterval(interval);
  }, [isRunning]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.fillStyle = '#1f1f1f';
    ctx.fillRect(0, 0, width, height);
    
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let i = 0; i < height; i += 20) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(width, i);
      ctx.stroke();
    }
    
    ctx.strokeStyle = handState === 'clenched' ? '#ef4444' : '#22c55e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    signalHistory.forEach((value, index) => {
      const x = (index / signalHistory.length) * width;
      const y = height - (value / 100) * height;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    const lastX = width;
    const lastY = height - (signalHistory[signalHistory.length - 1] / 100) * height;
    ctx.fillStyle = handState === 'clenched' ? '#ef4444' : '#22c55e';
    ctx.beginPath();
    ctx.arc(lastX - 5, lastY, 4, 0, Math.PI * 2);
    ctx.fill();
    
  }, [signalHistory, handState]);

  const toggleMonitoring = () => {
    setIsRunning(!isRunning);
  };

  const glowColor = handState === 'clenched' ? '#ef4444' : '#22c55e';
  const boxShadow = isRunning 
    ? `0 0 40px ${handState === 'clenched' ? 'rgba(239, 68, 68, 0.6)' : 'rgba(34, 197, 94, 0.6)'}, 
       0 0 80px ${handState === 'clenched' ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)'}`
    : 'none';

  const styles = {
    container: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
      color: 'white',
      padding: '2rem',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    },
    header: {
      textAlign: 'center',
      marginBottom: '2rem'
    },
    title: {
      fontSize: '3rem',
      fontWeight: 'bold',
      background: 'linear-gradient(to right, #60a5fa, #a78bfa)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      marginBottom: '0.5rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '1rem'
    },
    icon: {
      fontSize: '2rem'
    },
    buttonContainer: {
      display: 'flex',
      justifyContent: 'center',
      marginBottom: '2rem'
    },
    button: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      padding: '0.75rem 1.5rem',
      borderRadius: '0.5rem',
      fontSize: '1.125rem',
      fontWeight: '600',
      border: 'none',
      cursor: 'pointer',
      transition: 'all 0.3s',
      backgroundColor: isRunning ? '#dc2626' : '#16a34a'
    },
    topSection: {
      maxWidth: '1200px',
      margin: '0 auto',
      display: 'grid',
      gridTemplateColumns: '1fr 1fr 1.5fr',
      gap: '1.5rem',
      marginBottom: '2rem'
    },
    card: {
      backgroundColor: '#1f1f1f',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '1px solid #333'
    },
    cardTitle: {
      color: '#9ca3af',
      fontSize: '0.875rem',
      fontWeight: '500',
      marginBottom: '0.5rem'
    },
    signalValue: {
      fontSize: '2.5rem',
      fontWeight: 'bold',
      color: '#60a5fa',
      display: 'flex',
      alignItems: 'baseline',
      gap: '0.5rem'
    },
    signalUnit: {
      fontSize: '1.125rem',
      color: '#6b7280'
    },
    progressBar: {
      marginTop: '1rem',
      height: '0.5rem',
      backgroundColor: '#374151',
      borderRadius: '9999px',
      overflow: 'hidden'
    },
    progressFill: {
      height: '100%',
      background: 'linear-gradient(to right, #22c55e, #eab308, #ef4444)',
      transition: 'width 0.3s',
      width: `${emgSignal}%`
    },
    stateText: {
      fontSize: '2.5rem',
      fontWeight: 'bold',
      color: handState === 'clenched' ? '#f87171' : '#4ade80'
    },
    stateSubtext: {
      color: '#9ca3af',
      fontSize: '0.875rem',
      marginTop: '0.5rem'
    },
    handBox: {
      backgroundColor: '#1f1f1f',
      borderRadius: '1rem',
      padding: '2rem',
      border: `2px solid ${glowColor}`,
      boxShadow: boxShadow,
      transition: 'all 0.3s',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center'
    },
    handEmoji: {
      fontSize: '6rem',
      animation: 'pulse 2s infinite'
    },
    visualizerSection: {
      maxWidth: '1200px',
      margin: '0 auto',
      marginBottom: '2rem'
    },
    visualizerCard: {
      backgroundColor: '#1f1f1f',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '1px solid #333'
    },
    canvas: {
      width: '100%',
      height: '150px',
      borderRadius: '0.5rem'
    },
    footer: {
      textAlign: 'center',
      marginTop: '2rem',
      color: '#9ca3af',
      fontSize: '0.875rem'
    },
    code: {
      color: '#60a5fa',
      backgroundColor: '#1f1f1f',
      padding: '0.125rem 0.375rem',
      borderRadius: '0.25rem'
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>
          <span style={styles.icon}>⚡</span>
          EMG Hand Monitor
        </h1>
      </div>

      <div style={styles.buttonContainer}>
        <button
          onClick={toggleMonitoring}
          style={styles.button}
          onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
          onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
        >
          <span>{isRunning ? '⏹' : '▶'}</span>
          {isRunning ? 'Stop Monitoring' : 'Start Monitoring'}
        </button>
      </div>

      {/* Top Section: EMG Signal + Hand State + Hand Visual */}
      <div style={styles.topSection}>
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>EMG Signal</h3>
          <div style={styles.signalValue}>
            <span>{emgSignal.toFixed(1)}</span>
            <span style={styles.signalUnit}>μV</span>
          </div>
          <div style={styles.progressBar}>
            <div style={styles.progressFill} />
          </div>
        </div>

        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Hand State</h3>
          <span style={styles.stateText}>
            {handState === 'clenched' ? 'CLENCHED' : 'OPEN'}
          </span>
          <p style={styles.stateSubtext}>
            {isRunning ? 'Live from Flask' : 'Monitoring paused'}
          </p>
        </div>

        <div style={styles.handBox}>
          <div style={styles.handEmoji}>
            {handState === 'clenched' ? '✊' : '✋'}
          </div>
        </div>
      </div>

      {/* Signal Visualizer */}
      <div style={styles.visualizerSection}>
        <div style={styles.visualizerCard}>
          <h3 style={styles.cardTitle}>Live Signal Waveform</h3>
          <canvas 
            ref={canvasRef} 
            width={1100} 
            height={150}
            style={styles.canvas}
          />
        </div>
      </div>

      <div style={styles.footer}>
        {isRunning ? (
          <>
            Live data updates every 150 ms from Flask endpoint{' '}
            <code style={styles.code}>/prediction</code>
          </>
        ) : (
          <span style={{ color: '#eab308' }}>⏸ Monitoring paused - Click Start to resume</span>
        )}
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.8;
          }
        }
      `}</style>
    </div>
  );
}