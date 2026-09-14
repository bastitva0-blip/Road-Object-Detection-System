/**
 * Road Object Detection Dashboard - Frontend Application
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    
    // Initialize event listeners
    initializeControls();
    updateStats();
    
    // Auto-update stats every 2 seconds
    setInterval(updateStats, 2000);
});

function initializeControls() {
    const recordingToggle = document.getElementById('recording-toggle');
    const heatmapToggle = document.getElementById('heatmap-toggle');
    const confidenceSlider = document.getElementById('confidence');
    
    if (recordingToggle) {
        recordingToggle.addEventListener('change', function() {
            console.log('Recording:', this.checked);
            // Implement recording toggle
        });
    }
    
    if (heatmapToggle) {
        heatmapToggle.addEventListener('change', function() {
            console.log('Heatmap:', this.checked);
            // Implement heatmap toggle
        });
    }
    
    if (confidenceSlider) {
        confidenceSlider.addEventListener('input', function() {
            const value = (this.value / 100).toFixed(2);
            document.getElementById('confidence-value').textContent = value;
            console.log('Confidence threshold:', value);
            // Send to backend
        });
    }
}

async function updateStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        document.getElementById('vehicle-count').textContent = data.vehicle_count || 0;
        document.getElementById('pedestrian-count').textContent = data.pedestrian_count || 0;
        document.getElementById('animal-count').textContent = data.animal_count || 0;
        document.getElementById('alert-count').textContent = data.recent_events.length || 0;
        
        updateAlertsFeed(data.recent_events || []);
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

function updateAlertsFeed(events) {
    const feed = document.getElementById('alerts-feed');
    
    if (events.length === 0) {
        feed.innerHTML = '<div class="alert-empty">No events yet</div>';
        return;
    }
    
    feed.innerHTML = events.slice(-10).reverse().map(event => {
        return `<div class="alert-item">
            <strong>${event.type}</strong>
            <p>${event.description}</p>
            <small>${event.timestamp}</small>
        </div>`;
    }).join('');
}

// WebSocket connection for real-time video
function connectVideoStream() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/video`);
    
    ws.onopen = () => console.log('Video stream connected');
    ws.onclose = () => console.log('Video stream disconnected');
    ws.onerror = (error) => console.error('Video stream error:', error);
}

// Export functions for testing
window.dashboardApp = {
    updateStats,
    initializeControls,
    connectVideoStream
};
